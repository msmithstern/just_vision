import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from skimage.color import label2rgb
from tqdm import tqdm
from skimage.transform import resize

# Set up parameters and image paths.
DEPTH_PATH = "pose-depth-masks/arm-out_isolation_masks.npy"
NUM_JOINTS = 24

def extract_features(depth_img, i, j): 
    local_patch = depth_img[max(0, i-2):min(depth_img.shape[0], i+3), max(0, j-2):min(depth_img.shape[1], j+3)]
    local_mean = np.mean(local_patch)
    local_std = np.std(local_patch)
    return [depth_img[i, j], i, j, local_mean, local_std]

def load_models(): 
    # Load coarse classifier
    dir = "trained_regressors/"
    clf_c = joblib.load(dir + 'rf_classifier_coarse.joblib')
    # Load fine-grained classifiers
    clf_u = joblib.load(dir + 'rf_classifier_upper.joblib')
    clf_l = joblib.load(dir + 'rf_classifier_lower.joblib')
    # Load joint regressors
    joint_regressors = []
    for joint_id in range(NUM_JOINTS):
        regr_path = dir + f'rf_regressor_joint{joint_id}.joblib'
        regr = joblib.load(regr_path)
        joint_regressors.append(regr)
    return clf_c, clf_u, clf_l, joint_regressors

def load_depth_map():
    depth = np.load(DEPTH_PATH)
    if depth.ndim == 3:
        depth_map = depth[0]  # Take first frame
    else:
        depth_map = depth
    resize_shape = (240, 320)
    depth_map_resized = resize(depth_map, resize_shape, preserve_range=True, anti_aliasing=True).astype(depth_map.dtype)
    H, W = depth_map_resized.shape
    return depth_map_resized, H, W

def predict_segm(depth_map_resized, clf_c, clf_u, clf_l, H, W): 
    pred_segm = np.zeros((H, W), dtype=np.int32)
    valid_pixels = np.where(depth_map_resized != 0) # ignore the background pixels
    
    # Process pixels in batches to avoid memory issues
    batch_size = 10000
    X_test = []
    for i, j in zip(*valid_pixels):
        X_test.append(extract_features(depth_map_resized, i, j))
        if len(X_test) >= batch_size:
            X_test = np.array(X_test)
            # First predict coarse labels
            coarse_preds = clf_c.predict(X_test)
            # Then predict fine-grained labels based on coarse predictions
            refined_preds = []
            for feat, c in zip(X_test, coarse_preds):
                if c == 1:  # upper body
                    refined_preds.append(clf_u.predict([feat])[0])
                elif c == 2:  # lower body
                    refined_preds.append(clf_l.predict([feat])[0])
                else:
                    refined_preds.append(0)  # background
            pred_segm[valid_pixels[0][len(X_test)-batch_size:len(X_test)], 
                     valid_pixels[1][len(X_test)-batch_size:len(X_test)]] = refined_preds
            X_test = []
    
    # Process remaining pixels
    if len(X_test) > 0:
        X_test = np.array(X_test)
        coarse_preds = clf_c.predict(X_test)
        refined_preds = []
        for feat, c in zip(X_test, coarse_preds):
            if c == 1:  # upper body
                refined_preds.append(clf_u.predict([feat])[0])
            elif c == 2:  # lower body
                refined_preds.append(clf_l.predict([feat])[0])
            else:
                refined_preds.append(0)  # background
        pred_segm[valid_pixels[0][-len(X_test):], valid_pixels[1][-len(X_test):]] = refined_preds
    
    pred_segm[depth_map_resized == 0] = 0 # background is set to 0 
    return pred_segm

def predict_joints(pred_segm, joint_regressors, depth_map_resized): 
    joint_preds = [] 
    for joint_id in tqdm(range(NUM_JOINTS), desc="Predicting joints per joint"):
        mask = (pred_segm == (joint_id + 1))
        ys, xs = np.where(mask)
        test_points = []
        confidences = []
        
        # Get predictions and their confidences
        for y, x in zip(ys, xs):
            feat = extract_features(depth_map_resized, y, x)
            pred = joint_regressors[joint_id].predict([feat])
            # Calculate confidence based on depth value and local statistics
            depth_val = depth_map_resized[y, x]
            local_patch = depth_map_resized[max(0, y-2):min(depth_map_resized.shape[0], y+3), 
                                          max(0, x-2):min(depth_map_resized.shape[1], x+3)]
            local_std = np.std(local_patch)
            confidence = 1.0 / (1.0 + local_std) if local_std > 0 else 1.0
            test_points.append(pred[0])
            confidences.append(confidence)
        
        test_points = np.array(test_points)
        confidences = np.array(confidences)
        
        if len(test_points) > 0:
            # Use MeanShift with adjusted bandwidth based on confidence
            meanshift = MeanShift(bandwidth=25, bin_seeding=True)  # Reduced bandwidth for more precise clustering
            meanshift.fit(test_points)
            
            if len(meanshift.cluster_centers_) > 0:
                # Get the cluster with highest total confidence
                cluster_labels = meanshift.labels_
                best_cluster = 0
                max_confidence = 0
                for cluster in range(len(meanshift.cluster_centers_)):
                    cluster_conf = np.sum(confidences[cluster_labels == cluster])
                    if cluster_conf > max_confidence:
                        max_confidence = cluster_conf
                        best_cluster = cluster
                joint_pred = meanshift.cluster_centers_[best_cluster]
            else:
                joint_pred = [np.nan, np.nan]
        else:
            joint_pred = [np.nan, np.nan]
        joint_preds.append(joint_pred)
    joint_preds = np.array(joint_preds)
    return joint_preds

def plot_preds(depth_map_resized, pred_segm, joint_preds):
    plt.figure(figsize=(12, 8))
    plt.imshow(depth_map_resized, cmap='gray')
    # Plot points with joint numbers
    for i, (x, y) in enumerate(joint_preds):
        if not np.isnan(x) and not np.isnan(y):  # Only plot if joint was found
            plt.scatter(x, y, c='blue', marker='x', s=100)  # Increased marker size
            plt.text(x + 5, y + 5, str(i), color='red', fontsize=10, fontweight='bold')  # Increased font size and made bold
    plt.title("Predicted Joints on Input Depth Map", fontsize=14)
    plt.axis("off")
    plt.savefig('joints_prediction_on_input.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(label2rgb(pred_segm, bg_label=0))
    ax.set_title("Predicted Segmentation", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig('segmentation_pred_on_input.png', dpi=300, bbox_inches='tight')
    plt.show()

def process_depth_map(depth_map, models=None):
    """
    Process a single depth map to get joint predictions and segmentation.
    
    Args:
        depth_map: numpy array of depth values
        models: tuple of (clf_c, clf_u, clf_l, joint_regressors) if already loaded, None to load new
    
    Returns:
        tuple of (depth_map_resized, pred_segm, joint_preds)
    """
    # Load models if not provided
    if models is None:
        clf_c, clf_u, clf_l, joint_regressors = load_models()
    else:
        clf_c, clf_u, clf_l, joint_regressors = models
    
    # Resize depth map
    resize_shape = (240, 320)
    depth_map_resized = resize(depth_map, resize_shape, preserve_range=True, anti_aliasing=True).astype(depth_map.dtype)
    H, W = depth_map_resized.shape
    
    # Get predictions
    pred_segm = predict_segm(depth_map_resized, clf_c, clf_u, clf_l, H, W)
    joint_preds = predict_joints(pred_segm, joint_regressors, depth_map_resized)
    
    return depth_map_resized, pred_segm, joint_preds

def main():
    # Load depth map from file
    depth = np.load(DEPTH_PATH)
    if depth.ndim == 3:
        depth_map = depth[0]  # Take first frame
    else:
        depth_map = depth
    
    # Process the depth map
    depth_map_resized, pred_segm, joint_preds = process_depth_map(depth_map)
    
    # Plot results
    plot_preds(depth_map_resized, pred_segm, joint_preds)

if __name__ == "__main__":
    main() 