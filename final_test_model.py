import numpy as np
import joblib 
import matplotlib.pyplot as plt 
from sklearn.cluster import MeanShift
from skimage.color import label2rgb
from tqdm import tqdm 
from skimage.transform import resize 

# Set up parameters and image paths 
DEPTH_PATH = "pose-depth-masks/arm-out_isolation_masks.npy"
NUM_JOINTS = 24

def extract_features(depth_img, i, j):
    """
    This function extracts features using 5x5 local patches 
    """
    local_patch = depth_img[max(0, i-2):min(depth_img.shape[0], i+3), max(0, j-2):min(depth_img.shape[1], j+3)]
    local_mean = np.mean(local_patch)
    local_std = np.std(local_patch)
    return [depth_img[i, j], i, j, local_mean, local_std]

def load_models():
    """
    This function loads saved job lib files for the upper, lower, and coarse
    random forest classifiers, and each of the joint regressors 

    Returns: course classifier, upper classifier, lower classifier, 
    list of joint regressors 
    """

    dir = "trained_regressors/"
    clf_coarse = joblib.load(dir + 'rf_classifier_coarse.joblib')
    clf_upper = joblib.load(dir + 'rf_classifier_upper.joblib')
    clf_lower = joblib.load(dir + 'rf_classifier_lower.joblib')
     
    # Load joint regressors 
    joint_regressors = []
    for joint_id in range(NUM_JOINTS):
        regr_path = dir + f'rf_regressor_joint{joint_id}.joblib'
        regr = joblib.load(regr_path)
        joint_regressors.append(regr)
    return clf_coarse, clf_upper, clf_lower, joint_regressors

def load_depth_map(): 
    """
    This function loads a deepth image and returns the resized image and its dimensions
    
    Returns: resized depth image, height, width    
    """
    depth = np.load(DEPTH_PATH)
    if depth.ndim == 3: 
        depth_map = depth[0] # Taking first frame 
    else: 
        depth_map = depth 
    resize_shape = (240, 320)
    depth_map_resized = resize(depth_map, resize_shape, preserve_range=True, anti_aliasing=True).astype(depth_map.dtype)
    H, W = depth_map_resized.shape
    return depth_map_resized, H, W

def predict_segm(depth_map_resized, clf_coarse, clf_upper, clf_lower, H, W): 
    pred_segm = np.zeros((H, W), dtype=np.int32) 

    """
    This function predicts the segmentation masks for a given depth image usingothe course, upper, 
    and lower classifiers
    
    Args:
        depth_map_resized: numpy depth image resized to 240, 320
        clf_coarse: coarse random forest classifier 
        clf_upper: upper body random forest classifier
        clf_lower: lower body random forest classifier
        H: height of depth image
        W: width  of depth image 
    
    Returns:  redicted segmentation         
    """    
    valid_pixels = np.where(depth_map_resized != 0) # Ignore the background pixels 

    # Process pixels in batches to avoid memory issues 
    batch_size = 10000
    X_test = []
    for i, j in zip(*valid_pixels):
        X_test = np.array(X_test)
        # Predict coarse labels 
        coarse_preds = clf_coarse.predict(X_test)
        # Predict fine-grained labels based on coarse predictions 
        refined_preds = [] 
        for feat, c in zip(X_test , coarse_preds): 
            if c == 1: # upper body 
                refined_preds.append(clf_upper.predict([feat])[0])
            elif c == 2: # lower body 
                refined_preds.append(clf_lower.predict([feat])[0])
            else: 
                refined_preds.append(0) # background
        pred_segm[valid_pixels[0][len(X_test)-batch_size:len(X_test)], 
                  valid_pixels[1][len(X_test)-batch_size:len(X_test)]] = refined_preds
        X_test = [] 

    # Process remaining pixels
        if len(X_test) > 0: 
            X_test = np.array(X_test)
            coarse_preds = clf_coarse.predict(X_test)
            refined_preds = [] 
            for feat, c in zip(X_test, coarse_preds):
                if c == 1: # upper body 
                    refined_preds.append(clf_upper.predict([feat])[0])
                elif c == 2: # lower body 
                    refined_preds.append(clf_lower.predict([feat])[0])
                else: 
                    refined_preds.append(0) # background
            pred_segm[valid_pixels[0][-len(X_test):], 
                      valid_pixels[1][-len(X_test):]] = refined_preds
            
        pred_segm[depth_map_resized == 0] = 0 # background is set to 0 
        return pred_segm 
    
def predict_joints(pred_segm, joint_regressors, depth_map_resized): 
    """
    This function predicts joints using the predicted segmentation and joint regressors. 

    Parameters: 
        pred_segm - predicted segmentation array 
        joint_regressors - joint regressors 
        depth_map_resized - depth map array resized 

    Returns: 
        joint predictions 
    """ 
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
            # Use Meanshift with adjusted bandwidth based on confidence 
            meanshift = MeanShift(bandwidth=25, bin_seeding=True) # Lower bandwidth for more precise clustering 
            meanshift.fit(test_points)

            if len(meanshift.cluster_centers_) > 0: 
                # Get the cluster with highest total confidence 
                cluster_labels = meanshift.labels_
                best_cluster = 0 
                max_confidence = 0 
                for cluster in range(len(meanshift.cluster_centers_)):
                    cluster_confidence = np.sum(confidences[cluster_labels == cluster])
                    if cluster_confidence > max_confidence:
                        max_confidence = cluster_confidence
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
    """
    This funtction plots predictions using the deptth map and predicted segmentations

    Parameters: 
        depth_map_resized - resized depth map
        pred_segm - predicted segmentation
        joint_preds - joint predictoins 
    
    Returns: 
        Nothing
    """
    plt.figure(figsize=(12,8))
    plt.imshow(depth_map_resized, cmap='gray')
    #plot points with joint numbers
    for i, (x, y) in enumerate(joint_preds):
        if not np.isnan(x) and not np.isnan(y):
            plt.scatter(x, y, c='blue', marker='x', s = 100)
            plt.text(x+5, y+5, str(i), color='red', fontzie=10, fontweight='bold')
    plt.title("Predicted Joints on Input Depth Map", fontsize = 14)
    plt.axis("off")
    plt.savefig('joints_prediction_on_input.png', dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(label2rgb(pred_segm, bg_label=0))
    ax.set_title("Predicted Segmentation", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig('segmentation_pred_on_input.png', dpi = 300, bbox_inches='tight')
    plt.show()

def process_depth_map(depth_map, models = None):
    """
    Process a single depth image to get joint predictions and segmentation. 

    Args: 
        depth_map: numpy array of depth image
        models: tuple of (clf_c, clf_u, clf_l, joint_regressors)

    Returns:
        tuple of (depth_map_resize, pred_segm, joint_preds)
    """

    #load models if not provided
    if models is None: 
        clf_c, clf_u, clf_l, joint_regressors = load_models()
    else:
        clf_c, clf_u, clf_l, joint_regressors = models
    
    #Resize depth map
    resize_shape = (240, 320)
    depth_map_resized = resize(depth_map, resize_shape, preserve_range=True, anti_aliasing=True).astype(depth_map.dtype)
    H, W = depth_map_resized.shape
    
    #get prediction
    pred_segm = predict_segm(depth_map_resized, clf_c, clf_u, clf_l, H, W)
    joint_preds = predict_joints(pred_segm, joint_regressors, depth_map_resized)

    return depth_map_resized, pred_segm, joint_preds

def main(): 
    # load depth map from file 
    depth = np.load(DEPTH_PATH)
    if depth.ndim == 3:
        depth_map = depth[0]
    else:
        depth_map = depth
    
    # process the depth map 
    depth_map_resized, pred_segm, joint_preds = process_depth_map(depth_map)

    # plot results
    plot_preds(depth_map_resized, pred_segm, joint_preds)

if __name__ == "__main__":
    main()