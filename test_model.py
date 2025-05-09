import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from skimage.color import label2rgb
from tqdm import tqdm
from skimage.transform import resize

# Set up parameters and image paths.
DEPTH_PATH = "pose-depth-masks/clap-right_isolation_masks.npy"
RF_CLASSIFIER_PATH = 'rf_classifier2.joblib'
INDEX = 3
NUM_JOINTS = 24

def extract_features(depth_img, i, j): 
    local_patch = depth_img[max(0, i-2):min(depth_img.shape[0], i+3), max(0, j-2):min(depth_img.shape[1], j+3)]
    local_mean = np.mean(local_patch)
    local_std = np.std(local_patch)
    return [depth_img[i, j], i, j, local_mean, local_std]

def load_models(): 
    rf_classifier = joblib.load(RF_CLASSIFIER_PATH)
    joint_regressors = []
    for joint_id in range(NUM_JOINTS):
        regr_path = 'rf_regressor2_joint' + str(joint_id) + '.joblib'
        regr = joblib.load(regr_path)
        joint_regressors.append(regr)
    return rf_classifier, joint_regressors

def load_depth_map():
    depth = np.load(DEPTH_PATH)
    if depth.ndim == 3:
        depth_map = depth[INDEX]
    else:
        depth_map = depth
    resize_shape = (240, 320)
    depth_map_resized = resize(depth_map, resize_shape, preserve_range=True, anti_aliasing=True).astype(depth_map.dtype)
    H, W = depth_map_resized.shape
    return depth_map_resized, H, W

def predict_segm(depth_map_resized, rf_classifier, H, W): 
    pred_segm = np.zeros((H, W), dtype=np.int32)
    valid_pixels = np.where(depth_map_resized != 0) # ignore the background pixels
    X_img = np.array([extract_features(depth_map_resized, i, j) for i, j in zip(*valid_pixels)])
    if len(X_img) > 0:
        pred_labels = rf_classifier.predict(X_img)
        pred_segm[valid_pixels] = pred_labels
    pred_segm[depth_map_resized == 0] = 0 # background is set to 0 
    return pred_segm

def predict_joints(pred_segm, joint_regressors, depth_map_resized): 
    joint_preds = [] 
    for joint_id in tqdm(range(NUM_JOINTS), desc="Predicting joints per joint"):
        mask = (pred_segm == (joint_id + 1))
        ys, xs = np.where(mask)
        test_points = []
        for y, x in zip(ys, xs):
            pred = joint_regressors[joint_id].predict([extract_features(depth_map_resized, y, x)])
            test_points.append(pred[0])
        test_points = np.array(test_points)
        if len(test_points) > 0:
            meanshift = MeanShift(bandwidth=40) # bin seeding works only with a bandwidth of > 60
            meanshift.fit(test_points)
            joint_pred = meanshift.cluster_centers_[0]
        else:
            joint_pred = [np.nan, np.nan]
        joint_preds.append(joint_pred)
    joint_preds = np.array(joint_preds)
    return joint_preds

def plot_preds(depth_map_resized, pred_segm, joint_preds):
    plt.figure()
    plt.imshow(depth_map_resized, cmap='gray')
    plt.scatter(joint_preds[:, 0], joint_preds[:, 1], c='blue', marker='x', label='Predicted Joints')
    plt.title("Predicted Joints on Input Depth Map")
    plt.legend()
    plt.savefig('joints_prediction_on_input.png')

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(label2rgb(pred_segm, bg_label=0))
    ax.set_title("Predicted Segmentation")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig('segmentation_pred_on_input.png')
    plt.show()

def main():
    rf_classifier, joint_regressors = load_models()
    depth_map_resized, H, W = load_depth_map()
    pred_segm = predict_segm(depth_map_resized, rf_classifier, H, W)
    joint_preds = predict_joints(pred_segm, joint_regressors, depth_map_resized)
    plot_preds(depth_map_resized, pred_segm, joint_preds)


main() 