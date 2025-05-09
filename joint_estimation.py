import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MeanShift
from skimage.color import label2rgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import os

STEP = 4 # downsample stride

def load_files():
    depth = np.load("depth.npy")      # shape: (T, H, W)
    segm = np.load("segm.npy")        # shape: (T, H, W)
    joints2d = np.load("joints2d.npy") 
    return depth, segm, joints2d

def extract_features(depth_img, i, j):
    local_patch = depth_img[max(0, i-2):min(depth_img.shape[0], i+3), max(0, j-2):min(depth_img.shape[1], j+3)]
    local_mean = np.mean(local_patch)
    local_std = np.std(local_patch)
    return [depth_img[i, j], i, j, local_mean, local_std]

def train_downsample(train_idx, H, W, step, depth, segm):
    X_class, y_class = [], []
    for t in tqdm(train_idx, desc="Preparing segmentation training data"):
        for i in range(0, H, step):
            for j in range(0, W, step):
                if depth[t, i, j] == 0:
                    continue  # skip background pixels
                X_class.append(extract_features(depth[t], i, j))
                y_class.append(segm[t, i, j])
    X_class = np.array(X_class)
    y_class = np.array(y_class)

def train_classifier(X_class, y_class):
    rf_classifier_path = 'rf_classifier2.joblib'
    if os.path.exists(rf_classifier_path):
        rf_classifier = joblib.load(rf_classifier_path)
    else:
        rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=15)
        rf_classifier.fit(X_class, y_class)
        joblib.dump(rf_classifier, rf_classifier_path)
    return rf_classifier

def pred_segm(test_idx, depth, segm, rf_classifier, H, W):
    test_frame = test_idx[4]
    depth_test = depth[test_frame]
    segm_gt = segm[test_frame]
    background_threshold = 1e6
    mask_background = depth > background_threshold
    depth[mask_background] = 0
    pred_segm = np.zeros((H, W), dtype=np.int32)  # or whatever dtype your labels use
    valid_pixels = np.where(depth_test != 0) # only classify non-background pixels
    X_test = np.array([extract_features(depth_test, i, j) for i, j in zip(*valid_pixels)])
    if len(X_test) > 0:
        pred_labels = rf_classifier.predict(X_test)
        pred_segm[valid_pixels] = pred_labels
    return depth_test, segm_gt, test_frame

def train_regressors(num_joints, train_idx, joints2d, depth, H, W):
    joint_regressors = []
    for joint_id in tqdm(range(num_joints), desc="Training regressors per joint"):
        regr_path = f'rf_regressor2_joint{joint_id}.joblib'
        if os.path.exists(regr_path):
            regr = joblib.load(regr_path)
        else:
            X_regr, y_regr = [], []
            for t in train_idx:
                x, y = joints2d[t, joint_id]
                x, y = int(x), int(y)
                if 0 <= x < W and 0 <= y < H:
                    X_regr.append(extract_features(depth[t], y, x))
                    y_regr.append([x, y])
            X_regr = np.array(X_regr)
            y_regr = np.array(y_regr)
            regr = RandomForestRegressor(n_estimators=500, max_depth=15)
            if len(X_regr) > 0:
                regr.fit(X_regr, y_regr)
                joblib.dump(regr, regr_path)
        joint_regressors.append(regr)
    return joint_regressors

def pred_joints(num_joints, joint_regressors, depth_test):
    joint_preds = []
    for joint_id in tqdm(range(num_joints), desc="Predicting joints per joint"):
        # Use only pixels classified as this joint's body part
        mask = (pred_segm == (joint_id + 1))
        ys, xs = np.where(mask)
        test_points = []
        for y, x in zip(ys, xs):
            pred = joint_regressors[joint_id].predict([extract_features(depth_test, y, x)])
            test_points.append(pred[0])
        test_points = np.array(test_points)
        if len(test_points) > 0:
            ms = MeanShift(bandwidth=30, bin_seeding=True, min_bin_freq=1, cluster_all=True)
            ms.fit(test_points)
            joint_pred = ms.cluster_centers_[0]
        else:
            joint_pred = [np.nan, np.nan]
        joint_preds.append(joint_pred)
    joint_preds = np.array(joint_preds)
    return joint_preds

def plot_preds(test_frame, depth_test, segm_gt, joints2d, joint_preds): 
    plt.figure()
    plt.imshow(depth_test, cmap='gray')
    plt.scatter(joints2d[test_frame, :, 0], joints2d[test_frame, :, 1], c='lime', label='GT Joints', s=40)
    plt.scatter(joint_preds[:, 0], joint_preds[:, 1], c='blue', marker='x', label='Predicted Joints')
    plt.title("Predicted vs Ground Truth Joints on Depth Map (Per-Joint Clustering)")
    plt.legend()
    plt.savefig(f'joints_prediction_testframe_{test_frame}.png')

    # === 8. Plot segmentation comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(label2rgb(segm_gt, bg_label=0))
    axes[0].set_title("Ground Truth Segmentation")
    axes[1].imshow(label2rgb(pred_segm, bg_label=0))
    axes[1].set_title("Predicted Segmentation")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f'segmentation_comparison_testframe_{test_frame}.png')
    plt.show()


def main():
    depth, segm, joints2d = load_files()
    T, H, W = depth.shape
    num_joints = joints2d.shape[1]
    indices = np.arange(T)
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    X_class, y_class = train_downsample(train_idx, H, W, STEP, depth, segm)
    rf_classifier = train_classifier(X_class, y_class)
    depth_test, segm_gt, test_frame = pred_segm(test_idx, depth, rf_classifier, H, W)
    joint_regressors = train_regressors(num_joints, train_idx, joints2d, depth, H, W)
    joint_preds = pred_joints(num_joints, joint_regressors, depth_test)
    plot_preds(test_frame, depth_test, segm_gt, joints2d, joint_preds)

main()




