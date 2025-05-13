import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MeanShift
from skimage.color import label2rgb
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import os

STEP = 4  # downsample stride
dir = "trained_regressors/"
# Define coarse label mapping for Human3.6M
def get_coarse_label(joint_id):
    upper_body_joints = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22}
    lower_body_joints = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 23}
    if joint_id in upper_body_joints:
        return 1  # upper
    elif joint_id in lower_body_joints:
        return 2  # lower
    return -1  # background or unassigned

def load_files():
    dir = "numpy_files/"
    depth = np.load(dir + "depth.npy")
    segm = np.load(dir + "segm.npy")
    joints2d = np.load(dir + "joints2d.npy")
    return depth, segm, joints2d

def extract_features(depth_img, i, j):
    local_patch = depth_img[max(0, i-2):min(depth_img.shape[0], i+3), max(0, j-2):min(depth_img.shape[1], j+3)]
    local_mean = np.mean(local_patch)
    local_std = np.std(local_patch)
    return [depth_img[i, j], i, j, local_mean, local_std]

def train_downsample(train_idx, H, W, step, depth, segm):
    X_c, y_c = [], []
    X_u, y_u = [], []
    X_l, y_l = [], []
    for t in tqdm(train_idx, desc="Preparing training data"):
        for i in range(0, H, step):
            for j in range(0, W, step):
                label = segm[t, i, j]
                if depth[t, i, j] == 0:
                    continue
                coarse = get_coarse_label(label)
                if coarse == -1:
                    continue
                feat = extract_features(depth[t], i, j)
                X_c.append(feat)
                y_c.append(coarse)
                if coarse == 1:
                    X_u.append(feat)
                    y_u.append(label)
                elif coarse == 2:
                    X_l.append(feat)
                    y_l.append(label)
    return np.array(X_c), np.array(y_c), np.array(X_u), np.array(y_u), np.array(X_l), np.array(y_l)

def train_classifiers(X_c, y_c, X_u, y_u, X_l, y_l):
    clf_c_path, clf_u_path, clf_l_path = dir + 'rf_classifier_coarse.joblib', dir + 'rf_classifier_upper.joblib', dir + 'rf_classifier_lower.joblib'
    if os.path.exists(clf_c_path):
        clf_c = joblib.load(clf_c_path)
    else:
        clf_c = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced')
        clf_c.fit(X_c, y_c)
        joblib.dump(clf_c, clf_c_path)
    if os.path.exists(clf_u_path):
        clf_u = joblib.load(clf_u_path)
    else:
        clf_u = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced')
        clf_u.fit(X_u, y_u)
        joblib.dump(clf_u, clf_u_path)
    if os.path.exists(clf_l_path):
        clf_l = joblib.load(clf_l_path)
    else:
        clf_l = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced')
        clf_l.fit(X_l, y_l)
        joblib.dump(clf_l, clf_l_path)
    return clf_c, clf_u, clf_l

def pred_segmentation(test_idx, depth, segm, clf_c, clf_u, clf_l, H, W):
    test_frame = test_idx[4]
    depth_test = depth[test_frame]
    segm_gt = segm[test_frame]
    pred_segm = np.zeros((H, W), dtype=np.int32)
    valid_pixels = np.where(depth_test != 0)
    X_test = [extract_features(depth_test, i, j) for i, j in zip(*valid_pixels)]
    if not X_test:
        return depth_test, segm_gt, test_frame, pred_segm
    coarse_preds = clf_c.predict(X_test)
    refined_preds = []
    for feat, c in zip(X_test, coarse_preds):
        if c == 1:
            refined_preds.append(clf_u.predict([feat])[0])
        elif c == 2:
            refined_preds.append(clf_l.predict([feat])[0])
        else:
            refined_preds.append(0)  # leave background as 0
    pred_segm[valid_pixels] = refined_preds
    return depth_test, segm_gt, test_frame, pred_segm

def train_regressors(num_joints, train_idx, joints2d, depth, H, W):
    joint_regressors = []
    for joint_id in tqdm(range(num_joints), desc="Training regressors per joint"):
        regr_path = dir + f'rf_regressor_joint{joint_id}.joblib'
        if os.path.exists(regr_path):
            regr = joblib.load(regr_path)
        else:
            X_regr, y_regr = [], []
            for t in train_idx:
                x, y = joints2d[t, joint_id]
                x, y = int(x), int(y)
                if 0 <= x < W and 0 <= y < H and depth[t, y, x] != 0:
                    X_regr.append(extract_features(depth[t], y, x))
                    y_regr.append([x, y])
            regr = RandomForestRegressor(n_estimators=100, max_depth=15)
            if X_regr:
                regr.fit(X_regr, y_regr)
                joblib.dump(regr, regr_path)
        joint_regressors.append(regr)
    return joint_regressors

def pred_joints(num_joints, joint_regressors, depth_test, pred_segm):
    joint_preds = []
    for joint_id in tqdm(range(num_joints), desc="Predicting joints"):
        mask = (pred_segm == (joint_id + 1))
        ys, xs = np.where(mask)
        test_points = []
        for y, x in zip(ys, xs):
            feat = extract_features(depth_test, y, x)
            pred = joint_regressors[joint_id].predict([feat])
            test_points.append(pred[0])
        if test_points:
            ms = MeanShift(bandwidth=30)
            ms.fit(test_points)
            joint_preds.append(ms.cluster_centers_[0])
        else:
            joint_preds.append([0, 0])
    return np.array(joint_preds)

def main():
    depth, segm, joints2d = load_files()
    T, H, W = depth.shape
    num_joints = joints2d.shape[1]
    indices = np.arange(T)
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    X_c, y_c, X_u, y_u, X_l, y_l = train_downsample(train_idx, H, W, STEP, depth, segm)
    clf_c, clf_u, clf_l = train_classifiers(X_c, y_c, X_u, y_u, X_l, y_l)
    depth_test, segm_gt, test_frame, pred_segm = pred_segmentation(test_idx, depth, segm, clf_c, clf_u, clf_l, H, W)

    joint_regressors = train_regressors(num_joints, train_idx, joints2d, depth, H, W)
    joint_preds = pred_joints(num_joints, joint_regressors, depth_test, pred_segm)

    # Save joint predictions and ground truth to file
    np.savetxt(f'joint_predictions_frame_{test_frame}.txt', joint_preds, fmt='%.2f', header='x y')
    np.savetxt(f'joint_groundtruth_frame_{test_frame}.txt', joints2d[test_frame], fmt='%.2f', header='x y (GT)')

    plt.figure()
    plt.imshow(depth_test, cmap='gray')
    
    # Plot ground truth joints with numbers
    for i, (x, y) in enumerate(joints2d[test_frame]):
        plt.scatter(x, y, c='lime', label='GT Joints' if i == 0 else "", s=40)
        plt.text(x + 5, y + 5, f'GT{i}', color='lime', fontsize=8)
    
    # Plot predicted joints with numbers
    for i, (x, y) in enumerate(joint_preds):
        plt.scatter(x, y, c='blue', marker='x', label='Predicted Joints' if i == 0 else "")
        plt.text(x + 5, y + 5, f'P{i}', color='blue', fontsize=8)
    
    plt.title("Depth Map with Joint Predictions")
    plt.axis("off")
    plt.legend()
    plt.savefig(f'depth_with_joints_{test_frame}.png')
    plt.show()

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

main()
