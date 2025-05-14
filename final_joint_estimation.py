import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MeanShift
from skimage.color import label2rgb 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
import os 

STEP = 4 # down sample stride
dir = "trained_regressors/"
# define coarse label mapping
def get_coarse_label(joint_id):
    """
    This function gets a coarse label which will partition upper and lower body joints. 

    Parameters: 
        joint_id: int id of joint
    Returns:
        1 if in upper body 
        2 if in lower body 
        -1 if it's background 
    """
    upper_body_joints = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22}
    lower_body_joints  = {0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 23}
    if joint_id in upper_body_joints:
        return 1
    elif joint_id in lower_body_joints:
        return 2
    return -1 # background 

def load_files(): 
    """
    This function loads in the numpy files into numpy objects

    Parameters: 
        None
    Returns: 
        depth segmentation and joints as numpy arrays 
    """
    dir = "numpy_files/"
    depth = np.load(dir + "depth.npy")
    segm = np.load(dir + "segm.npy")
    joints2d = np.load(dir + "joints2d.npy")
    return depth, segm, joints2d

def extract_features(depth_img, i, j):
    """
    This function extracts a 5x5 feauture patch to improve joint estimation

    Parameters:
        depth_img: depth image 
        i, j: indices to index into the depth image to extract feature patch
    """
    local_patch = depth_img[max(0, i-2):min(depth_img.shape[0], i+3), max(0, j-2):min(depth_img.shape[1], j+3)]
    local_mean = np.mean(local_patch)
    local_std = np.std(local_patch)
    return [depth_img[i, j], i, j, local_mean, local_std]

def train_downsample(train_idx, H, W, step, depth, segm):
    """
    This function prepares downsampled training data. 

    Parameters:
        train_idx - training index 
        H - height 
        W - width
        step - step/stride
        depth - depth array
        segm - segmentation array 
    Returns coarse numpy arrays and upper and lower body numpy arrays 

    """
    X_coarse, y_coarse = [], []
    X_upper, y_upper = [], []
    X_lower, y_lower = [], []

    #loop through height and width 
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
                X_coarse.append(feat)
                y_coarse.append(coarse)
                if coarse == 1:
                    X_upper.append(feat)
                    y_upper.append(label)
                elif coarse == 2:
                    X_lower.append(feat)
                    y_lower.append(label)
    return np.array(X_coarse), np.array(y_coarse), np.array(X_upper), np.array(y_upper), np.array(X_lower), np.array(y_lower)

def train_classifiers(X_coarse, y_coarse, X_upper, y_upper, X_lower, y_lower):
    """
    This function trains classifiers from .joblib files and creates them if they don't exist. 

    Paramters:
        X_coarse: data coarse 
        y_coarse: labels coarse 
        X_upper: data upper 
        y_upper: labels upper
        X_lower: data lower
        y_lower: labess lower
    Returns: 
        classifiers for coarse/upper/lower
    """
    #paths 
    clf_coarse_path, clf_upper_path, clf_lower_path = dir + 'rf_classifier_coarse.joblib', dir +  'rf_classifier_upper.joblib', dir + 'rf_classifier_lower.joblib'
    
    #if path exists load it otherwise train random forest classifiers (does this for coarse/upper/lower)
    if os.path.exists(clf_coarse_path):
        clf_coarse_path = joblib.load(clf_coarse_path)
    else: 
        clf_coarse = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight='balanced')
        clf_coarse.fit(X_coarse, y_coarse)
        joblib.dump(clf_coarse, clf_coarse_path)
    
    if os.path.exists(clf_upper_path):
        clf_upper_path = joblib.load(clf_upper_path)
    else: 
        clf_upper = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced')
        clf_upper.fit(X_upper, y_upper)
        joblib.dump(clf_upper, clf_upper_path)

    if os.path.exists(clf_lower_path):
        clf_lower_path = joblib.load(clf_lower_path)
    else: 
        clf_lower = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced')
        clf_lower.fit(X_lower, y_lower)
        joblib.dump(clf_lower, clf_lower_path)
    return clf_coarse, clf_upper, clf_lower

def pred_segmentation(test_idx, depth, segm, clf_coarse, clf_upper, clf_lower, H, W): 
    """
    This function predicts a sementation map. 

    Paramters: 
        test_idx- testing index
        depth - depth array
        segm - segmentation array
        clf_coarse - coarse classifier 
        clf_upper - upper claassifer 
        clf_lower - lower classifier
        H - height
        W - width 
    
    Returns depth testing degmentation ground truth testing frame and the predicted segmentation 
    """
    test_frame = test_idx[4]
    #get tests and ground truths using the test frame 
    depth_test = depth[test_frame]
    segm_gt = segm[test_frame] 
    pred_segm = np.zeros((H, W), dtype=np.int32)
    valid_pixels = np.where(depth_test != 0)
    X_test = [extract_features(depth_test, i, j) for i, j in zip(*valid_pixels)]
    if not X_test: 
        return depth_test, segm_gt, test_frame, pred_segm 
    coarse_preds = clf_coarse.predict(X_test)
    refined_preds = []
    #do predictions 
    for feat, c in zip(X_test, coarse_preds):
        if c == 1: 
            refined_preds.append(clf_upper.predict([feat])[0])
        elif c == 2: 
            refined_preds.append(clf_lower.predict([feat])[0])
        else: 
            refined_preds.append(0) # leave background as 0 
    pred_segm[valid_pixels] = refined_preds
    return depth_test, segm_gt, test_frame, pred_segm

def train_regressors(num_joints, train_idx, joints2d, depth, H, W): 
    """
    This function trains the random forest regressors. 

    Parameters:
        num_joints - number of joints 
        train_idx - training index 
        joints2d - joints array 
        depth - depth array
        H - height 
        W - width 
    
    Returns: 
        joint regressors
    """
    joint_regressors = []
    #trains regressors for each joint 
    for joint_id in tqdm(range(num_joints), desc="Training regressors per joint"):
        regr_path = dir + f'rf_regressor_joint{joint_id}.joblib'
        if os.path.exists(regr_path): #checks for joblib files if regressor alread trained 
            regr = joblib.load(regr_path)
        else: #if not it trains it 
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
    """
    This function predicts the joints using the regressors using mean shift. 

    Parameters:
        num_joints - number of joints 
        joint_regressors - joint regressors 
        depth_test - depth test values 
        pred_segm - predicted segmentation 
    
    Returns: 
        joint predictions as a numpy array 
    """
    joint_preds = [] 
    #predict each joint using feature extraction and meanshift 
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
    indices = np.arrange(T)
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    X_coarse, y_coarse, X_upper, y_upper, X_lower, y_lower = train_downsample(train_idx, H, W, STEP, depth, segm)
    clf_coarse, clf_upper, clf_lower = train_classifiers(X_coarse, y_coarse, X_upper, y_upper, X_lower, y_lower)
    depth_test, segm_gt, test_frame, pred_segm = pred_segmentation(test_idx, depth, segm, clf_coarse, clf_upper, clf_lower, H, W)

    joint_regressors = train_regressors(num_joints, train_idx, joints2d, depth, H, W)
    joint_preds = pred_joints(num_joints, joint_regressors, depth_test, pred_segm)

    np.savetxt(f'joint_predictions_frame_{test_frame}.txt', joint_preds, fmt='%.2f', header='x y')
    np.savetxt(f'joint_groundtruth_frame_{test_frame}.txt', joints2d[test_frame], fmt='%.2f', header='x y (GT)')

    plt.figure()
    plt.imshow(depth_test, cmap='gray')

    #plot ground truth joints with indices 
    for i, (x,y) in enumerate(joints2d[test_frame]):
        plt.scatter(x, y, c='lime', label='GT Joints' if i == 0 else "", s=40)
        plt.text(x + 5, y + 5, f'GT{i}', color='lime', fontsize=8)

    #plot predicted joints with indices 
    for i, (x,y) in enumerate(joint_preds):
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