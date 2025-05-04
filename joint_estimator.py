
import numpy as np 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MeanShift
import os
import matplotlib.pyplot as plt
import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
"""
This class contains the code for training a random forest classifier to classify each pixel 
"""
def find_valid_pairs(root_dir):
    files = os.listdir(root_dir)
    depth_files = [f for f in files if f.endswith("_depth.mat")]
    segm_files  = [f for f in files if f.endswith("_segm.mat")]

    depth_bases = {f.replace("_depth.mat", "") for f in depth_files}
    segm_bases  = {f.replace("_segm.mat", "") for f in segm_files}
    common_bases = sorted(depth_bases & segm_bases)

    pairs = []
    for base in common_bases:
        print(f"🔍 Found pair: {base}")
        dpath = os.path.join(root_dir, base + "_depth.mat")
        spath = os.path.join(root_dir, base + "_segm.mat")
        pairs.append((dpath, spath))

    return pairs

def get_pixel_feature(img, offsets, x, y):
    """
    This function returns a depth feature descriptor for a pixel
    """
    # compute the feature response for each pixel 
    ft_vector = np.zeros(len(offsets))
    h, w = img.shape
    for i, offset in enumerate(offsets): 
        d = img[x][y]
        delta_x, delta_y = offset
        if 0 <= x + delta_x < h and 0 <=  y + delta_y < w:
            ft_vector[i] = d - img[x + delta_x, y + delta_y]
        else: 
            ft_vector[i] = 0
    return ft_vector

def random_sample_offsets():
    """
    This function randomly samples offset values for the feature response function
    """
    num_offsets = 100 # number of offsets to sample, 
    offset_threshold = 30 # highest offset value 
    offsets = []
    for _ in range(num_offsets): 
        x = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        y = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        offsets.append((x, y))
    return offsets 

def normalize_depth(depth):
    return depth 

def learn_offsets(depths, segms, joints3ds):
    """
    Learn average offsets from pixels of each body part to their respective joint positions.
    Returns: Dictionary of class_id -> list of offset vectors (joint - pixel)
    """
    joint_offsets = {}  # class_id: [offset vectors]

    for depth, segm, joints in zip(depths, segms, joints3ds):
        h, w = depth.shape
        for joint_id in range(joints.shape[0]):
            joint_pos = joints[joint_id]  # (x, y, z) in 3D
            mask = (segm == (joint_id + 1))  # labels assumed to be 1-indexed for body parts
            xs, ys = np.where(mask)
            for x, y in zip(xs, ys):
                z = depth[x, y]
                pixel_3d = np.array([x, y, z])
                offset = joint_pos - pixel_3d
                if joint_id not in joint_offsets:
                    joint_offsets[joint_id] = []
                joint_offsets[joint_id].append(offset)

    # Average offsets per joint
    for joint_id in joint_offsets:
        joint_offsets[joint_id] = np.mean(joint_offsets[joint_id], axis=0)

    return joint_offsets

def train_random_forest_classifier(depth_imgs, segm_maps, offsets):
    """
    This function trains the random forest classifier using the training data 
    and returns the trained model. It takes in the training data and training labels
    """
    #for each image in the training set 
    X_train = []
    y_train = []
    for depth, segm in tqdm.tqdm(zip(depth_imgs, segm_maps), desc="Processing images", total=len(depth_imgs)): 
        h, w = depth.shape 
        # get feature and label for each pixel in the image 
        for x in range(h):
            for y in range(w): 
                label = segm[x, y]
                if label == 0:
                    continue  # skip the background pixels
                feat = get_pixel_feature(depth, offsets, x, y)
                X_train.append(feat)
                y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # train random forest classifier 
    rf = RandomForestClassifier(n_estimators = 10, max_depth = 15, random_state = 42)
    rf.fit(X_train, y_train)
    return rf

def classify_pixels(depths, depth_offsets, rf):
    segm_maps = []
    for depth in tqdm.tqdm(depths, desc="Classifying pixels", total=len(depths)):
        h, w = depth.shape
        features = []
        for x in range(h):
            for y in range(w):
                feat = get_pixel_feature(depth, depth_offsets, x, y)
                features.append(feat)
        features = np.array(features)
        pred_labels = rf.predict(features)
        segm_map = pred_labels.reshape((h, w))
        segm_maps.append(segm_map)
    return segm_maps

# MEAN SHIFT CLUSTERING 
def estimate_joints(depths, segm_maps, joint_offsets):
    """
    Estimate 3D joint positions using mean shift clustering on offset-corrected pixels.
    Returns: Array of shape (num_joints, 3)
    """
    all_joints = []
    for depth, segm in tqdm.tqdm(zip(depths, segm_maps), desc="Estimating joints", total=len(depths)):
        num_joints = 24
        estimated_joints = []

        for joint_id in range(num_joints):
            est_points = []
            mask = (segm == (joint_id + 1))  # skip if label not present
            xs, ys = np.where(mask)
            zs = depth[xs, ys]
            valid = zs > 0
            xs, ys, zs = xs[valid], ys[valid], zs[valid]
            if len(xs) == 0:
                estimated_joints.append(np.array([0, 0, 0]))
                continue

            pixel_coords = np.stack([xs, ys, zs], axis=1)
            est_points = pixel_coords + joint_offsets[joint_id]

            # Downsample if needed
            if len(est_points) > 500:
                idx = np.random.choice(len(est_points), 500, replace=False)
                est_points = est_points[idx]

            ms = MeanShift(bandwidth=20)
            ms.fit(est_points)
            estimated_joints.append(ms.cluster_centers_[0])
        all_joints.append(np.array(estimated_joints))
    return np.array(all_joints)

#K MEANS CLUSTERING 

# def estimate_joints(depths, segm_maps, joint_offsets):
#     """
#     Estimate 3D joint positions using KMeans (1 cluster) on offset-corrected pixels.
#     Returns: Array of shape (num_images, num_joints, 3)
#     """
#     all_joints = []
#     for depth, segm in tqdm.tqdm(zip(depths, segm_maps), desc="Estimating joints", total=len(depths)):
#         num_joints = 24
#         estimated_joints = []

#         for joint_id in range(num_joints):
#             mask = (segm == (joint_id + 1))
#             xs, ys = np.where(mask)
#             zs = depth[xs, ys]

#             # Remove pixels with zero depth
#             valid = (zs > 0) & (zs < 10000)
#             if np.count_nonzero(valid) == 0 or joint_id not in joint_offsets:
#                 estimated_joints.append(np.array([0, 0, 0]))
#                 continue

#             xs, ys, zs = xs[valid], ys[valid], zs[valid]
#             pixel_coords = np.stack([xs, ys, zs], axis=1)
#             est_points = pixel_coords + joint_offsets[joint_id]  # apply learned offset

#             # Downsample for efficiency
#             if len(est_points) > 500:
#                 idx = np.random.choice(len(est_points), 500, replace=False)
#                 est_points = est_points[idx]

#             # Cluster using KMeans with 1 cluster (just finds the centroid)
#             kmeans = KMeans(n_clusters=1, random_state=42, n_init='auto')
#             kmeans.fit(est_points)
#             estimated_joints.append(kmeans.cluster_centers_[0])

#         all_joints.append(np.array(estimated_joints))
#     return np.array(all_joints)


def plot_depth_with_joints(depth_frame, joints_gt, joints_pred=None):
    fx = fy = 1050.0
    cx, cy = 160.0, 120.0

    plt.figure()
    plt.imshow(depth_frame, cmap="gray")

    if joints_gt is not None:
        valid_gt = joints_gt[:, 2] > 0
        u_gt = (joints_gt[valid_gt, 0] * fx / joints_gt[valid_gt, 2]) + cx
        v_gt = (joints_gt[valid_gt, 1] * fy / joints_gt[valid_gt, 2]) + cy
        plt.scatter(u_gt, v_gt, c="g", label="Ground Truth")

    if joints_pred is not None:
        valid_pred = joints_pred[:, 2] > 0
        u_pred = (joints_pred[valid_pred, 0] * fx / joints_pred[valid_pred, 2]) + cx
        v_pred = (joints_pred[valid_pred, 1] * fy / joints_pred[valid_pred, 2]) + cy
        plt.scatter(u_pred, v_pred, c="r", marker="x", label="Predicted")

    plt.title("Joints on Depth Image")
    plt.axis("off")
    plt.legend()
    plt.show()

    plt.close()

def load_data():
    """
    load from depths.npy, segms.npy, joints3d.npy"""

    depth = np.load("depth.npy")
    segm = np.load("segm.npy")
    joints3d = np.load("joints3d.npy")
    return depth, segm, joints3d

def main(): 
    # load the training data and labels, normalizing the data
    if not os.path.exists("pose_classifier.pkl"): 
        print("Loading training data...")
        train_depth, train_segm, train_joints = load_data() # load_data()
        # train_depth = train_depth[:50]
        # train_segm = train_segm[:50]
        # train_joints = train_joints[:50]
        train_depth = normalize_depth(train_depth) # replace with max function 
        # train the pixel classifier using the training data and labels 
        print("Generating random offsets...")
        depth_offsets = random_sample_offsets()
        print("Training RF classifier...")
        rf = train_random_forest_classifier(train_depth, train_segm, depth_offsets)
        # learn offsets for joint locations 
        print("Learning joint offsets...")
        joint_offsets = learn_offsets(train_depth, train_segm, train_joints)
        # save
        joblib.dump(rf, "pose_classifier.pkl")
        np.save("joint_offsets.npy", joint_offsets)
        np.save("depth_offset.npy", depth_offsets)
    else: 
        train_depth, train_segm, train_joints = load_data()
        rf = joblib.load("pose_classifier.pkl")
        joint_offsets = np.load("joint_offsets.npy", allow_pickle=True)
        depth_offsets = np.load("depth_offsets.npy", allow_pickle=True)
    print("Testing...")
    test_depth, test_segm, test_joints = train_depth[:10], train_segm[:10], train_joints[:10] # load_data()
    test_depth = normalize_depth(test_depth)
    # test! 
    print("Classifying pixels...")
    pred_segm = classify_pixels(test_depth, depth_offsets, rf)
    print("Estimtating joints...")
    pred_joints = estimate_joints(test_depth, pred_segm, joint_offsets)
    plot_depth_with_joints(test_depth[0], test_joints[0], pred_joints[0])
    # save rf, joint_offsets, depth_offsets to disk

    # Flatten segmentation maps for pixel-wise comparison
    segm_mse = []
    for gt, pred in zip(test_segm, pred_segm):
        gt_flat = gt.flatten()
        pred_flat = pred.flatten()
        segm_mse.append(mean_squared_error(gt_flat, pred_flat))

    avg_segm_mse = np.mean(segm_mse)
    print(f"📊 Mean Squared Error (Segmentation Maps): {avg_segm_mse:.4f}")

    # Joint coordinate MSE
    joint_mse = []
    for gt, pred in zip(test_joints, pred_joints):
        joint_mse.append(mean_squared_error(gt, pred))

    avg_joint_mse = np.mean(joint_mse)
    print(f"📌 Mean Squared Error (Joint Positions): {avg_joint_mse:.4f}")
    return 0


main()
