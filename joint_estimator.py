import numpy as np 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MeanShift
import os
import matplotlib.pyplot as plt
import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
"""
This class contains the code for training a random forest classifier to classify each pixel 
"""

def get_pixel_feature(img, offsets, x, y):
    """
    Fast vectorized version of pixel-wise depth-invariant features.
    """
    h, w = img.shape
    d = img[x, y]
    if d == 0:
        return np.zeros(len(offsets), dtype=np.float32)

    offsets = np.array(offsets)  # shape: (N, 2, 2)
    u_offsets = offsets[:, 0, :] / d  # shape: (N, 2)
    v_offsets = offsets[:, 1, :] / d

    # Compute target coordinates
    u_coords = np.stack([x + u_offsets[:, 0], y + u_offsets[:, 1]], axis=1)  # (N, 2)
    v_coords = np.stack([x + v_offsets[:, 0], y + v_offsets[:, 1]], axis=1)

    # Round to nearest integer (nearest neighbor sampling)
    u_coords_rounded = np.round(u_coords).astype(int)
    v_coords_rounded = np.round(v_coords).astype(int)

    def valid_sample(coords):
        return (
            (0 <= coords[:, 0]) & (coords[:, 0] < h) &
            (0 <= coords[:, 1]) & (coords[:, 1] < w)
        )

    u_valid = valid_sample(u_coords_rounded)
    v_valid = valid_sample(v_coords_rounded)

    d_u = np.zeros(len(offsets), dtype=np.float32)
    d_v = np.zeros(len(offsets), dtype=np.float32)

    d_u[u_valid] = img[u_coords_rounded[u_valid, 0], u_coords_rounded[u_valid, 1]]
    d_v[v_valid] = img[v_coords_rounded[v_valid, 0], v_coords_rounded[v_valid, 1]]

    return (d_u - d_v) / d

def random_sample_offsets():
    num_offsets = 50
    offset_threshold = 30
    offsets = []
    for _ in range(num_offsets):
        u = (np.random.randint(-offset_threshold, offset_threshold + 1),
             np.random.randint(-offset_threshold, offset_threshold + 1))
        v = (np.random.randint(-offset_threshold, offset_threshold + 1),
             np.random.randint(-offset_threshold, offset_threshold + 1))
        offsets.append((u, v))
    return offsets

def normalize_depth(depth_maps):
    # Compute min and max per map along the spatial dimensions (H and W)
    min_val = tf.reduce_min(depth_maps, axis=[1, 2], keepdims=True)
    max_val = tf.reduce_max(depth_maps, axis=[1, 2], keepdims=True)
    
    # Normalize each depth map (element-wise)
    return (depth_maps - min_val) / (max_val - min_val + 1e-6)

def learn_offsets(depths, segms, joints3ds):
    """
    Learn average offsets from pixels of each body part to their respective joint positions.
    Returns: Dictionary of class_id -> list of offset vectors (joint - pixel)
    """
    joint_offsets = {}  # class_id: [offset vectors]

    for depth, segm, joints in zip(depths, segms, joints3ds):
        h, w = depth.shape
        for joint_id in range(joints.shape[0]):
            joint_pos = joints[joint_id]  # (x, y) in 2D
            if(joint_pos[0] < 0 or joint_pos[1] < 0 or joint_pos[0] >= h or joint_pos[1] >= w):
                print(f"Joint {joint_id} out of bounds: {joint_pos}")
                joint_pos_z = 0 # out of bounds depth set to 0 
            else:
                joint_pos_z = depth[int(joint_pos[0]), int(joint_pos[1])]
            joint_pos = np.array([joint_pos[0], joint_pos[1], joint_pos_z])
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

def learn_bandwidths(segms, num_joints=24, scale_factor=0.3, default_bandwidth=15.0):
    """
    Learn per-joint bandwidths based on average segmentation size.

    Parameters:
        segms: List of segmentation maps
        num_joints: Number of joints
        scale_factor: Scales the avg width/height of segment to determine bandwidth
        default_bandwidth: Fallback value if a joint is never seen

    Returns:
        np.ndarray of shape (num_joints,) with learned bandwidths
    """
    bandwidths = np.zeros(num_joints)
    counts = np.zeros(num_joints)

    for seg in segms:
        for c in range(num_joints):
            mask = (seg == (c + 1))
            ys, xs = np.where(mask)

            if len(xs) > 0:
                width = xs.max() - xs.min()
                height = ys.max() - ys.min()
                avg_size = (width + height) / 2
                bandwidths[c] += avg_size
                counts[c] += 1

    # Average and apply scale factor
    for c in range(num_joints):
        if counts[c] > 0:
            bandwidths[c] = (bandwidths[c] / counts[c]) * scale_factor
        else:
            bandwidths[c] = default_bandwidth
    return bandwidths


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


def classify_pixels(depths, gt_segm_maps, depth_offsets, rf):
    segm_maps = []
    for depth, gt_segm_map in tqdm.tqdm(zip(depths, gt_segm_maps), desc="Classifying pixels", total=len(depths)):
        h, w = depth.shape
        features = []
        for x in range(h):
            for y in range(w):
                feat = get_pixel_feature(depth, depth_offsets, x, y)
                features.append(feat)
        features = np.array(features)
        pred_labels = rf.predict(features)
        segm_map = pred_labels.reshape((h, w))
        # Zero out all entries that are 0 in the ground truth segmentation map
        segm_map[gt_segm_map == 0] = 0
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
                estimated_joints.append(np.array([0, 0, 0])) # joint not found, append dummy position
                continue

            pixel_coords = np.stack([xs, ys, zs], axis=1)
            est_points = pixel_coords + joint_offsets[joint_id]

            # Downsample if needed -> could be problematic 
            print("number of estimated points:", len(est_points))
            if len(est_points) > 500:
                idx = np.random.choice(len(est_points), 500, replace=False)
                est_points = est_points[idx]

            ms = MeanShift(bandwidth=20) # experiment with this value 
            ms.fit(est_points)
            labels = ms.labels_
            cluster_sizes = np.bincount(labels)
            largest_cluster_idx = cluster_sizes.argmax()
            best_joint = ms.cluster_centers_[largest_cluster_idx]
            print("number of clusters: ", len(ms.cluster_centers_))
            estimated_joints.append(best_joint)
            # estimated_joints.append(ms.cluster_centers_[0]) # might not always be index 0 
        all_joints.append(np.array(estimated_joints))
    return np.array(all_joints)

def estimate_joints_meanshift(depths, segm_maps, bandwidths, z_offsets):
    """
    Estimate joint positions from depth and segmentation maps using mean shift.
    Parameters:
        depths (List[np.ndarray]): Depth maps (H x W)
        segm_maps (List[np.ndarray]): Segmentation maps (H x W)
        bandwidths (np.ndarray): Per-joint bandwidths (num_joints,)
        z_offsets (np.ndarray): Per-joint correction vectors (num_joints, 3)
    Returns:
        np.ndarray: Estimated joints (N_frames, num_joints, 3)
    """
    num_joints = len(bandwidths)
    all_joints = []

    for depth, segm in tqdm.tqdm(zip(depths, segm_maps), desc="Estimating joints", total=len(depths)):
        estimated_joints = []

        for joint_id in range(num_joints):
            mask = (segm == (joint_id + 1))
            ys, xs = np.where(mask)
            zs = depth[ys, xs]
            valid = zs > 0
            xs, ys, zs = xs[valid], ys[valid], zs[valid]
            if len(xs) == 0:
                estimated_joints.append(np.array([0.0, 0.0, 0.0]))
                continue

            proposals = np.stack([xs, ys, zs], axis=1)

            # Optional: Downsample large clusters
            if len(proposals) > 1000:
                idx = np.random.choice(len(proposals), 1000, replace=False)
                proposals = proposals[idx]

            # Apply mean shift
            ms = MeanShift(bandwidth=bandwidths[joint_id], bin_seeding=True)
            ms.fit(proposals)
            labels = ms.labels_
            centers = ms.cluster_centers_

            # Choose largest cluster
            counts = np.bincount(labels)
            best_cluster = centers[counts.argmax()]

            # Apply z-offset correction
            corrected_joint = best_cluster + z_offsets[joint_id]
            estimated_joints.append(corrected_joint)

        all_joints.append(np.stack(estimated_joints, axis=0))

    return np.stack(all_joints, axis=0)

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
#             kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
#             kmeans.fit(est_points)
#             estimated_joints.append(kmeans.cluster_centers_[0])

#         all_joints.append(np.array(estimated_joints))
#     return np.array(all_joints)

def plot_confusion_matrix(true_segm, pred_segm, num_classes, filename="confusion_matrix.png"):
    """
    Plot confusion matrix for pixel classification from segmentation masks.
    true_segm, pred_segm: 2D arrays of ground truth and predicted segmentation maps.
    num_classes: Number of classes including background (e.g., 25 for 24 joints + background).
    """
    # Flatten the segmentation masks to 1D arrays
    y_true = true_segm.flatten()
    y_pred = pred_segm.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(num_classes))

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='viridis', xticks_rotation='vertical')
    plt.title("Confusion Matrix (Pixel Classification)")
    
    # Save the plot
    plt.savefig(filename)
    plt.close()

    print(f"Saved confusion matrix to {filename}")


def load_data():
    """
    load from depths.npy, segms.npy, joints3d.npy"""

    depth = np.load("depth.npy")
    segm = np.load("segm.npy")
    joints2d = np.load("joints2d.npy")
    return depth, segm, joints2d

def plot_segmentation_comparison(segm_gt, segm_pred, num_joints, filename="segmentation_comparison.png"):
    """ Plots GT and Predicted segmentation maps side-by-side. """
    max_label = max(np.max(segm_gt), np.max(segm_pred), num_joints) # Determine range for colormap
    cmap = plt.get_cmap('tab20', max_label + 1) # Use a categorical colormap with enough colors

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(segm_gt, cmap=cmap, vmin=0, vmax=max_label)
    plt.title(f'Ground Truth Segmentation (Labels 0-{np.max(segm_gt)})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segm_pred, cmap=cmap, vmin=0, vmax=max_label)
    plt.title(f'Predicted Segmentation (Labels 0-{np.max(segm_pred)})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved segmentation comparison plot to {filename}")
    plt.close()
def plot_test_and_pred_joints(depth_mask, segmentation_mask, test_joints, pred_joints):
    """
    Plots depth mask with segmentation overlay and both test and predicted 2D joints.

    Parameters:
        depth_mask (np.ndarray): 2D array of depth values.
        segmentation_mask (np.ndarray): Binary mask of the segmented region (same shape as depth_mask).
        test_joints (np.ndarray): Ground truth joint coordinates, shape (num_joints, 3).
        pred_joints (np.ndarray): Predicted joint coordinates, shape (num_joints, 3).
    """
    plt.figure(figsize=(8, 8))
    
    # Show depth mask
    plt.imshow(depth_mask, cmap='gray')
    
    # Overlay segmentation in red (transparent)
    masked_seg = np.ma.masked_where(segmentation_mask == 0, segmentation_mask)
    plt.imshow(masked_seg, cmap='Reds', alpha=0.5)

    # Plot test joints (in green)
    plt.scatter(test_joints[:, 0], test_joints[:, 1], c='lime', label='Test Joints', marker='o')

    # Plot predicted joints (in blue)
    plt.scatter(pred_joints[:, 0], pred_joints[:, 1], c='dodgerblue', label='Predicted Joints', marker='x')

    plt.legend()
    plt.title("Test vs Predicted Joints on Segmented Depth Mask")
    plt.axis('off')
    plt.show()

def main(): 
    # load the training data and labels, normalizing the data
     # Camera intrinsics
    fx, fy = 1050.0, 1050.0
    cx, cy = 160.0, 120.0
    if not os.path.exists("pose_classifier.pkl"): 
        print("Loading training data...")
        train_depth, train_segm, train_joints = load_data() # load_data()
        # train_depth = train_depth[:50]
        # train_segm = train_segm[:50]
        # train_joints = train_joints[:50]
        #train_depth = normalize_depth(train_depth) # replace with max function 
        # train the pixel classifier using the training data and labels 
        print("Learning joint offsets...")
        joint_offsets = learn_offsets(train_depth, train_segm, train_joints)
        print("Generating random offsets...")
        depth_offsets = random_sample_offsets()
        bandwidths = learn_bandwidths(train_segm, num_joints=24, scale_factor = 0.3, default_bandwidth=15.0)
        print("Training RF classifier...")
        rf = train_random_forest_classifier(train_depth, train_segm, depth_offsets)
        # learn offsets for joint locations 
        # save
        joblib.dump(rf, "pose_classifier.pkl")
        np.save("depth_offset.npy", depth_offsets, allow_pickle=True)
    else: 
        train_depth, train_segm, train_joints = load_data()
        rf = joblib.load("pose_classifier.pkl")
        print("Learning joint offsets...")
        joint_offsets = learn_offsets(train_depth, train_segm, train_joints)
        depth_offsets = np.load("depth_offsets.npy", allow_pickle=True)
        bandwidths = learn_bandwidths(train_segm, num_joints=24, scale_factor = 0.3, default_bandwidth=15.0)
    print("Testing...")
    test_depth, test_segm, test_joints = train_depth[:10], train_segm[:10], train_joints[:10] # load_data()
    #test_depth, test_segm = normalize_depth(test_depth)
    # test! 
    print("Classifying pixels...")
    pred_segm = classify_pixels(test_depth, test_segm, depth_offsets, rf)
    print("Estimtating joints...")
    pred_joints = estimate_joints(test_depth, pred_segm, joint_offsets)
    print("Estimating joints using mean shift...")
    pred_joints_meanshift = estimate_joints_meanshift(test_depth, pred_segm, bandwidths, joint_offsets)
    plot_segmentation_comparison(test_segm[0], pred_segm[0], num_joints=24, filename="segmentation_comparison.png")
    plot_test_and_pred_joints(test_depth[0], test_segm[0], test_joints[0], pred_joints[0])
    plot_test_and_pred_joints(test_depth[0], test_segm[0], test_joints[0], pred_joints_meanshift[0])
    # save rf, joint_offsets, depth_offsets to disk
    plot_confusion_matrix(test_segm[0], pred_segm[0], num_classes=25, filename="confusion_matrix.png")
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