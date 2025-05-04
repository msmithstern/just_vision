import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_depth_and_segmentation(depth_path, segm_path):
    depth_mat = scipy.io.loadmat(depth_path)
    segm_mat = scipy.io.loadmat(segm_path)
    
    # Assume variable names follow SURREAL's format
    depth = np.stack([depth_mat[k] for k in depth_mat if 'depth' in k.lower()][0], axis=0)
    segm = np.stack([segm_mat[k] for k in segm_mat if 'segm' in k.lower()][0], axis=0)
    
    return depth, segm  # shape: [T, 240, 320]

def estimate_joints_from_segmentation(depth, segm, num_parts=24):
    """
    Estimate joint positions by computing the mean 3D location of each segmented body part.
    """
    T, H, W = depth.shape
    joints_3d = np.zeros((T, num_parts, 3))

    for t in range(T):
        frame_depth = depth[t]
        frame_segm = segm[t]
        
        for part_id in range(1, num_parts+1):  # parts labeled 1..24
            mask = (frame_segm == part_id)
            if np.sum(mask) == 0:
                continue

            # Get pixel coordinates
            v, u = np.where(mask)
            z = frame_depth[v, u]
            
            # Intrinsics (estimated from SURREAL: 60 deg FOV, 320x240)
            fx = fy = 240.0 / np.tan(np.deg2rad(30))
            cx, cy = W / 2.0, H / 2.0
            
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            x_mean = np.mean(x)
            y_mean = np.mean(y)
            z_mean = np.mean(z)

            joints_3d[t, part_id-1] = [x_mean, y_mean, z_mean]

    return joints_3d  # shape: [T, 24, 3]

def plot_joints_on_depth(depth_frame, joints_frame):
    plt.imshow(depth_frame, cmap='gray')
    plt.scatter(
        (joints_frame[:, 0] * 240 / joints_frame[:, 2]) + 160,
        (joints_frame[:, 1] * 240 / joints_frame[:, 2]) + 120,
        c='r'
    )
    plt.title("Estimated joints on depth frame")
    plt.show()