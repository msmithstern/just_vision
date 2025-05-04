import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib



def load_depth_and_segm(depth_path, segm_path):
    depth_mat = scipy.io.loadmat(depth_path)
    segm_mat = scipy.io.loadmat(segm_path)
    depth_arr = [v for k, v in depth_mat.items() if "depth" in k.lower()][0]
    segm_arr  = [v for k, v in segm_mat.items() if "segm" in k.lower()][0]

    if depth_arr.ndim == 2:
        depth = depth_arr[np.newaxis, ...]
    else:
        depth = depth_arr

    if segm_arr.ndim == 2:
        segm = segm_arr[np.newaxis, ...]
    else:
        segm = segm_arr

    return depth, segm

def estimate_joints(depth, segm, num_parts=24):
    if depth.ndim == 2:
        depth = depth[np.newaxis, ...]
        segm = segm[np.newaxis, ...]
    elif depth.ndim != 3:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    T, H, W = depth.shape

    fx = fy = 1050.0
    cx, cy = 160.0, 120.0

    joints = np.zeros((T, num_parts, 3))
    for t in range(T):
        for pid in range(1, num_parts + 1):
            mask = (segm[t] == pid)
            if np.sum(mask) == 0: continue
            v, u = np.where(mask)
            z = depth[t][v, u]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            joints[t, pid-1] = [np.mean(x), np.mean(y), np.mean(z)]
    return joints

def plot_depth_with_joints(depth_frame, joints_gt, joints_pred=None, save_path=None):
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

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")
    else:
        plt.show()

    plt.close()



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

def extract_training_data(pairs, max_samples_per_file=3):
    X, Y = [], []
    for depth_path, segm_path in pairs:
        try:
            depth, segm = load_depth_and_segm(depth_path, segm_path)
            joints = estimate_joints(depth, segm)
            for t in range(min(max_samples_per_file, depth.shape[0])):
                X.append(depth[t].flatten())
                Y.append(joints[t].flatten())
        except Exception as e:
            print(f"⚠️ Skipping {depth_path}: {e}")
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    train_root = "SURREAL"
    test_root = "SURREAL"

    print("🔍 Scanning training data...")
    train_pairs = find_valid_pairs(train_root)
    X_train, Y_train = extract_training_data(train_pairs)

    if X_train.size == 0:
        print("❌ No training data found.")
        exit()

    print(f"✅ Loaded {X_train.shape[0]} training samples.")

    print("🧠 Training model...")
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=20, n_jobs=-1))
    model.fit(X_train, Y_train)
    joblib.dump(model, "trained_pose_model.joblib")
    print("✅ Model saved as trained_pose_model.joblib")

    print("🔍 Running on test data...")
    test_pairs = find_valid_pairs(test_root)
    if not test_pairs:
        print("❌ No test pairs found.")
        exit()

    depth_path, segm_path = test_pairs[0]
    depth, segm = load_depth_and_segm(depth_path, segm_path)
    joints_gt = estimate_joints(depth, segm)

    os.makedirs("output_images", exist_ok=True)
    for t in range(min(20, depth.shape[0])):
        x_input = depth[t].flatten().reshape(1, -1)
        y_pred = model.predict(x_input).reshape(-1, 3)
        save_path = f"output_images/frame_{t:02d}.png"
        plot_depth_with_joints(depth[t], joints_gt[t], y_pred, save_path=save_path)