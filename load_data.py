import os
import numpy as np
import scipy.io

def load_triplet(depth_path, segm_path, info_path):
    try:
        depth_mat = scipy.io.loadmat(depth_path)
        segm_mat  = scipy.io.loadmat(segm_path)
        info_mat  = scipy.io.loadmat(info_path)

        depth_arr = [v for k, v in depth_mat.items() if "depth" in k.lower()][0]
        segm_arr  = [v for k, v in segm_mat.items()  if "segm" in k.lower()][0]
        joints3d = info_mat.get('joints3D', None)

        if joints3d is None or joints3d.shape[0] != 3:
            raise ValueError("Invalid or missing joints3D")

        if depth_arr.ndim == 2:
            depth_arr = depth_arr[np.newaxis, ...]
        if segm_arr.ndim == 2:
            segm_arr = segm_arr[np.newaxis, ...]

        # Transpose to (T, 24, 3)
        joints3d = np.transpose(joints3d, (2, 1, 0))  # shape: (T, 24, 3)

        # Match shortest time dimension (often depth/segm)
        T = depth_arr.shape[0]
        if joints3d.shape[0] < T:
            raise ValueError("Not enough joints3D frames for available depth frames.")

        return depth_arr, segm_arr, joints3d[:T]

    except Exception as e:
        print(f"⚠️ Skipping triplet due to error: {e}")
        return None, None, None


def find_valid_triplets(root_dir):
    files = os.listdir(root_dir)
    bases = set()

    for f in files:
        if f.endswith("_depth.mat"):
            bases.add(f.replace("_depth.mat", ""))
        if f.endswith("_segm.mat"):
            bases.add(f.replace("_segm.mat", ""))
        if f.endswith("_info.mat"):
            bases.add(f.replace("_info.mat", ""))

    valid_triplets = []
    for base in sorted(bases):
        dpath = os.path.join(root_dir, base + "_depth.mat")
        spath = os.path.join(root_dir, base + "_segm.mat")
        ipath = os.path.join(root_dir, base + "_info.mat")
        if all(os.path.exists(p) for p in [dpath, spath, ipath]):
            valid_triplets.append((dpath, spath, ipath))

    print(f"✅ Found {len(valid_triplets)} valid triplets")
    return valid_triplets

def load_all_triplets(root_dir):
    triplets = find_valid_triplets(root_dir)
    if not triplets:
        raise RuntimeError("❌ Something has gone wrong.")
    depth_all, segm_all, joints3d_all = [], [], []

    for dpath, spath, ipath in triplets:
        depth, segm, joints3d = load_triplet(dpath, spath, ipath)
        if depth is None: continue  # Skip bad samples
        depth_all.append(depth)
        segm_all.append(segm)
        joints3d_all.append(joints3d)

    if not depth_all:
        raise RuntimeError("❌ No valid triplets loaded.")

    depth_all   = np.concatenate(depth_all, axis=0)
    segm_all    = np.concatenate(segm_all, axis=0)
    joints3d_all = np.concatenate(joints3d_all, axis=0)

    print(f"📦 Final shapes — Depth: {depth_all.shape}, Segm: {segm_all.shape}, Joints3D: {joints3d_all.shape}")
    return depth_all, segm_all, joints3d_all

if __name__ == "__main__":
    root_dir = "SURREAL"  # Replace with your path
    depth_all, segm_all, joints3d_all = load_all_triplets(root_dir)

    # Optionally save for reuse
    np.save("depth.npy", depth_all)
    np.save("segm.npy", segm_all)
    np.save("joints3d.npy", joints3d_all)
    print("💾 Saved depth.npy, segm.npy, joints3d.npy")
