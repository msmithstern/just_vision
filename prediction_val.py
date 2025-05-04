import os
import numpy as np
import joblib
import scipy.io
import matplotlib.pyplot as plt
from test_and_train import load_depth_and_segm, estimate_joints, plot_depth_with_joints, find_valid_pairs

# Load model
model = joblib.load("trained_pose_model.joblib")
print("✅ Loaded trained model.")

# Load test data
test_root = "surreal/data/cmu/val"
test_pairs = find_valid_pairs(test_root)
depth_path, segm_path = test_pairs[40]

depth, segm = load_depth_and_segm(depth_path, segm_path)
joints_gt = estimate_joints(depth, segm)

# Predict and visualize
os.makedirs("output_preds", exist_ok=True)
print(f"🔢 Number of frames: {depth.shape[0]}")
print(f"{os.path.basename(depth_path)} shape: {depth.shape}")


for t in range(min(20, depth.shape[0])):
    x_input = depth[t].flatten().reshape(1, -1)
    y_pred = model.predict(x_input).reshape(-1, 3)
    save_path = f"output_preds/pred_{t:02d}.png"
    plot_depth_with_joints(depth[t], joints_gt[t], y_pred, save_path=save_path)