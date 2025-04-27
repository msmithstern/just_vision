import h5py
import numpy as np
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import imageio
import os
import matplotlib.pyplot as plt
import joblib
import random

import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if headless

MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None

# Define joint index pairs to swap for horizontal flipping (update as per ITOP format)
LEFT_RIGHT_PAIRS = [
    (2, 3),  # left/right shoulder
    (4, 5),  # left/right elbow
    (6, 7),  # left/right hand
    (8, 9),  # left/right hip
    (10, 11),  # left/right knee
    (12, 13),  # left/right foot
]

def augment_horizontal_flip(depth_img, joints):
    flipped_depth = np.fliplr(depth_img)
    flipped_joints = joints.copy()
    flipped_joints[:, 0] = -flipped_joints[:, 0]
    for l, r in LEFT_RIGHT_PAIRS:
        flipped_joints[[l, r]] = flipped_joints[[r, l]]
    return flipped_depth, flipped_joints

def augment_random_scale(depth_img, joints, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    scaled_joints = joints * scale
    scaled_depth = resize(depth_img, (int(depth_img.shape[0] * scale), int(depth_img.shape[1] * scale)), anti_aliasing=True)
    scaled_depth = resize(scaled_depth, depth_img.shape, anti_aliasing=True)
    return scaled_depth, scaled_joints

def augment_gaussian_noise(depth_img, joints, sigma=0.01):
    noisy_depth = depth_img + np.random.normal(0, sigma, size=depth_img.shape)
    noisy_depth = np.clip(noisy_depth, 0, 1)
    return noisy_depth, joints.copy()

def load_or_create_data():
    if all(os.path.exists(f) for f in ['train_features.npy', 'train_labels.npy', 'test_features.npy', 'test_labels.npy']):
        print("Loading saved train/test data...")
        X_train = np.load('train_features.npy')
        y_train = np.load('train_labels.npy')
        X_test = np.load('test_features.npy')
        y_test = np.load('test_labels.npy')
        depth_test = np.load('depth_test.npy')
        joints_test = np.load('joints_test.npy')
    else:
        print("Processing and augmenting training data...")
        with h5py.File('dataset/ITOP_side_train_depth_map.h5', 'r') as f_depth, \
             h5py.File('dataset/ITOP_side_train_labels.h5', 'r') as f_label:
            depth_train = f_depth['data'][:] if MAX_TRAIN_SAMPLES is None else f_depth['data'][:MAX_TRAIN_SAMPLES]
            joints_train = f_label['real_world_coordinates'][:] if MAX_TRAIN_SAMPLES is None else f_label['real_world_coordinates'][:MAX_TRAIN_SAMPLES]

        X_train = []
        y_train = []

        for i in range(len(depth_train)):
            depth = depth_train[i]
            joints = joints_train[i].reshape(15, 3)

            for aug_func in [lambda d, j: (d, j), augment_horizontal_flip, augment_random_scale, augment_gaussian_noise]:
                aug_depth, aug_joints = aug_func(depth, joints)
                aug_feat = resize(aug_depth, (60, 80), anti_aliasing=True).flatten()
                X_train.append(aug_feat)
                y_train.append(aug_joints.flatten())

        print("Processing test data...")
        with h5py.File('dataset/ITOP_side_test_depth_map.h5', 'r') as f_depth, \
             h5py.File('dataset/ITOP_side_test_labels.h5', 'r') as f_label:
            depth_test = f_depth['data'][:] if MAX_TEST_SAMPLES is None else f_depth['data'][:MAX_TEST_SAMPLES]
            joints_test = f_label['real_world_coordinates'][:] if MAX_TEST_SAMPLES is None else f_label['real_world_coordinates'][:MAX_TEST_SAMPLES]

        X_test = []
        y_test = []
        for i in range(len(depth_test)):
            depth = depth_test[i]
            joints = joints_test[i].flatten()
            feat = resize(depth, (60, 80), anti_aliasing=True).flatten()
            X_test.append(feat)
            y_test.append(joints)

        # Save data
        np.save('train_features.npy', np.array(X_train))
        np.save('train_labels.npy', np.array(y_train))
        np.save('test_features.npy', np.array(X_test))
        np.save('test_labels.npy', np.array(y_test))
        np.save('depth_test.npy', np.array(depth_test))
        np.save('joints_test.npy', np.array(joints_test))
        print("Saved processed and augmented data to disk.")

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), depth_test, joints_test
def get_feature_vector(img, offsets, joints):
    """
    This function returns a depth feature descriptor of each pixel in a joint vector  
    using the surrouding pixels and the feature response function. it concatenates each pixel 
    descriptor into a vector of num_joints x num_offsets x 2 (15 x 100 x 2)
    """
    # compute the feature response for each pixel 
    joints = np.array(joints)
    ft_vector = np.zeros((joints.shape[0], len(offsets)))
    for i, joint in enumerate(joints): 
        feature = np.zeros(len(offsets))
        x, y = joint
        d = img[x][y]
        for j, offset in enumerate(offsets): 
            delta_x, delta_y = offset
            feature[j] = d - img[x + delta_x, y + delta_y]
        ft_vector[i] = feature 
    return ft_vector

def random_sample_offsets():
    """
    This function randomly samples offset values for the feature response function
    """
    num_offsets = 100 # number of offsets to sample, 
    offset_threshold = 20 # highest offset value 
    offsets = []
    for _ in range(num_offsets): 
        x = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        y = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        offsets.append((x, y))
    return offsets 

X_train, y_train, X_test, y_test, depth_test, joints_test = load_or_create_data()

model_path = 'random_forest_joint_estimator.pkl'

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    rf = joblib.load(model_path)
else:
    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}.")

print("Evaluating model...")
y_pred = rf.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, y_pred))

def predict_joints_from_image(image_path):
    print(f"Predicting joints for: {image_path}")
    depth_input = imageio.imread(image_path)
    depth_resized = resize(depth_input, (60, 80), anti_aliasing=True).flatten()
    predicted_joints = rf.predict([depth_resized]).reshape(15, 3)
    return predicted_joints

def plot_test_joints(index):
    depth_img = depth_test[index]
    true_joints = joints_test[index].reshape(15, 3)
    predicted_joints = rf.predict([X_test[index]])[0].reshape(15, 3)

    fx, fy = 285.63, 285.63
    cx, cy = 160, 120

    def project_to_image(joints):
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2] + 1e-5
        u = fx * x / z + cx
        v = -fy * y / z + cy
        return np.clip(u, 0, depth_img.shape[1]-1), np.clip(v, 0, depth_img.shape[0]-1)

    pred_u, pred_v = project_to_image(predicted_joints)
    true_u, true_v = project_to_image(true_joints)

    plt.figure(figsize=(6, 6))
    plt.imshow(depth_img, cmap='gray')
    plt.scatter(pred_u, pred_v, color='r', label='Predicted', marker='o', s=40, alpha=0.7)
    plt.scatter(true_u, true_v, color='g', label='True', marker='x', s=40, alpha=0.7)
    plt.legend()
    plt.title(f"Joint Prediction vs Ground Truth (Index {index})")
    plt.tight_layout()
    plt.show()

# Example:
plot_test_joints(604)
