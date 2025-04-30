import h5py
from tqdm import tqdm 
import numpy as np
from skimage.transform import resize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import imageio
import os
import matplotlib.pyplot as plt
import joblib  # for saving and loading model

# Ensure matplotlib uses a backend that supports plotting
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' if running headless

# Limit number of samples for quicker testing
# Use all available samples
MAX_TRAIN_SAMPLES = None
MAX_TEST_SAMPLES = None
Cz = 0.0035      # instrincic camera calibration parameter 

def project_to_image(depth_img, joints):
    fx, fy = 285.63, 285.63  # Example focal length
    cx, cy = 160, 120        # Principal point
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2] + 1e-5  # prevent divide-by-zero
    u = fx * x / z + cx
    v = -fy * y / z + cy  # Flip vertical axis to match image coordinates
    return np.clip(u, 0, depth_img.shape[1]-1), np.clip(v, 0, depth_img.shape[0]-1)

def bilinear_interpolate_depth(img, u, v):
    """
    Performs bilinear interpolation for subpixel depth at (u, v) in the depth image.
    (u, v) can be float — img is accessed as [v][u], i.e., row-major.
    """
    h, w = img.shape

    # Clamp to valid range for interpolation
    if u < 0 or u >= w - 1 or v < 0 or v >= h - 1:
        return 0.0  # or np.nan

    x0 = int(np.floor(u))
    x1 = x0 + 1
    y0 = int(np.floor(v))
    y1 = y0 + 1

    # Get fractional parts
    dx = u - x0
    dy = v - y0

    # Get pixel values
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    # Interpolate
    top = Ia * (1 - dx) + Ib * dx
    bottom = Ic * (1 - dx) + Id * dx
    value = top * (1 - dy) + bottom * dy

    return value

import numpy as np

def project_to_image(depth_img, joints):
    fx, fy = 285.63, 285.63  # Example focal length
    cx, cy = 160, 120        # Principal point
    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2] + 1e-5  # prevent divide-by-zero
    u = fx * x / z + cx
    v = -fy * y / z + cy  # Flip vertical axis to match image coordinates
    return np.clip(u, 0, depth_img.shape[1] - 1), np.clip(v, 0, depth_img.shape[0] - 1)

def bilinear_interpolate_depth(img, u, v):
    """
    Performs bilinear interpolation for subpixel depth at (u, v) in the depth image.
    (u, v) can be float — img is accessed as [v][u], i.e., row-major.
    """
    h, w = img.shape

    # Clamp to valid range for interpolation
    if u < 0 or u >= w - 1 or v < 0 or v >= h - 1:
        return 0.0  # or np.nan

    x0 = int(np.floor(u))
    x1 = x0 + 1
    y0 = int(np.floor(v))
    y1 = y0 + 1

    # Get fractional parts
    dx = u - x0
    dy = v - y0

    # Get pixel values
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    # Interpolate
    top = Ia * (1 - dx) + Ib * dx
    bottom = Ic * (1 - dx) + Id * dx
    value = top * (1 - dy) + bottom * dy

    return value

def get_feature_vector(img, offsets, joints):
    """
    This function returns a depth feature descriptor for each joint vector 
    using surrounding pixels and the feature response function. It concatenates 
    each pixel descriptor into a vector of num_joints x num_offsets x 2.
    """
    joints = np.array(joints)
    num_joints = joints.shape[0]
    num_offsets = len(offsets)
    
    # Initialize feature vector
    ft_vector = np.zeros((num_joints, num_offsets + 3))

    # Project joints to image coordinates
    image_u, image_v = project_to_image(img, joints)

    # bilinearly interpolate depth values at projected joint locations 
    depths = np.array([bilinear_interpolate_depth(img, u, v) for u, v in zip(image_u, image_v)])

    # caluclate offsets to speed calculations
    offsets_x = np.array([offset[0] for offset in offsets])
    offsets_y = np.array([offset[1] for offset in offsets])

    for i in range(num_joints):
        u, v = image_u[i], image_v[i]
        joint = joints[i]
        depth = depths[i]
        
        # Calculate feature for each joint
        feature = np.zeros(num_offsets)

        for j in range(num_offsets):
            delta_x = offsets_x[j]
            delta_y = offsets_y[j]
            offset_d = (v + delta_y, u + delta_x)

            # Ensure offsets are within image bounds
            if 0 <= offset_d[0] < img.shape[0] and 0 <= offset_d[1] < img.shape[1]:
                feature[j] = depth - bilinear_interpolate_depth(img, offset_d[1], offset_d[0])
            else:
                feature[j] = 0
        
        # Concatenate joint position with feature vector
        ft_vector[i] = np.hstack((joint, feature))
    
    return ft_vector


def random_sample_offsets():
    """
    This function randomly samples offset values for the feature response function
    """
    num_offsets = 25 # number of offsets to sample, 
    offset_threshold = 10 # highest offset value 
    offsets = []
    for _ in range(num_offsets): 
        x = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        y = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        offsets.append((x, y))
    return offsets 

# ---------- Step 1: Load and prepare training data ----------

print("Loading training data...")
with h5py.File('dataset/dataset/ITOP_side_train_depth_map.h5', 'r') as f_depth, \
     h5py.File('dataset/dataset/ITOP_side_train_labels.h5', 'r') as f_label:
    depth_train = f_depth['data'][:] if MAX_TRAIN_SAMPLES is None else f_depth['data'][:MAX_TRAIN_SAMPLES]
    joints_train = f_label['real_world_coordinates'][:] if MAX_TRAIN_SAMPLES is None else f_label['real_world_coordinates'][:MAX_TRAIN_SAMPLES]

X_train = []
y_train = []
offsets = random_sample_offsets()

for i in tqdm(range(len(depth_train)), desc="Generating training features", unit="item"):
    depth = depth_train[i]
    joints = joints_train[i][:, :3]
    
    # get feature vector for each joint
    joint_feature = get_feature_vector(depth, offsets, joints)
    depth_feature = resize(depth, (240, 320), anti_aliasing=True)
    X_train.append(depth_feature.flatten())
    y_train.append(joint_feature.flatten())
# ---------- Step 2: Load test data ----------

print("Loading test data...")
with h5py.File('dataset/dataset/ITOP_side_test_depth_map.h5', 'r') as f_depth, \
     h5py.File('dataset/dataset/ITOP_side_test_labels.h5', 'r') as f_label:
    depth_test = f_depth['data'][:] if MAX_TEST_SAMPLES is None else f_depth['data'][:MAX_TEST_SAMPLES]
    joints_test = f_label['real_world_coordinates'][:] if MAX_TEST_SAMPLES is None else f_label['real_world_coordinates'][:MAX_TEST_SAMPLES]


X_test = []
y_test = []

for i in tqdm(range(len(depth_test)), desc="Generating test features", unit="item"):
    depth = depth_test[i]
    joints = joints_test[i][:, :3] # use 3d coordinates 
    depth_vector = resize(depth, (240, 320), anti_aliasing=True)
    joint_feature = get_feature_vector(depth, offsets, joints)

    X_test.append(depth_vector.flatten())
    y_test.append(joint_feature.flatten())

# ---------- Step 2.5: Save train and test data to disk ----------

np.save('train_features.npy', np.array(X_train))
np.save('train_labels.npy', np.array(y_train))
np.save('test_features.npy', np.array(X_test))
np.save('test_labels.npy', np.array(y_test))
print("Saved training and testing data to disk.")

# ---------- Step 3: Train the model ----------

model_path = 'random_forest_joint_estimator.pkl'

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    rf = joblib.load(model_path)
else:
    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    joblib.dump(rf, model_path)
    print(f"Model trained and saved to {model_path}.")

# ---------- Step 4: Evaluate the model ----------

print("Evaluating on test set...")
y_pred = rf.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, y_pred))

# ---------- Step 5: Predict joints from a new depth image ----------

def predict_joints_from_image(image_path):
    print(f"Predicting joints for: {image_path}")
    depth_input = imageio.imread(image_path)
    depth_resized = resize(depth_input, (240, 320), anti_aliasing=True).flatten()
    full_prediction = rf.predict([depth_resized])[0]         # shape: (15 × (3 + 25) )
    predicted_joints = full_prediction[:45].reshape(15, 3)  # ignore feature vectors 
    return predicted_joints
 
# ---------- Step 6: Plot joints on a test image ----------

def plot_test_joints(index):
    depth_img = depth_test[index]
    true_joints = joints_test[index].reshape(15, 3)
    predicted_joints = rf.predict([X_test[index]])[0].reshape(15, 3)

    # Convert joint positions from meters to image coordinates


    pred_u, pred_v = project_to_image(depth_img, predicted_joints)
    true_u, true_v = project_to_image(depth_img, true_joints)

    plt.figure(figsize=(6, 8))
    plt.imshow(depth_img, cmap='gray')
    plt.scatter(pred_u, pred_v, color='r', label='Predicted', marker='o', s=40, alpha=0.7)
    plt.scatter(true_u, true_v, color='g', label='True', marker='x', s=40, alpha=0.7)
    plt.legend()
    plt.title(f"Joint Prediction vs Ground Truth (Test index {index})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

# Example usage:
# predicted = predict_joints_from_image('my_depth_image.png')
plot_test_joints(604)





