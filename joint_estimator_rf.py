import h5py
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

def get_feature_vector(img, offsets, joints):
    """
    This function returns a depth feature descriptor of each pixel in a joint vector  
    using the surrouding pixels and the feature response function. it concatenates each pixel 
    descriptor into a vector of num_joints x num_offsets x 2 (15 x 25 x 2)
    """
    # compute the feature response for each pixel 
    joints = np.array(joints)
    ft_vector = np.zeros((joints.shape[0], len(offsets) + 3))
    # for ecah joint in image 
    image_u, image_v = project_to_image(img, joints) # project to image
    for i, joint in enumerate(joints): 
        # create a feature entry for every offset 
        u = int(image_u[i])
        v = int(image_v[i])
        feature = np.zeros(len(offsets))
        x, y, z = joint
        print(z, img[v][u])
        assert abs(z - img[v][u]) > 1e-3, "Depth values do not match joint coordinates %d %d" % (z, img[v][u])
        for j, offset in enumerate(offsets): 
            delta_x, delta_y = offset
            offset_d = v + delta_y, u + delta_x
            # Ensure offsets are within image bounds
            if 0 <= offset_d[0] < img.shape[0] and 0 <= offset_d[1] < img.shape[1]:
                feature[j] = z - img[offset_d[0], offset_d[1]]
            else:
                # If out of bounds, set feature to zero 
                feature[j] = 0
        joint = np.array(joint).reshape(1, 3)
        feature = np.array(feature).reshape(1, -1)
        ft_vector[i] = np.hstack((joint, feature))
        print("Shape", np.array(ft_vector[i]).shape)
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
index = 80
print("plotexample image")
joints = joints_train[index].reshape(15, 3)
x = joints[:, 0]
y = joints[:, 1]
print(x)
print(y)
true_u, true_v = project_to_image(depth_train[index], joints)
plt.figure(figsize=(6, 6))
plt.imshow(depth_train[index], cmap='gray')
plt.scatter(true_u, true_v, color='g', label='True', marker='x', s=40, alpha=0.7)
plt.scatter(x, y, color='r', label='Predicted', marker='o', s=40, alpha=0.7)
plt.legend()
plt.title(f"Joint Prediction vs Ground Truth (Test inSdex {index})")
plt.xlabel("X")
plt.ylabel("Y")
plt.tight_layout()
plt.show()

for i in range(len(depth_train)):
    depth = depth_train[i]
    joints = joints_train[i][:, :3]
    
    # get feature vector for each joint
    joint_feature = get_feature_vector(depth, offsets, joints)
    depth_feature = resize(depth, (320, 240), anti_aliasing=True)
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

for i in range(len(depth_test)):
    depth = depth_test[i]
    joints = joints_test[i][:, :3] # use 3d coordinates 
    depth_vector = resize(depth, (320, 240), anti_aliasing=True)
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
    depth_resized = resize(depth_input, (320, 240), anti_aliasing=True).flatten()
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





