from sklearn.ensemble import RandomForestClassifier 
import joblib # to save/load rf model
import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# TODO create dictionary to map labels to encodings 
pose_labels = ["squat", "blow-right", "blow-left","point-forward", "thumb-right", "thumb-left",
               "side", "hip", "hip-forward", "hip-backward", "y", "m", "c", "a", "arm-out", "clap-right", 
               "clap-left", "point-left", "jump-right", "jump-left", "shoot-left", "shoot-right", 
               ]
num_poses = len(pose_labels)
pose_label_dict = {label: i for i, label in enumerate(pose_labels)}

def get_or_train_model():
    if os.path.exists("pose_classifier.pkl"): 
        rf = joblib.load("pose_classifier.pkl")
    else:
        rf = train_random_forest_classifier()
        joblib.dump(rf, "pose_classifier.pkl")
    return rf

def train_random_forest_classifier():
    X_train, y_train = load_data()
    # Train pose Random Forest classifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def compute_score(rf, poses, target_poses):
    # Get pose probabilities for each class
    pose_probabilities = rf.predict_proba(poses)
    print(pose_probabilities)
    # get probability of target pose from joint estimation of deptimage 
    score = 0 
    for i, pose in enumerate(target_poses):
        if pose in pose_label_dict:
            target_pose_id = pose_label_dict[pose]
            print(f"Pose: {pose}, ID: {target_pose_id}, Probability: {pose_probabilities[i][target_pose_id]}")
            score += pose_probabilities[i][target_pose_id]
    score /= len(target_poses)
    return score, pose_probabilities 

def load_data():
    data_dir = "pose-dataset/pose-depth-masks"
    depths_train = []
    y_train = []
    X_train = []
    for files in os.listdir(data_dir):
        if files.endswith(".npy"):
            file_path = os.path.join(data_dir, files)
            filename = os.path.basename(file_path)
            label = filename.split("_")[0]
            print(label)
            if label in pose_label_dict:
                # Load the depth image and append to the list
                depth_images = np.load(file_path)
                for depth_image in depth_images:
                # Append the depth image and its corresponding label
                    depth_image = depth_image / np.max(depth_image) 
                    depths_train.append(np.array((depth_image).flatten()))
                    y_train.append(np.array([pose_label_dict[label]]).reshape(-1, 1))
    y_train = np.array(y_train).flatten()
    X_train = np.array(depths_train)
    return X_train, y_train

rf = train_random_forest_classifier()
X_test, y_test = load_data()
sample_indices = np.random.choice(len(X_test), size=200, replace=False)
poses, target_poses = X_test[sample_indices], y_test[sample_indices]
score, pose_probabilities = compute_score(rf, poses, target_poses)


# Example: Generate random predictions and true labels for a pose classification problem
n_classes = 22  # Let's assume 5 classes for pose classification

# Generate random true labels (0 to n_classes-1)
y_true = target_poses

# Generate random predicted probabilities for each class (size: n_samples x n_classes)
y_pred = pose_probabilities

# Normalize the predicted probabilities (they might not sum to 1)
y_pred /= y_pred.sum(axis=1, keepdims=True)

# Create a confusion matrix considering probabilities
cm = np.zeros((n_classes, n_classes))

# Loop through each sample and add probabilities to the confusion matrix
for true_label, probs in zip(y_true, y_pred):
    for predicted_label in range(n_classes):
        cm[true_label, predicted_label] += probs[predicted_label]

# Plot the confusion matrix with color gradients
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt='.2f', cmap='viridis', cbar=True, square=True, xticklabels=np.arange(n_classes), yticklabels=np.arange(n_classes))

# Add title and labels
plt.title("Confusion Matrix with Pose Probabilities")
plt.xlabel("Predicted Pose")
plt.ylabel("True Pose")
plt.show()

# get labels
labels = []
for probs, true_label in zip(pose_probabilities, target_poses): 
    labels.append(probs[true_label])

# Corresponding string annotations (same length as labels)
class_totals = {}
for value, name in zip(labels, target_poses):
    if name not in class_totals:
        class_totals[name] = []
    class_totals[name].append(value)
# Compute average for each class
aggregated_values = {name: np.mean(vals) for name, vals in class_totals.items()}

# Plotting
unique_classes = list(pose_labels[key] for key in aggregated_values.keys())
values = [aggregated_values[name] for name in aggregated_values.keys()]

plt.figure(figsize=(10, 6))
bars = plt.bar(unique_classes, values)
plt.xticks(rotation = 60, fontsize=6)  
# Label axes
plt.xlabel('Pose')
plt.ylabel('Average Score')
plt.title('Average Score per Pose')
plt.ylim(0, 1.1)

plt.show()