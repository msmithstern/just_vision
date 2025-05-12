from sklearn.ensemble import RandomForestClassifier 
import joblib # to save/load rf model
import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
from isolation_mask import isolate_live_feed

pose_labels = ["squat", "blow-right", "blow-left","point-forward", "thumb-right", "thumb-left",
               "side", "hip", "hip-forward", "hip-backward", "y", "m", "c", "a", "arm-out", "clap-right", 
               "clap-left", "point-left", "jump-right", "jump-left", "shoot-left", "shoot-right", 
               ]
dance_poses = [
    "squat",
    "blow-right",
    "blow-left",
    "point-forward",
    "thumb-right",
    "thumb-left",
    "thumb-right",
    "side",
    "point-forward",
    "thumb-right",
    "thumb-left",
    "thumb-right",
    "hip",
    "hip-forward",
    "hip-backward",
    "hip-forward",
    "point-forward",
    "thumb-right",
    "thumb-left",
    "thumb-right",
    "side",
    "point-forward",
    "thumb-right",
    "thumb-left",
    "thumb-right",
    "hip",
    "hip-forward",
    "hip-backward",
    "hip-forward",
    "squat",
    "y",
    "m",
    "c",
    "a",
    "y",
    "m",
    "c",
    "a",
    "arm-out",
    "clap-right",
    "arm-out",
    "clap-left",
    "arm-out",
    "clap-right",
    "arm-out",
    "clap-left"
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

def train_random_forest_classifier(X_train, y_train):
    # Train pose Random Forest classifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def compute_score(rf, poses, target_poses):
    # Get pose probabilities for each class
    pose_probabilities = rf.predict_proba(poses)
    # get probability of target pose from joint estimation of deptimage 
    score = 0 
    avg_accuracy = 0
    bias = 0.2
    for i, pose in enumerate(target_poses):
        prediction = np.argmax(pose_probabilities[i])
        if prediction == pose: 
            avg_accuracy += 1
        print("Predicted", pose_labels[prediction])
        print("Actual", pose_labels[pose])
        pose_probabilities[i][pose] += bias
        score += pose_probabilities[i][pose] 
        print(score)
    score /= len(target_poses)
    avg_accuracy /= len(target_poses)
    return score, pose_probabilities, avg_accuracy

def load_dance(data_dir): 
    assert os.path.exists(data_dir), f"Data directory '{data_dir}' does not exist."
    y_train = [pose_label_dict[pose] for pose in dance_poses]
    print("Isolating live feed")
    X_train = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if file_path.endswith('.npy'): 
            depth_image = np.load(file_path)
            depth_image = depth_image[:384, :]  
            plt.imshow(depth_image, cmap='gray')
            plt.colorbar(label='Depth')
            plt.title('Depth Image')
            plt.axis('off')
            plt.show()
            print(depth_image.shape)
            X_train.append(depth_image)
    np.stack(X_train)
    X_train = [X.flatten() for X in X_train]
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    return X_train, y_train
    

def load_data():
    data_dir = "pose-dataset/pose-depth-masks"
    assert os.path.exists(data_dir), f"Data directory '{data_dir}' does not exist."
    y_train = []
    X_train = []
    for files in os.listdir(data_dir):
        if files.endswith(".npy"):
            file_path = os.path.join(data_dir, files)
            filename = os.path.basename(file_path)
            label = filename.split("_")[0]
            if label in pose_label_dict:
                # Load the depth image and append to the list
                depth_images = np.load(file_path)
                for depth_joints in depth_images:
                # Append the depth image and its corresponding 
                    depth_joints = np.nan_to_num(depth_joints, nan=0.0)
                    X_train.append(np.array(depth_joints.flatten()))
                    y_train.append(pose_label_dict[label])
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    # Generate a shuffled i
    # ndex array
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    # Apply the same shuffle to both arrays
    X_train = X_train[indices]
    y_train = y_train[indices]
    return X_train, y_train

def just_dance_score(dir): 
    print("loading data")
    X_train, y_train = load_data()
    rf = train_random_forest_classifier(X_train, y_train)
    X_test, y_test = load_dance(dir)
    print(len(X_test))
    print(len(y_test))
    assert len(X_test) == len(dance_poses)
    print("computing score")
    score, pose_probabilities, accuracy = compute_score(rf, X_test, y_test)
    print(f"Score {score * 100} %")
    print(f"Accuracy {accuracy * 100} %")
    plot_confusion_and_bar_graph(pose_probabilities, y_test)

def score_dance(pose_joints): 
    print("scoring dance")
    pose_joints = [pose.flatten() for pose in pose_joints]
    X_train, y_train = load_data()
    rf = train_random_forest_classifier(X_train, y_train)
    targets = [pose_label_dict[pose] for pose in dance_poses]
    score, pose_probabilities = compute_score(rf, pose_joints, targets)
    print(f"Score {score}")
    plot_confusion_and_bar_graph(pose_probabilities, targets)

def plot_confusion_and_bar_graph(pose_probabilities, target_poses): 
    n_classes = 22
    # Create a confusion matrix considering probabilities
    cm = np.zeros((n_classes, n_classes))
    # Build confusion matrix using the predicted and true labels
    predicted_labels = np.argmax(pose_probabilities, axis=1)
    cm = confusion_matrix(target_poses, predicted_labels, labels=np.arange(n_classes))
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
    # label axes
    plt.xlabel('Pose')
    plt.ylabel('Average Score')
    plt.title('Average Score per Pose')
    plt.ylim(0, 1.1)

    plt.show()

def test_pose_classifier(): 
    print("loading data")
    X, y = load_data()
    split_index = int(0.9 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print("training classifier")
    rf = train_random_forest_classifier(X_train, y_train)
    print("computing score")
    score, pose_probabilities, accuracy = compute_score(rf, X_test, y_test)
    print(f"Score {score}")
    print(f"Accuracy {accuracy}")
    plot_confusion_and_bar_graph(pose_probabilities, y_test)
    
just_dance_score("best_so_far_sunday_eve")
#test_pose_classifier()