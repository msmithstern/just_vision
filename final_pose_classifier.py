from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
from test_model import process_depth_map

# labels of all the possible poses 
pose_labels = ["squat", "blow-right", "blow-left","point-forward", "thumb-right", "thumb-left",
               "side", "hip", "hip-forward", "hip-backward", "y", "m", "c", "a", "arm-out", "clap-right", 
               "clap-left", "point-left", "jump-right", "jump-left", "shoot-left", "shoot-right", 
               ]
# the sequence of poses for the YMCA dance 
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
    """
    Description: Either retrieves classifier from job lib file or trains a new one
    Returns: Random forest classifier 
    """
    if os.path.exists("pose_classifier.pkl"):
        rf = joblib.load("pose_classifier.pk1")
    else:
        rf = train_random_forest_classifier()
        joblib.dump(rf, "pose_classifier.pkl")
    return rf 

def train_random_forest_classifier(X_train, y_train):
    """
    Description: Trains random forest classifier on training data 
    Parameters: 
        - X_train joint predictions of training dataset 
        - Y_train pose labels of training dataset 
    Returns: Random forest classifier
    """
    # Train random forest pose classifier
    rf = RandomForestClassifier(n_estimators=200, random_state = 42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def compute_score(rf, poses, target_poses): 
    """
    Description: Computes dance scores based on the predicted and actual poses 
    Parameters: 
        - rf: random forest classifier 
        - poses: pose probabilities for each pose in the dance 
        - target_poses: the actual poses  
    Returns: score number, pose probabilities, average accuracy
    """
    # get pose probabilities for each pose class
    pose_probabilities = rf.predict_proba(poses)
    score = 0
    avg_accuracy = 0
    bias = 0
    for i, pose in enumerate(target_poses):
        #get highest prediction 
        pose_probabilities[i][pose] += bias
        prediction = np.argmax(pose_probabilities[i])
        if prediction == pose:
            avg_accuracy += 1
        score += pose_probabilities[i][pose]
    score /= len(target_poses)
    avg_accuracy /= len(target_poses)
    return score, pose_probabilities, avg_accuracy

def load_dance(data_dir):
    """
    Description: Loads snapshots of captured dance from a given directory 
    Parameters: 
        - data_dir: path to dance captures 
    Returns: testing data and ground truth labels x_test, y_test 
    """
    y_test = [pose_label_dict[pose] for pose in dance_poses]
    X_test = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if file_path.endswith('.npy'):
            depth_image = np.load(file_path)
            depth_image = depth_image[:384, :]
            _, _, joints = process_depth_map(depth_image)
            X_test.append(joints)
    np.stack(X_test)
    X_test = [X.flatten() for X in X_test]
    y_test = np.array(y_test)
    X_test = np.array(X_test)
    return X_test, y_test

def load_data():
    """
    Description: Loads training data from files 
    Returns: processed training data with labels obtained from file names 
    """
    data_dir = "joint_pred"
    y_train = []
    X_train = []
    for files in os.listdir(data_dir):
        if files.endswith(".npy"):
            file_path = os.path.join(data_dir, files)
            filename = os.path.basename(file_path)
            label = filename.split("_")[0]
            if label in pose_label_dict:
                # load depth image and append to list
                depth_images = np.load(file_path)
                for depth_joints in depth_images: 
                    depth_joints = np.nan_to_num(depth_joints, nan=0.0)
                    X_train.append(np.array(depth_joints.flatten()))
                    y_train.append(pose_label_dict[label])
            y_train = np.array(y_train)
            X_train = np.array(X_train)
            #shuffle indices to shuffle training x and y 
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            #apply shuffle
            X_train = X_train[indices]
            y_train = y_train[indices]
            return X_train, y_train

def just_dance_score(dir):
    """
    Description: Loads and scores one run of the just dance program. Prints out the score and accuracy 
    and plots a confusion matrix and histogram to analyze the results. 
    Parameters: 
        - dir: folder path to snapshots of the dance 
    """
    # loading data
    X_train, y_train = load_data()
    rf = train_random_forest_classifier(X_train, y_train)
    X_test, y_test = load_dance(dir)
    # computing score 
    score, pose_probabilities, accuracy = compute_score(rf, X_test, y_test)
    print(f"Score {score * 100} %")
    print(f"Accuracy {accuracy * 100} %")
    plot_confusion_and_bar_graph(pose_probabilities)

def plot_confusion_and_bar_graph(pose_probabilities, target_poses):
    """
    Description: Plot confusion and bar graph to demonstrate precision and accuray
    Parameters: 
        - pose_probabilities: For each pose in the sequence, the probabilities that the user
        is hitting any of the poses in our dataset
        - target_poses: the target poses in the sequence 
    """
    #create confusion marix 
    predicted_labels = np.argmax(pose_probabilities, axis=1)
    cm = confusion_matrix(target_poses, predicted_labels, labels=np.arange(num_poses))
    #plot confusion matrix
    plt.figure(figsize = (8,6))
    sns.heatmap(cm, annot=False, fmt='.2f', cmap='viridis', cbar=True, square=True, xticklabels=np.arange(num_poses), yticklabels=np.arange(num_poses))
    #add title and labels
    plt.title("Confusion Matrix with Pose Probabilities")
    plt.xlabel("Predicted Pose")
    plt.ylabel("True Pose")
    plt.show()
    
    #plot bar graph
    labels = []
    for probs, true_label in zip(pose_probabilities, target_poses):
        labels.append(probs[true_label])
    class_totals = {}
    for value, name in zip(labels, target_poses):
        if name not in class_totals:
            class_totals[name] = []
        class_totals[name].append(value)
    # compute average 
    means = {name : np.mean(vals) for name, vals in class_totals.items()}
    classes = list(pose_labels[key] for key in means.keys())
    values = [means[name] for name in means.keys()]
    plt.figre(figsize=(10, 6))
    bars = plt.bar(classes, values)
    plt.xticks(rotation = 60, fontsize = 6)
    plt.xlabel('Pose')
    plt.ylabel('Average Score')
    plt.title('Average Score per Pose')
    plt.ylim(0, 1.1)
    plt.show()

def test_pose_classifier():
    """
    Description: Test method that splits dataset into test and train and prints out evaluation metrics
    Not used in just dance program 
    """
    X, y = load_data()
    split_index = int(0.9 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    rf = train_random_forest_classifier(X_train, y_train)
    score, pose_probabilities, accuracy = compute_score(rf, X_test, y_test)
    print(f"Score {score * 100} %")
    print(f"Accuracy {score * 100} %")
    plot_confusion_and_bar_graph(pose_probabilities, y_test)