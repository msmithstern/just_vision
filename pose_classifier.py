from sklearn.ensemble import RandomForestClassifier 
import joblib # to save/load rf model
import os 
import numpy as np

# TODO create dictionary to map labels to encodings 
pose_label_dict = {
    "hip" : 0, 
    "knee" : 1, 
}

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
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def compute_score(rf, poses, target_poses):
    # Get pose probabilities for each class
    pose_probabilities = rf.predict_proba(poses)
    # get probability of target pose from joint estimation of deptimage 
    score = 0 
    for i, pose in enumerate(target_poses):
        if pose in pose_label_dict:
            target_pose_id = pose_label_dict[pose]
            score += pose_probabilities[i][target_pose_id]
    score /= len(target_poses)
    return score 

def load_data():
    # TODO load real pose data with labels once i get the data 
    # dummy data for now 
    X_train = np.random.rand(40, 15 * 2)  # 40 samples, 15 joints, 2D coordinates
    y_train = np.random.randint(0, 20, size=40) # 40 samples, 1 label per sample 
    return X_train, y_train

rf = train_random_forest_classifier()
poses = np.random.rand(2, 15 * 2)  # Dummy pose data for testing
target_poses = ["hip", "knee"]
score = compute_score(rf, poses, target_poses)
print(f"Score: {score}")
