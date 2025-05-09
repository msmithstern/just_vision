
from isolation_mask import isolate_live_feed
from pose_classifier import score_dance
from pose_classifier import load_data
import numpy as np
pose_dir = "dancing"
dance_poses = ["squat", "blow-right", "blow-left", "point-forward", "thumb-right", "thumb-left", "thumb-right", 
               "side", "point-forward", "thumb-right", "thumb-left", "thumb-right", "hip", "hip-forward",
               "hip-backward", "hip-forward", "point-forward", "thumb-right", "thumb-left", "thumb-right", 
               "hip", "hip-forward", "hip-backward", "hip-forward", "point-forward", "thumb-right", "thumb-left",
               "thumb-right", "side", "point-forward", "thumb-right", "thumb-left", "thumb-right", "hip", 
               "hip-forward", "hip-backward", "hip-forward","squat", "y", "m", "c", "a", "y", "m", "c", "a",
               "arm-out", "clap-right", "arm-out", "clap-left", "arm-out", "clap-right", "arm-out", "clap-left"]

def dummy_joint_function(masks):
    return masks
def just_dance():
    X_test, _ = load_data()
    sample_indices = np.random.choice(len(X_test), size=len(dance_poses), replace=False)
    X_test = X_test[sample_indices]
    joints = dummy_joint_function(X_test)
    score_dance(joints)

just_dance()

