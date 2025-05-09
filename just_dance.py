
from isolation_mask import isolate_live_feed
from pose_classifier import score_dance
pose_dir = "dancing"

def dummy_joint_function(masks):
    return masks
def just_dance():
    masks = isolate_live_feed(pose_dir)
    joints = dummy_joint_function(masks)
    score_dance(joints)

just_dance()

