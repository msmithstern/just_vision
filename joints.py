from enum import Enum
"Class to represent joints as an enum for better readability, each output of estimated joint positions "
"is an array indexed by the enum values to allow for access to joint positions by name."
class Joint(Enum): 
    HEAD = 0
    NECK = 1
    RSHOULDER = 2
    LSHOULDER = 3
    RELBOW = 4
    LELBOW = 5
    RHAND = 6
    LHAND = 7
    TORSO = 8 
    RHIP = 9 
    LHIP = 10 
    RKNEE = 11
    LKNEE = 12
    RFOOT = 13
    LFOOT = 14