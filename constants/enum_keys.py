# key for dictionaries.
from enum import Enum, auto

class HK(Enum):  # Human Keypoint
    B1_OUT = auto()
    B2_OUT = auto()
    B1_SUPERVISION = auto()
    B2_SUPERVISION = auto()  # The intermediate supervision at each stage



class PG(Enum):  # Police Gesture
    VIDEO_PATH = auto()  # path of police gesture video
    GESTURE_LABEL = auto()  # list of gestures corresponding to frames

    COORD_NATIVE = auto()  # native keypoint x,y (int)
    COORD_NORM = auto()  # normalized keypoint x,y 0.~1.

    BONE_LENGTH = auto()  # length of bones
    BONE_ANGLE_SIN = auto()  # angle of bones
    BONE_ANGLE_COS = auto()

    OUT_SCORES = auto()  # prediction output scores, shape: (classes)
    OUT_ARGMAX = auto()  # predicted class, shape: (1)