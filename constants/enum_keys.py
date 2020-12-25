# key for dictionaries.
from enum import Enum, auto


class PG(Enum):
    VIDEO_PATH = auto()  # path of police gesture video
    GESTURE_LABEL = auto()  # list of gestures corresponding to frames

    COORD_NATIVE = auto()  # native keypoint x,y (int)
    COORD_NORM = auto()  # normalized keypoint x,y 0.~1.

    BONE_LENGTH = auto()  # length of bones
    BONE_ANGLE_SIN = auto()  # angle of bones
    BONE_ANGLE_COS = auto()
