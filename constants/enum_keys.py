# key for dictionaries.
from enum import Enum, auto


# Note: Adding enum.auto() breaks the saved models.
class HK(Enum):  # Human Keypoint
    NATIVE_IMAGE = auto()  # Native image, no resize and augmentation.
    KEYPOINTS = auto()  # Joint keypoint (x,y) coordinates, shape: (Person, J, X)
    VISIBILITIES = auto()  # Joint keypoint visibilities, shape: (Person, J)
    BOXES = auto()  # Person boxes, shape: (Person, X4). X4: (x1,y1,x2,y2)
    NUM_PEOPLE = auto()  # Number of people in an image

    RE_IMAGE = auto()  # Resized image
    RE_KEYPOINTS = auto()
    RE_BOXES = auto()

    AUG_IMAGE = auto()  # Augmented image
    AUG_KEYPOINTS = auto()
    AUG_BOXES = auto()

    HEAT_KEYPOINTS = auto()  # keypoints for heatmap, usually several times smaller.

    PCM_ALL = auto()  # PCM heatmap of visible and occluded keypoints, shape: JHW
    PCM_NOT_OCC = auto()  # PCM heatmap of visible (not occluded) keypoints, shape: JHW
    PAF_ALL = auto()
    PAF_NOT_OCC = auto()

    NORM_IMAGE = auto()

    DEBUG_NATIVE_IMAGE = auto()  # Image with keypoints for visual debug
    DEBUG_RE_IMAGE = auto()
    DEBUG_AUG_IMAGE = auto()
    DEBUG_PCM_ALL = auto()  # Ground truth PCM on one image, shape (H, W)
    DEBUG_PCM_NOOCC = auto()  # Visible points only
    DEBUG_PAF_ALL = auto()
    DEBUG_PAF_NOOCC = auto()

    B1_OUT = auto()
    B2_OUT = auto()
    B1_SUPERVISION = auto()
    B2_SUPERVISION = auto()  # The intermediate supervision at each stage


class PG(Enum):  # Police Gesture
    VIDEO_PATH = auto()  # path of police gesture video
    VIDEO_NAME = 'VIDEO_NAME'
    NUM_FRAMES = 'NUM_FRAMES'
    GESTURE_LABEL = auto()  # list of gestures corresponding to frames

    COORD_NATIVE = auto()  # native keypoint x,y (int)
    COORD_NORM = auto()  # normalized keypoint x,y 0.~1.

    BONE_LENGTH = auto()  # length of bones
    BONE_ANGLE_SIN = auto()  # angle of bones
    BONE_ANGLE_COS = auto()
    ALL_HANDCRAFTED = "ALL_HANDCRAFTED"

    OUT_SCORES = auto()  # prediction output scores, shape: (classes)
    OUT_ARGMAX = auto()  # predicted class, shape: (1)

    PRED_GESTURES = 'PRED_GESTURES'  # Array of PG.OUT_ARGMAX
