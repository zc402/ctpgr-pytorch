from typing import Iterable
from constants.enum_keys import HK, PG
from models.gesture_recognition_model import GestureRecognitionModel
from models.pose_estimation_model import PoseEstimationModel
import torch
import numpy as np

from pgdataset.s3_handcrafted_features import BoneLengthAngle
from pred.human_keypoint_pred import HumanKeypointPredict


class GesturePred:
    def __init__(self):
        self.p_predictor = HumanKeypointPredict()
        self.bla = BoneLengthAngle()
        self.g_model = GestureRecognitionModel(1)
        self.g_model.load_ckpt()

    def get_gesture(self, img_iter: Iterable):
        h, c = self.g_model.h0(), self.g_model.c0()
        for img in img_iter:
            assert isinstance(img, np.ndarray)
            assert img.dtype == np.uint8 and img.ndim == 3, "Expect ndarray of shape (H, W, C)"
            p_res = self.p_predictor.get_coordinates(img)
            features = self.bla.handcrafted_features(p_res[PG.COORD_NORM])
            output, h, c = self.g_model(features, h, c)  # Output shape: (1, 1, num_classes)
            np_out = output[0, 0].cpu().numpy()
            max_arg = np.argmax(np_out)
            res_dict = {PG.OUT_ARGMAX: max_arg, PG.OUT_SCORES: np_out}
            yield res_dict