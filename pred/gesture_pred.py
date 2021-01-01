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
        self.g_model.eval()
        self.h, self.c = self.g_model.h0(), self.g_model.c0()

    def from_skeleton(self, coord_norm):
        # coord_norm: FXJ, F==1
        ges_data = self.bla.handcrafted_features(coord_norm)  # Shape: (F, C) F==1
        features = np.concatenate((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                              ges_data[PG.BONE_ANGLE_SIN]), axis=1)
        features = features[np.newaxis]
        features = features.transpose((1, 0, 2))  # NFC->FNC
        features = torch.from_numpy(features)
        features = features.to(self.g_model.device, dtype=torch.float32)
        with torch.no_grad():
            output, h, c = self.g_model(features, self.h, self.c)  # Output shape: (1, 1, num_classes)
        np_out = output[0, 0].cpu().numpy()
        max_arg = np.argmax(np_out)
        res_dict = {PG.OUT_ARGMAX: max_arg, PG.OUT_SCORES: np_out}
        return res_dict

    def from_img(self, img: np.ndarray):

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8 and img.ndim == 3, "Expect ndarray of shape (H, W, C)"
        p_res = self.p_predictor.get_coordinates(img)
        res_dict = self.from_skeleton(p_res[PG.COORD_NORM])
        yield res_dict
