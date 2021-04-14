from typing import Iterable
from constants.enum_keys import HK, PG
from models.bla_lstm import BLA_LSTM
from models.paf import PAF
import torch
import numpy as np

from pgdataset.bone_length_angle.bone_length_angle import BoneLengthAngle
from pred.human_keypoint_pred import HumanKeypointPredict


class GesturePred:
    def __init__(self):
        self.p_predictor = HumanKeypointPredict()
        self.bla = BoneLengthAngle()
        self.g_model = BLA_LSTM(1)
        self.g_model.load_ckpt()
        self.g_model.eval()
        self.h, self.c = self.g_model.h0(), self.g_model.c0()

    def from_skeleton(self, coord_norm):
        # coord_norm: FXJ, F==1
        assert coord_norm.ndim == 3 and coord_norm.shape[0] == 1
        ges_data = self.bla.parse(coord_norm)  # Shape: (F, C) F==1
        features = np.concatenate((ges_data[PG.BONE_LENGTH], ges_data[PG.BONE_ANGLE_COS],
                              ges_data[PG.BONE_ANGLE_SIN]), axis=1)
        features = features[np.newaxis]
        features = features.transpose((1, 0, 2))  # NFC->FNC
        features = torch.from_numpy(features)
        features = features.to(self.g_model.device, dtype=torch.float32)
        with torch.no_grad():
            _, h, c, class_out = self.g_model(features, self.h, self.c)  # Output shape: (1, 1, num_classes)
        self.h, self.c = h, c
        np_out = class_out[0].cpu().numpy()
        max_arg = np.argmax(np_out)
        res_dict = {PG.OUT_ARGMAX: max_arg, PG.OUT_SCORES: np_out, PG.COORD_NORM: coord_norm}
        return res_dict

    def from_img(self, img: np.ndarray):

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8 and img.ndim == 3, "Expect ndarray of shape (H, W, C)"
        p_res = self.p_predictor.get_coordinates(img)
        res_dict = self.from_skeleton(p_res[PG.COORD_NORM][np.newaxis])
        return res_dict
