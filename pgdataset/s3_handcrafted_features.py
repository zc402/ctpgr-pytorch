import numpy as np
import math
from pathlib import Path
from pgdataset.s2_truncate import TruncateDataset
from constants.enum_keys import PG
from constants.keypoints import aic_bones, aic_bone_pairs

class HandCraftedFeaturesDataset(TruncateDataset):
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, clip_len: int):
        super().__init__(data_path, is_train, resize_img_size, clip_len)
        self.bla = BoneLengthAngle()

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        # PG.COORD_NORM: numpy array of shape: (F, X, K)
        #   F: frames, X: xy(2), K: keypoints
        feature_dict = self.bla.handcrafted_features(res_dict[PG.COORD_NORM])
        res_dict.update(feature_dict)
        del res_dict[PG.VIDEO_PATH]
        return res_dict

class BoneLengthAngle:
    """
    F: 帧数
    X: xy (2) 坐标轴维度
    K: keypoints (joints) 一人关键点数量
    B: num_bones 骨头数量
    E: endpoints (2) 骨头端点
    P: num_pairs 骨头配对数量（用于夹角计算）
    """
    def __init__(self):
        self.connections = np.asarray(aic_bones, np.int) - 1
        self.pairs = np.asarray(aic_bone_pairs, np.int) - 1

    def handcrafted_features(self, coord_norm):
        assert len(coord_norm.shape) == 3  # (F, X, K)
        feature_dict = {}
        bone_len = self.__bone_len(coord_norm)
        bone_sin, bone_cos = self.__bone_pair_angle(coord_norm)
        feature_dict[PG.BONE_LENGTH] = bone_len
        feature_dict[PG.BONE_ANGLE_SIN] = bone_sin
        feature_dict[PG.BONE_ANGLE_COS] = bone_cos
        return feature_dict

    def __bone_len(self, coord):

        xy_coord = np.asarray(coord)  # coordinate values. shape: (F, X, K)
        # connect: shape (B, E). B: num_bones, E==2: endpoints
        # Bone coordinate values. shape: (F, X, B, E)
        xy_val = np.take(xy_coord, self.connections, axis=2)
        xy_diff = xy_val[:, :, :, 0] - xy_val[:, :, :, 1]  # shape: (F, X, B)
        xy_diff = xy_diff ** 2  # shape: (F, X, B)
        bone_len = np.sqrt(xy_diff[:, 0] + xy_diff[:, 1])  # shape: (F, B)

        return bone_len

    def __bone_pair_angle(self, coord):
        """
        Compute angle between bones
        :param coord: coordinate of each joint, shape:(F,X,K)
        :return:
        """
        xy_coord = np.asarray(coord)  # shape: (F,X,K)
        xy_val = np.take(xy_coord, self.pairs, axis=2)  # shape: (F, X, P, B, E)
        xy_vec = xy_val[:, :, :, :, 1] - xy_val[:, :, :, :, 0]  # shape: (F,X,P,B)
        ax = xy_vec[:, 0, :, 0]
        bx = xy_vec[:, 0, :, 1]
        ay = xy_vec[:, 1, :, 0]
        by = xy_vec[:, 1, :, 1]
        # dot: a · b = ax × bx + ay × by
        dot_product = ax * bx + ay * by  # shape: (F,P)
        # cross: cz = axby − aybx
        cross_product = ax * by - ay * bx  # shape: (F,P)
        # Magnitude (Length)
        magnitude = np.einsum('fxpb,fxpb->fpb', xy_vec, xy_vec)  # a^2+b^2
        magnitude = np.sqrt(magnitude)  # shape: (F,P,B)
        magnitude[magnitude < 10e-3] = 10e-3  # Filter zero value
        mag_AxB = magnitude[:, :, 0] * magnitude[:, :, 1]  # shape: (F,P)
        cos = dot_product / mag_AxB
        sin = cross_product / mag_AxB
        return sin, cos