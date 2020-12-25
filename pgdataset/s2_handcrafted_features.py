import numpy as np
import math
from pathlib import Path
from pgdataset.s1_skeleton_coords import SkeletonCoords
from constants.enum_keys import PG
from constants.keypoints import aic_bones, aic_bone_pairs

class HandCraftedFeatures(SkeletonCoords):
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple):
        super().__init__(data_path, is_train, resize_img_size)
        self.bla = BoneLengthAngle()

    def __getitem__(self, index):
        res_dict = super().__getitem__(index)
        feature_dict = self.bla.handcrafted_features(res_dict[PG.COORD_NORM])
        res_dict.update(feature_dict)
        return res_dict

class BoneLengthAngle:

    def __init__(self):
        self.connections = np.asarray(aic_bones, np.int) - 1
        self.pairs = np.asarray(aic_bone_pairs, np.int) - 1

    def handcrafted_features(self, coord_norm):
        assert len(coord_norm) == 2 and coord_norm[0].shape == coord_norm[1].shape
        feature_dict = {}
        bone_len = self.__bone_len(coord_norm)
        bone_sin, bone_cos = self.__bone_pair_angle(coord_norm)
        feature_dict[PG.BONE_LENGTH] = bone_len
        feature_dict[PG.BONE_ANGLE_SIN] = bone_sin
        feature_dict[PG.BONE_ANGLE_COS] = bone_cos
        return feature_dict

    def __bone_len(self, coord):

        xy_coord = np.asarray(coord)  # [x1 x2 ...] [y1 y2 ...] shape: (2, num_bones)
        # [[xA xB] [xA xB]...], [[yA yB]...] shape: (2, num_bones, 2)
        xy_val = np.take(xy_coord, self.connections, axis=1)
        xy_diff = xy_val[:, :, 0] - xy_val[:, :, 1]  # shape: (2, num_bones)
        xy_diff = xy_diff ** 2
        bone_len = np.sqrt(xy_diff[0] + xy_diff[1])

        # len_list = []
        # for c in self.connections:
        #     ai, bi = c
        #     ax, ay = coord[0][ai], coord[1][ai]
        #     bx, by = coord[0][bi], coord[1][bi]
        #     l = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
        #     len_list.append(l)
        return bone_len

    def __bone_pair_angle(self, coord):
        """
        Compute angle between bones
        :param coord: coordinate of each joint: [x1 x2 ...] [y1 y2 ...]
        :return:
        """
        xy_coord = np.asarray(coord)  # [x1 x2 ...] [y1 y2 ...] shape: (2, num_bones)
        xy_val = np.take(xy_coord, self.pairs, axis=1)  # shape: (2(xy), num_pairs, bone_pair(2), endpoints(2))
        xy_vec = xy_val[:, :, :, 1] - xy_val[:, :, :, 0]  # shape: (2(xy), num_pairs, bone_pair(2))
        ax = xy_vec[0, :, 0]
        bx = xy_vec[0, :, 1]
        ay = xy_vec[1, :, 0]
        by = xy_vec[1, :, 1]
        # dot: a · b = ax × bx + ay × by
        dot_product = ax * bx + ay * by  # shape: (num_pairs)
        # cross: cz = axby − aybx
        cross_product = ax * by - ay * bx
        magnitude = ((xy_val[0, :, :, 0] - xy_val[0, :, :, 1])**2) + ((xy_val[1, :, :, 0] - xy_val[1, :, :, 1])**2)
        magnitude = np.sqrt(magnitude)  # shape: (num_pairs, bone_pair)
        mag_AxB = magnitude[:, 0] * magnitude[:, 1]  # shape: (num_pairs)
        cos = dot_product / mag_AxB
        sin = cross_product / mag_AxB
        return sin, cos