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

    def handcrafted_features(self, coord_norm):
        assert len(coord_norm) == 2 and coord_norm[0].shape == coord_norm[1].shape
        feature_dict = {}
        bone_len = self.__bone_len(coord_norm)
        feature_dict[PG.BONE_LENGTH] = bone_len
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
        xs, ys = coord
