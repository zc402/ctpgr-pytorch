from aichallenger import AicGaussian
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List


class AicAffinityField(AicGaussian):
    """
    Construct "pafs"
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, heat_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, heat_size, **kwargs)
        self.__paf_line_width = 2
        self.__paf_generator = PartAffinityFieldGenerator(heat_size, self.__paf_line_width)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        pafs = self.__get_pafs_groundtruth(res_dict["aug_label"])
        res_dict["pafs"] = pafs
        return res_dict

    def __get_pafs_groundtruth(self, crowd: Crowd) -> np.ndarray:
        num_people = len(crowd)
        connections = [[1, 2], [2, 3], [4, 5], [5, 6], [14, 1], [14, 4], [7, 8], [8, 9], [10, 11], [11, 12],
                       [13, 14]]
        connections = np.asarray(connections, np.int) - 1
        zero_heat = np.zeros((self.heat_size[1], self.heat_size[0]), np.float)
        connect_heats = []  # Expected shape: (connections, H, W)
        for j1, j2 in connections:
            person_heats = []  # Expected shape: (person, H, W)
            for p in range(num_people):
                vis1 = crowd[p].joints[j1].v
                vis2 = crowd[p].joints[j2].v
                p1 = (crowd[p].joints[j1].x, crowd[p].joints[j1].y)
                p2 = (crowd[p].joints[j2].x, crowd[p].joints[j2].y)

                if vis1 == 1 and vis2 == 1:  # Both visible
                    person_paf = self.__paf_generator.gen_field_adjust_pts(p1, p2, self.resize_img_size)
                else:
                    person_paf = zero_heat
                person_heats.append(person_paf)
            img_heat = np.amax(person_heats, axis=0)
            connect_heats.append(img_heat)
        connect_heats = np.asarray(connect_heats, dtype=np.float)
        return connect_heats


# Generate a line with adjustable width. (float image 0~1 ranged)
class PartAffinityFieldGenerator:
    def __init__(self, heat_size: Tuple[int, int], thickness: int):
        self.thickness = thickness
        self.heat_size = heat_size  # Heatmap image size

    def gen_field(self, pt1: Tuple[int, int], pt2: Tuple[int, int]):
        canvas = np.zeros(self.heat_size, dtype=np.uint8)
        cv2.line(canvas, pt1, pt2, 255, self.thickness)

        # Convert to [0,1]
        canvas = canvas.astype(np.float)
        canvas = canvas / 255.
        return canvas

    def gen_field_adjust_pts(self, pt1: Tuple[int, int], pt2: Tuple[int, int], img_size):
        img_w, img_h = img_size
        w, h = self.heat_size
        ratio_w = w / img_w
        ratio_h = h / img_h
        new_pt1 = np.array(pt1) * (ratio_w, ratio_h)
        new_pt1 = new_pt1.astype(np.int)
        new_pt1 = tuple(new_pt1)
        new_pt2 = np.array(pt2) * (ratio_w, ratio_h)
        new_pt2 = new_pt2.astype(np.int)
        new_pt2 = tuple(new_pt2)
        pafs = self.gen_field(new_pt1, new_pt2)
        return pafs