from aichallenger import AicGaussian
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defination import Box, Joint, Person, Crowd
from typing import Tuple, List


class AicAffinityField(AicGaussian):
    """
    Construct "pafs"
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, heat_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, heat_size, **kwargs)
        self.__paf_line_width = 3
        self.__paf_generator = PartAffinityFieldGenerator(heat_size, self.__paf_line_width)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        pafs = self.__get_pafs_groundtruth(res_dict["aug_label"])
        res_dict["pafs"] = pafs
        return res_dict

    def __get_pafs_groundtruth(self, crowd: Crowd) -> List[np.ndarray]:
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
                p1, p2 = crowd[p].joints[j1][:2], crowd[p].joints[j2][:2]
                if vis1 == 1 and vis2 == 1:  # Both visible
                    person_paf = self.__paf_generator.gen_field(p1, p2)
                else:
                    person_paf = zero_heat
                person_heats.append(person_paf)
            img_heat = np.amax(person_heats, axis=0)
            connect_heats.append(img_heat)
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
