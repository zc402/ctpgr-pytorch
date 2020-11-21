from aichallenger import AicAffinityField
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List

class AicNorm(AicAffinityField):
    """
    Normalize images to 0~1, converts from HWC to CHW
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, heat_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, heat_size, **kwargs)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        norm_aug_img = res_dict['aug_img'].astype(np.float) / 255.
        res_dict['norm_aug_img'] = norm_aug_img.transpose((2, 0, 1))  # HWC -> CHW

        return res_dict

