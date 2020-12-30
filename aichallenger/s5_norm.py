from aichallenger import AicAffinityField
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List

from constants.enum_keys import HK


class AicNorm(AicAffinityField):
    """
    Normalize images to 0~1, converts from HWC to CHW
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, heat_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, heat_size, **kwargs)

    def __getitem__(self, index) -> dict:
        res_dict = super().__getitem__(index)
        norm_aug_img = res_dict[HK.AUG_IMAGE].astype(np.float) / 255.
        res_dict[HK.NORM_IMAGE] = norm_aug_img.transpose((2, 0, 1))  # HWC -> CHW

        # Each element in list of batch should be of equal size
        variable_len = [HK.NATIVE_IMAGE, HK.VISIBILITIES, HK.KEYPOINTS, HK.RE_KEYPOINTS, HK.AUG_KEYPOINTS,
                        HK.HEAT_KEYPOINTS, HK.RE_BOXES, HK.AUG_BOXES]

        [res_dict.pop(x) for x in variable_len]

        return res_dict

