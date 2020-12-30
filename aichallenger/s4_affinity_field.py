from imgaug import HeatmapsOnImage

from aichallenger import AicGaussian
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List

from constants.enum_keys import HK
from constants.keypoints import aic_bones

class AicAffinityField(AicGaussian):
    """
    Construct "pafs"
    """
    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, heat_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, heat_size, **kwargs)
        self.__paf_line_width = 2

    def __getitem__(self, index) -> dict:
        res = super().__getitem__(index)
        heat_keypoints = res[HK.HEAT_KEYPOINTS]  # Shape: (PeJX) Person,Joint,X
        visibility = res[HK.VISIBILITIES]  # Shape: (PeJ)
        # aic_bones.shape: (BE) Bone,Endpoint
        bones = np.array(aic_bones) - 1
        coord = np.take(heat_keypoints, bones, axis=1)  # Shape:(Person, Bone, Endpoint, X)
        map_PeBHW = [cv2.line(np.zeros(self.heat_size, dtype=np.uint8), (x1, y1), (x2, y2), 255, self.__paf_line_width)
                       for x1, y1, x2, y2 in coord.reshape(-1, 4).astype(np.int)]
        # To 0~1
        map_PeBHW = np.asarray(map_PeBHW, dtype=np.float32) / 255.
        map_PeBHW = map_PeBHW.reshape(coord.shape[0], coord.shape[1], self.heat_size[1], self.heat_size[0])
        # Masks
        j_vis = np.take(visibility, bones, axis=1)  # Shape:(Person, Bone, Endpoint)
        j_vis_all = np.logical_or(j_vis == 1, j_vis == 2)  # Shape: (PeBE)
        b_vis_all = np.logical_and(j_vis_all[..., 0], j_vis_all[..., 1]).astype(np.float32)  # Shape: (PeB)
        b_vis_all = np.expand_dims(b_vis_all, (2, 3))  # Shape: (PeBHW)
        j_vis_noocc = (j_vis == 1)  # Shape: (PeBE)
        b_vis_noocc = np.logical_and(j_vis_noocc[..., 0], j_vis_noocc[..., 1]).astype(np.float32)  # Shape: (PeBE)
        b_vis_noocc = np.expand_dims(b_vis_noocc, (2, 3))  # Shape: (PeBHW)
        paf_vis_all = map_PeBHW * b_vis_all
        paf_vis_noocc = map_PeBHW * b_vis_noocc
        paf_vis_all = paf_vis_all.max(axis=0, initial=0)  # Shape: (BHW)
        paf_vis_noocc = paf_vis_noocc.max(axis=0, initial=0)
        res[HK.PAF_ALL] = paf_vis_all  # Shape: (BHW)
        res[HK.PAF_NOT_OCC] = paf_vis_noocc

        if 'visual_debug' in self.kwargs and self.kwargs.get('visual_debug'):
            debug_paf_all = paf_vis_all.max(axis=0, initial=0)  # HW
            debug_paf_all = HeatmapsOnImage(debug_paf_all, shape=self.heat_size, min_value=0.0, max_value=1.0)
            res[HK.DEBUG_PAF_ALL] = debug_paf_all.draw_on_image(res[HK.AUG_IMAGE])[0]
        return res
