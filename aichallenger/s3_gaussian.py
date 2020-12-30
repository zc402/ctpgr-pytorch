from aichallenger import AicAugment
from pathlib import Path
import cv2
import numpy as np
from aichallenger.defines import Box, Joint, Person, Crowd
from typing import Tuple, List

from constants.enum_keys import HK
from imgaug.augmentables.heatmaps import HeatmapsOnImage

class AicGaussian(AicAugment):
    """
    Provides gaussian heatmaps for keypoints, ranged 0~1
    """

    def __init__(self, data_path: Path, is_train: bool, resize_img_size: tuple, heat_size: tuple, **kwargs):
        super().__init__(data_path, is_train, resize_img_size, **kwargs)
        self.heat_size = heat_size
        assert resize_img_size[0] % heat_size[0] == 0 and resize_img_size[1] % heat_size[1] == 0, \
            "Incorrect heat size: resize_img_size must be divisible by heat_size."
        self.h_r_ratio = np.array(self.heat_size, dtype=np.float32) / np.array(self.resize_img_size, dtype=np.float32)
        self.__theta = 4

    def __getitem__(self, index) -> dict:
        res = super().__getitem__(index)
        heat_keypoints = res[HK.AUG_KEYPOINTS] * self.h_r_ratio  # Shape: (P, J, X)
        res[HK.HEAT_KEYPOINTS] = heat_keypoints

        w, h = self.heat_size
        xy_mesh = np.meshgrid(np.arange(0, w), np.arange(0, h))  # shape: (X, H, W)
        xy_mesh = np.asarray(xy_mesh, dtype=np.float32)
        distance_map = heat_keypoints[..., np.newaxis][..., np.newaxis] - xy_mesh  # Shape: (PJXHW)
        pcm_map = np.exp(-(np.square(distance_map).sum(axis=2)) / np.square(self.__theta))  # Shape: (PJHW)
        vis = res[HK.VISIBILITIES]  # Shape: (PJ)
        vis = np.expand_dims(vis, (2, 3))  # Shape: (PJHW)
        # Occlusion masks
        vis_all = np.logical_or(vis == 1, vis == 2).astype(np.float32)  # Shape: (PJHW)
        vis_no_occ = (vis == 1).astype(np.float32)  # Shape: (PJHW)
        pcm_vis_all = pcm_map * vis_all
        pcm_vis_no_occ = pcm_map * vis_no_occ
        # Marge same joints from different person
        pcm_vis_all = np.amax(pcm_vis_all, axis=0)  # Shape: (JHW)
        pcm_vis_no_occ = np.amax(pcm_vis_no_occ, axis=0)
        res[HK.PCM_ALL] = pcm_vis_all
        res[HK.PCM_NOT_OCC] = pcm_vis_no_occ
        # Visual Debug
        if 'visual_debug' in self.kwargs and self.kwargs.get('visual_debug'):
            debug_pcm_all = pcm_vis_all.max(axis=0, initial=0)  # HW
            debug_pcm_all = HeatmapsOnImage(debug_pcm_all, shape=self.heat_size, min_value=0.0, max_value=1.0)
            res[HK.DEBUG_PCM_ALL] = debug_pcm_all.draw_on_image(res[HK.AUG_IMAGE])[0]

            debug_pcm_noocc = pcm_vis_no_occ.max(axis=0, initial=0)
            debug_pcm_noocc = HeatmapsOnImage(debug_pcm_noocc, shape=self.heat_size, min_value=0.0, max_value=1.0)
            res[HK.DEBUG_PCM_NOOCC] = debug_pcm_noocc.draw_on_image(res[HK.AUG_IMAGE])[0]

        return res
