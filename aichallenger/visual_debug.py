from pathlib import Path

import visdom
import numpy as np
from torch.utils.data import DataLoader
import torch
from aichallenger import AicNorm

# TODO: Check resize bugs
class VisualDebug:
    def __init__(self):
        self.vis = visdom.Visdom()
        root = Path.home() / "AI_challenger_keypoint"
        train_dataset = AicNorm(root, True, (512, 512), (64, 64), visual_debug=True)
        self.res_dict = train_dataset[1]

    def show(self):
        self.vis.close()
        self._show_bgr_uint8('resized_img')
        self.vis.image(self.res_dict['norm_aug_img'][::-1, ...], win='norm_aug_img', opts={'title': 'norm_aug_img'})
        self._show_bgr_uint8('img_before_aug')
        self._show_bgr_uint8('img_after_aug')
        self._show_heat_amax_float('gau_vis_or_not')
        self._show_heat_amax_float('pafs_vis_or_not')

    def _show_bgr_uint8(self, name: str):
        # Expected input shape: HWC uint8
        self.vis.image(self.res_dict[name].transpose((2, 0, 1))[::-1, ...], win=name, opts={'title': name})

    def _show_heat_amax_float(self, name: str):
        # Expected input shape: CHW float
        self.vis.heatmap(np.flipud(np.amax(self.res_dict[name], axis=0)), win=name, opts={'title': name})

if __name__ == '__main__':
    VisualDebug().show()
