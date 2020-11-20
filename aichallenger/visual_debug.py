from pathlib import Path

import visdom
import numpy as np
from aichallenger import AicNorm
class VisualDebug:
    def __init__(self):
        self.vis = visdom.Visdom()
        root = Path.home() / "AI_challenger_keypoint"
        train_dataset = AicNorm(root, True, (512, 512), (64, 64), visual_debug=True)
        self.res_dict = train_dataset[0]

    def show(self):

        # self._show_bgr_uint8(res_dict['native_img'], 'native_img')
        # self._show_bgr_uint8('resized_img')
        # self._show_bgr_uint8('aug_img')
        self._show_bgr_uint8('img_before_aug')
        self._show_bgr_uint8('img_after_aug')
        self._show_heat_amax_float('gau_vis')
        self._show_heat_amax_float('pafs')

    def _show_bgr_uint8(self, name: str):
        # Expected input shape: HWC uint8
        self.vis.image(self.res_dict[name].transpose((2, 0, 1))[::-1, ...], win=name, opts={'title': name})

    def _show_heat_amax_float(self, name: str):
        # Expected input shape: CHW float
        self.vis.heatmap(np.amax(self.res_dict[name], axis=0), win=name, opts={'title': name})

if __name__ == '__main__':
    VisualDebug().show()
