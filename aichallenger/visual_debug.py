from pathlib import Path

import visdom
import numpy as np
from torch.utils.data import DataLoader
import torch
from aichallenger import AicNorm
from constants.enum_keys import HK


class VisualDebug:
    def __init__(self):
        self.vis = visdom.Visdom()


    def show(self):
        root = Path.home() / "AI_challenger_keypoint"
        train_dataset = AicNorm(root, True, (512, 512), (64, 64), visual_debug=True)
        # dl = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
        res = train_dataset[1]

        self.vis.close()
        self.vis.image(res[HK.DEBUG_RE_IMAGE].transpose((2, 0, 1))[::-1, ...], opts={'title': 'RESIZE'})
        self.vis.image(res[HK.DEBUG_AUG_IMAGE].transpose((2, 0, 1))[::-1, ...], opts={'title': 'AUG'})
        self.vis.images(res[HK.DEBUG_PCM_ALL].transpose((2, 0, 1)), opts={'title': 'PCM_ALL'})
        self.vis.images(res[HK.DEBUG_PCM_NOOCC].transpose((2, 0, 1)), opts={'title': 'PCM_NOOCC'})
        self.vis.images(res[HK.DEBUG_PAF_ALL].transpose((2, 0, 1)), opts={'title': 'PAF_ALL'})

    # def _show_bgr_uint8(self, name: str):
    #     # Expected input shape: HWC uint8
    #     self.vis.image(self.res_dict[name].transpose((2, 0, 1))[::-1, ...], win=name, opts={'title': name})
    #
    # def _show_heat_amax_float(self, name: str):
    #     # Expected input shape: CHW float
    #     self.vis.heatmap(np.flipud(np.amax(self.res_dict[name], axis=0)), win=name, opts={'title': name})

if __name__ == '__main__':
    VisualDebug().show()
