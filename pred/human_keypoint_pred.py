from pathlib import Path

import torch
import numpy as np
from models.pafs_resnet import ResnetPAFs
from models.pafs_network import PAFsNetwork
from constants.enum_keys import PG, HK
from constants.keypoints import aic_bones
from models.pose_estimation_model import PoseEstimationModel


class HumanKeypointPredict:
    def __init__(self):
        self.model_pose = PoseEstimationModel()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_pose.to(self.device, dtype=torch.float)
        self.model_pose.load_ckpt(allow_new=False)
        self.model_pose.eval()

    def get_heatmaps(self, u8_image: np.ndarray):
        """
        Get PCM and PAF heatmap
        :param u8_image:
        :return: {HK.B1_OUT, HK.B2_OUT}
        """
        assert len(u8_image.shape) == 3, "expect npy image of shape (H, W, C)"
        assert u8_image.dtype == np.uint8, "expect uint8 image"
        norm_img = u8_image.astype(np.float32) / 255.
        norm_img = np.transpose(norm_img, axes=(2, 0, 1))[np.newaxis]  # HWC->NCHW
        norm_img = torch.from_numpy(norm_img)
        norm_img = norm_img.to(self.device)
        with torch.no_grad():
            res = self.model_pose(norm_img)
            del res[HK.B1_SUPERVISION]
            del res[HK.B2_SUPERVISION]
        return res

    def get_coordinates(self, norm_img: np.ndarray):
        # Output coordinates and norm_coordinates
        res = self.get_heatmaps(norm_img)
        b1, b2 = res[HK.B1_OUT][0].cpu().numpy(), res[HK.B2_OUT][0].cpu().numpy()
        chw_shape = b1.shape
        # TODO: Separate points for multiple person and compute mass center:
        # TODO: x = 1/(w*h) * integral_0^h integral_0^w u*Val(u,v) du dv
        max_indices = [np.argmax(hw, axis=None) for hw in b1]
        ys, xs = np.unravel_index(max_indices, b1[0].shape)  # array([y, y, y])), (array([x, x, x])
        xs_norm, ys_norm = xs / chw_shape[2], ys / chw_shape[1]
        # PG.COORD_NATIVE shape: (xy(2), num_keypoints)
        results = {PG.COORD_NATIVE: np.array((xs, ys)), PG.COORD_NORM: np.array((xs_norm, ys_norm))}
        return results
