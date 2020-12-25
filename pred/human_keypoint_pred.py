from pathlib import Path

import torch
import numpy as np
from networks.pafs_resnet import ResnetPAFs
from networks.pafs_network import PAFsNetwork
from constants.enum_keys import PG
from constants.keypoints import aic_bones

class HumanKeypointPredict:
    def __init__(self):
        self.model_pose = PAFsNetwork(14, len(aic_bones))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_pose.to(self.device, dtype=torch.float)
        self.model_path = Path("../checkpoints/pose_model.pt")

        self.__load_model()

    def get_heatmaps(self, u8_image: np.ndarray):
        assert len(u8_image.shape) == 3, "expect npy image of shape (H, W, C)"
        assert u8_image.dtype == np.uint8, "expect uint8 image"
        norm_img = u8_image.astype(np.float32) / 255.
        norm_img = np.transpose(norm_img, axes=(2, 0, 1))[np.newaxis]  # HWC->NCHW
        norm_img = torch.from_numpy(norm_img)
        norm_img = norm_img.to(self.device)
        with torch.no_grad():
            _, _, b1_out, b2_out = self.model_pose(norm_img)
        return b1_out, b2_out

    def get_coordinates(self, norm_img: np.ndarray):
        # Output coordinates and norm_coordinates
        with torch.no_grad():
            b1, b2 = self.get_heatmaps(norm_img)
        b1, b2 = b1[0].cpu().numpy(), b2[0].cpu().numpy()
        chw_shape = b1.shape
        # TODO: Separate points for multiple person and compute mass center:
        # TODO: x = 1/(w*h) * integral_0^h integral_0^w u*Val(u,v) du dv
        max_indices = [np.argmax(hw, axis=None) for hw in b1]
        xs, ys = np.unravel_index(max_indices, b1[0].shape)  # (array([x, x, x]), array([y, y, y]))
        xs_norm, ys_norm = xs / chw_shape[2], ys / chw_shape[1]
        results = {PG.COORD_NATIVE: (xs, ys), PG.COORD_NORM: (xs_norm, ys_norm)}
        return results

    def __load_model(self):
        if Path.is_file(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model_pose.load_state_dict(checkpoint)
        else:
            print("No model file found.")