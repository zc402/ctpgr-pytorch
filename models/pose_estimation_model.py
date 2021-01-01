# The model used for human pose estimation in this project
import torch
from torch import nn
from torch.nn import LSTM
from constants.keypoints import aic_bones, aic_bone_pairs
from pathlib import Path
from constants.enum_keys import HK
from models.pafs_network import PAFsNetwork


class PoseEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ckpt_path = Path("checkpoints/pose_model.pt")
        self.model_pose = PAFsNetwork(14, len(aic_bones))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device, dtype=torch.float32)

    def save_ckpt(self):
        torch.save(self.state_dict(), self.ckpt_path)
        print('Pose model ckpt saved.')

    def load_ckpt(self, allow_new=True):
        if Path.is_file(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.load_state_dict(checkpoint)
            print('Pose model ckpt loaded.')
        else:
            if allow_new:
                print('Pose model ckpt not found.')
            else:
                raise FileNotFoundError('Pose model ckpt not found.')

    def forward(self, img):
        b1_stages, b2_stages, b1_out, b2_out = self.model_pose(img)
        res_dict = {HK.B1_SUPERVISION: b1_stages, HK.B2_SUPERVISION: b2_stages,
                    HK.B1_OUT: b1_out, HK.B2_OUT: b2_out}
        return res_dict
