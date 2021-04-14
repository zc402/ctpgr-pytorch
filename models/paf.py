# The model used for human pose estimation in this project
import torch
from torch import nn
from constants.keypoints import aic_bones
from pathlib import Path
from constants.enum_keys import HK
from keypoint_network.pafs_network import PAFsNetwork
from models import BaseModel


class PAF(BaseModel):
    """
    Part Affinity Field
    """
    def __init__(self):
        super().__init__()
        self.model_pose = PAFsNetwork(14, len(aic_bones))
        self._to_device()

    def _get_model_name(self) -> str:
        return "PAF"

    def forward(self, img):
        b1_stages, b2_stages, b1_out, b2_out = self.model_pose(img)
        res_dict = {HK.B1_SUPERVISION: b1_stages, HK.B2_SUPERVISION: b2_stages,
                    HK.B1_OUT: b1_out, HK.B2_OUT: b2_out}
        return res_dict
