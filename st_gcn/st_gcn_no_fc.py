import torch

from st_gcn.layers.st_layer import STLayer
from torch import nn
import torch.nn.functional as F
from st_gcn.adjacency_matrix import AdjacencyMatrix
from st_gcn.stgcn_bone_network import StgcnBoneNetwork


class StgcnNoFc(nn.Module):
    """STGCN, no fully connected layer, output N, C"""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.bone = StgcnBoneNetwork(in_channels)

    def forward(self, x):
        x = self.bone(x)

        N, C, T, V = x.size()

        x = x.permute(0, 2, 1, 3)  # N, C, T, V -> N, T, C, V
        x = torch.mean(x, dim=3)  # NTC
        return x
