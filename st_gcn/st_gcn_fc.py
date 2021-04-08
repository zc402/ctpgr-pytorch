import torch

from st_gcn.layers.st_layer import STLayer
from torch import nn
import torch.nn.functional as F
from st_gcn.adjacency_matrix import AdjacencyMatrix
from .stgcn_bone_network import StgcnBoneNetwork

class StgcnFc(nn.Module):
    """ STGCN, output num_class scores"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bone = StgcnBoneNetwork(in_channels)

        self.fcn = nn.Linear(256, 9)

    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        x = self.bone(x)

        # 把V平均、C连接dense
        x = x.mean(dim=3)  # NCT
        x = x.permute(0, 2, 1)  # NTC
        x = x.reshape([N*T, 256])  # N*T, C
        x = self.fcn(x)

        # x = x.view(N, T, C)
        return x
