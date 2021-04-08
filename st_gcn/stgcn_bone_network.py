import torch

from st_gcn.layers.st_layer import STLayer
from torch import nn
import torch.nn.functional as F
from st_gcn.adjacency_matrix import AdjacencyMatrix

class StgcnBoneNetwork(nn.Module):
    """ STGCN bone network"""

    def __init__(self, in_channels):
        super().__init__()
        edge_importance_weighting = True
        A = AdjacencyMatrix().get_height_config_adjacency()
        self.register_buffer('A', A)
        num_spatial_labels = A.size(0)

        self.st_layers = nn.ModuleList((
            STLayer(in_channels, 64, num_spatial_labels),
            STLayer(64, 64,   num_spatial_labels),
            STLayer(64, 64,   num_spatial_labels),
            STLayer(64, 64,   num_spatial_labels),
            STLayer(64, 128,  num_spatial_labels),
            STLayer(128, 128, num_spatial_labels),
            STLayer(128, 128, num_spatial_labels),
            STLayer(128, 256, num_spatial_labels),
            STLayer(256, 256, num_spatial_labels),
            STLayer(256, 256, num_spatial_labels),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_layers
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        # for layer in self.st_layers:
        #     x, _ = layer(x, self.A)  # x:NCTV

        for gcn, importance in zip(self.st_layers, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x
