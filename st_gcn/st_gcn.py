from st_gcn.layers.st_layer import STLayer
from torch import nn
import torch.nn.functional as F
from st_gcn.adjacency_matrix import AdjacencyMatrix

class STGCNetwork(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
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

        # TODO: 边权重，attention
        # if edge_importance_weighting:
        #     self.edge_importance = nn.ParameterList([
        #         nn.Parameter(torch.ones(self.A.size()))
        #         for i in self.st_gcn_networks
        #     ])
        # else:
        #     self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: N,C,T,V. T: Temporal features; V: Spatial features
        N, C, T, V = x.size()

        for layer in self.st_layers:
            x, _ = layer(x, self.A)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # shape: N, C, 1, 1
        # x = x.view(x.size(0), -1)  # shape: N, C

        # 此处可接LSTM，或接fcn
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x
