"""
一层时空图卷积，先调用gcn，再调用tcn
"""
from torch import nn
from st_gcn.layers.spatial_conv_layer import GCN
from st_gcn.layers.temporal_conv_layer import TCN


class STLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_spatial_labels: int):
        super().__init__()

        self.gcn = GCN(in_channels, out_channels, num_spatial_labels)
        self.tcn = TCN(out_channels, t_kernel_size=9)
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        resi = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + resi
        x = self.relu(x)

        return x, A