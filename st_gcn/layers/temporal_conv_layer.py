"""
在时间维度上的卷积
"""
import torch
from torch import nn

class TCN(nn.Module):

    def __init__(self, channels: int, t_kernel_size: int, stride=1):
        super().__init__()

        assert t_kernel_size % 2 == 1
        # stride = 1
        padding = ((t_kernel_size - 1) // 2, 0)

        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                (t_kernel_size, 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(channels),
            # nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x):
        return self.temporal_conv(x)