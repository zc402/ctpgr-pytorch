"""
在空间维度上的图卷积，时间维度无关
"""
import torch
from torch import nn

class GCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_spatial_labels: int):
        """num_spatial_labels: 卷积核划分的标签数量（我的方法：low eq high 3种）"""
        super().__init__()

        self.num_spatial_labels = num_spatial_labels
        # 扩展channels，倍数等于划分标签数。相当于给每个channel分配3个卷积核，分别对应3个标签。
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_spatial_labels,
            kernel_size=(1, 1)
        )

    def forward(self, x, A):
        # x shape: N,C,T,V
        assert A.size(0) == self.num_spatial_labels
        x = self.conv(x)
        # n: batch; k: 划分标签数量; c: channels; t: 时间维度; v:关键点数量
        n, kc, t, v = x.size()
        x = x.view(n, self.num_spatial_labels, kc // self.num_spatial_labels, t, v)
        # v把一种标签的多个邻接结果数值加一起，k把每种标签结果数值加一起。w代表对每个节点都这么做。
        x = torch.einsum('nkctv,kvw->nctw', x, A)

        return x.contiguous(), A