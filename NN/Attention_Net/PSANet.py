import torch
from torch import nn
import torch.nn.functional as F


class PSAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.collect = nn.Conv2d(out_channels, out_channels, 1)
        self.distribute = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        # 网络由collect和distribute两个平行分支构成
        # 在每个分支中，首先应用1×1的卷积来减少输入特征图X的通道数（从C1到C2)，以减少计算开销
        x = self.conv_reduce(x)
        b, c, h, w = x.size()

        # Collect
        x_collect = self.collect(x).view(b, c, -1)
        x_collect = F.softmax(x_collect, dim=-1)

        # Distribute
        x_distribute = self.distribute(x).view(b, c, -1)
        x_distribute = F.softmax(x_distribute, dim=1)

        # Attention
        x_att = torch.bmm(x_collect, x_distribute.permute(0, 2, 1)).view(b, c, h, w)

        return x + x_att