import torch
from torch import nn
class Res2NetBlock(nn.Module):
    def __init__(self,in_channels,middle_channles,scales=4,expansion=1,stride=(1,1),bn_layer=None):
        '''
        :param in_channels:输入和最终输出通道数
        :param middle_channles: 第一次1x1卷积后输出的通道数,中转通道
        :param scales:将输出通道分为几块
        :param expansion：将out_channles扩大n倍
        '''
        super(Res2NetBlock, self).__init__()
        if middle_channles%scales!=0: # 输出通道为4的倍数
            raise ValueError('plase must be divisible by scales')
        if bn_layer:
            bn_layer=nn.BatchNorm2d
        bottle_neck_channel=expansion*middle_channles
        self.scales=scales
        self.relu=nn.ReLU()
        self.stride=stride
        # 对所用数据进行一次1x1卷积
        self.conv1_1x1=nn.Conv2d(in_channels,bottle_neck_channel,kernel_size=(1,1),stride=stride)
        self.bn1=bn_layer(bottle_neck_channel)
        # 3*3卷积，通道交互
        self.conv2=nn.ModuleList([nn.Conv2d(bottle_neck_channel//scales,bottle_neck_channel//scales,
                                            kernel_size=(3,3),stride=stride,padding=(1,1))
                                  for _ in range(scales-1)])
        self.bn2=nn.ModuleList([bn_layer(bottle_neck_channel//scales) for _ in range(scales-1)])
        # 交互后的1x1卷积
        self.conv3_1x1=nn.Conv2d(bottle_neck_channel,in_channels,kernel_size=(1,1),stride=stride)
        self.bn3=bn_layer(in_channels)

    def forward(self,x):
        identity=x
        # 1x1卷积
        out=self.relu(self.bn1(self.conv1_1x1(x)))
        # 通道交互
        xs=torch.chunk(out,self.scales,1) # 在通道方向上分割为scales份
        ys=[]  # 通道上每个组3x3后的集合
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        #   通道交互后的合并
        out = torch.cat(ys, dim=1)
        out = self.bn3(self.conv3_1x1(out))
        return self.relu(out+identity)