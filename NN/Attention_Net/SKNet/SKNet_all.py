import torch
from torch import nn


class SKNet(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=(1, 1), L=32):
        '''
        :param features: input channel dimensionality
        :param M: the number of branchs
        :param G: num of convulution groups
        :param r: the ratio
        :param stride: default:1
        :param L:minumun im of the vector z in paper, default: 32
        '''
        super(SKNet, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=(3, 3), stride=stride, padding=i + 1, dilation=(i + 1, i + 1),
                          groups=G, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=(1, 1), stride=stride, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, features, kernel_size=(1, 1), stride=stride))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


# 将 ResNet的block中3*3卷积用SKconv来替代即可构建论文中的SKNet
class SKBlock(nn.Module):
    '''
    基于Res Block构造的SK Block
    ResNeXt有  1x1Conv（通道数：x） +  SKConv（通道数：x）  + 1x1Conv（通道数：2x） 构成
    '''
    expansion = 2  # 指 每个block中 通道数增大指定倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SKBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True))
        self.conv2 = SKConv(planes, planes, stride)
        # 与 ResNet block最大的区别就在于中间的这个3*3的卷积 使用 SkConv进行取代
        self.conv3 = nn.Sequential(nn.Conv2d(planes, planes * self.expansion, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, input):
        shortcut = input
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        if self.downsample is not None:
            shortcut = self.downsample(input)
        output += shortcut
        return self.relu(output)


if __name__ == '__main__':
    x = torch.Tensor(8, 32, 24, 24)
    conv = SKNet(32, 32, 1, 2, 16, 32)

    print('size', conv(x).size())  # size torch.Size([8, 32, 24, 24])
