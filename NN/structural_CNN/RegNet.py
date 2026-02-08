import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNetBlock(nn.Module):
    """RegNet中的残差块，采用组卷积结构"""

    def __init__(self, in_channels, out_channels, stride, groups, bottleneck_ratio=1.0):
        super(RegNetBlock, self).__init__()
        # 计算瓶颈层的通道数
        bottleneck_channels = int(out_channels * bottleneck_ratio)

        # 主分支：1x1卷积 -> 3x3组卷积 -> 1x1卷积
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                               stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        #  shortcut分支：当步长为2或输入输出通道不同时，使用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class RegNet(nn.Module):
    """RegNet网络结构"""

    def __init__(self, config, num_classes=1000):
        super(RegNet, self).__init__()
        # 解析配置参数
        self.depth = config['depth']  # 每个阶段的块数量
        self.width = config['width']  # 每个阶段的输出通道数
        self.stride = config['stride']  # 每个阶段的步长
        self.groups = config['groups']  # 每个阶段的组卷积数量
        self.bottleneck_ratio = config.get('bottleneck_ratio', 1.0)  # 瓶颈比率

        # Stem层：3x3卷积，步长2
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 构建各个阶段
        self.stages = nn.ModuleList()
        in_channels = 32  # Stem层的输出通道数

        for i in range(len(self.depth)):
            stage = self._make_stage(
                in_channels=in_channels,
                out_channels=self.width[i],
                num_blocks=self.depth[i],
                stride=self.stride[i],
                groups=self.groups[i]
            )
            self.stages.append(stage)
            in_channels = self.width[i]

        # Head层：全局平均池化 + 全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, groups):
        """构建一个阶段，包含多个RegNetBlock"""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []

        for stride in strides:
            blocks.append(RegNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=groups,
                bottleneck_ratio=self.bottleneck_ratio
            ))
            in_channels = out_channels  # 后续块的输入通道等于当前块的输出通道

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def regnet_configs(model_name='regnetx_400mf'):
    """预定义一些RegNet配置，对应不同大小的模型"""
    configs = {
        # RegNetX-400MF
        'regnetx_400mf': {
            'depth': [1, 2, 7, 1],
            'width': [32, 64, 160, 384],
            'stride': [1, 2, 2, 2],
            'groups': [16, 16, 16, 16],
            'bottleneck_ratio': 1.0
        },
        # RegNetX-800MF
        'regnetx_800mf': {
            'depth': [1, 3, 7, 1],
            'width': [64, 128, 288, 672],
            'stride': [1, 2, 2, 2],
            'groups': [16, 16, 16, 16],
            'bottleneck_ratio': 1.0
        }
    }
    return configs[model_name]


# 创建RegNet模型的函数
def regnet(model_name='regnetx_400mf', num_classes=1000):
    config = regnet_configs(model_name)
    return RegNet(config, num_classes)


# 测试代码
if __name__ == "__main__":
    # 创建一个RegNetX-400MF模型
    model = regnet('regnetx_400mf', num_classes=1000)

    # 生成随机输入 (batch_size=2, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)

    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 (2, 1000)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params / 1e6:.2f} M")

