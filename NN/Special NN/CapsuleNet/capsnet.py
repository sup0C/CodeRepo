import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    '''
    卷积+relu。前置预处理一下输入
    '''
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        # x.shape=(32,1,28,28)
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        # 动态路由算法。通常包括一个或多个迭代过程，用于计算每个底层胶囊应该传递多少信息给每个上层胶囊。
        # 胶囊层，这通常由多个单独的胶囊组成。每个胶囊都是一个小型神经网络，可以通过标准的全连接层或卷积层来实现。
        # x.shape = (32, 256, 20, 20)=(32,in_channels,20,20)
        # output.shape = （32,32*6*6，8）=(32,num_routes,num_capsules)
        u = [capsule(x) for capsule in self.capsules] # 多个Conv2d卷积层。u.shape=（32,32,6,6）
        u = torch.stack(u, dim=1) # u.shape=（32,8，32,6,6）
        u = u.view(x.size(0), self.num_routes, -1) # u.shape=（32,32*6*6，8）
        return self.squash(u)

    def squash(self, input_tensor): # squash非线性激活函数。input_tensor.shape=（32,32*6*6，8）
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) # （32,32*6*6，1）。模长。
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm)) # （32,32*6*6，8）。
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16,epsilon=1e-8):
        super(DigitCaps, self).__init__()
        self.in_channels = in_channels # 8
        self.num_routes = num_routes
        self.num_capsules = num_capsules # 10
        # shape=（1,32*6*6，10,16，8）= （1,num_routes,num_capsules,out_channels,in_channels）
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        self.epsilon = epsilon # 计算∥s_j∥时加了一个epsilon以确保它不会变为零。如果它变为0，它开始给出nan值，训练失败。
    def forward(self, x):
        # 胶囊间的动态路由算法。x.shape=（32,32*6*6，8）;
        batch_size = x.size(0) # 32
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4) # 将输入复制为10份，即10个胶囊。shape=（batch_size,32*6*6，10,8,1）=（batch_size,num_routes,num_capsules,in_channels）
        # 10份输入的权重
        W = torch.cat([self.W] * batch_size, dim=0)  # shape=（batch_size,32*6*6，10,16，8）=（batch_size,num_routes,num_capsules,out_channels,in_channels）
        u_hat = torch.matmul(W, x) # 对10份输入分别进行线性变换。shape=（batch_size,32*6*6，10,16，1）=（batch_size,num_routes，num_capsules,out_channels，1）
        # 初始化路由权重为0。b_ij.shape=（1,32*6*6，10,1）=(1,num_routes，num_capsules,1)。
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()
        # 论文里描述的路由机制，可以在模型中使用多个路由层
        num_iterations = 3
        for iteration in range(num_iterations):
            # 计算低层胶囊向量i的对应所有高层胶囊的权重，即一个样本或每个胶囊的归一化的路由权重。
            c_ij = F.softmax(b_ij, dim=1) # （1,32*6*6，10,1）=(1,num_routes，num_capsules,1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4) # 所有样本的归一化的路由权重。（batch_size,num_routes，num_capsules,1,1）=（batch_size,32*6*6，10,1,1）
            # 计算经前一步确定的路由系数ci加权后的输入向量的总和，得到输出向量sj。加权后的输出。
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) # （batch_size,1，num_capsules,out_channels,1）
            # 经过非线性变换得到高层胶囊的向量vj
            v_j = self.squash(s_j) # （batch_size,1，num_capsules,out_channels,1）
            # 更新原来的权重bij
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1)) # （batch_size,num_routes,num_capsules,1,1）
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True) # 低层向量对高层胶囊的权重. (1,num_routes，num_capsules,1)。

        return v_j.squeeze(1) # （batch_size,num_capsules,out_channels,1）=(32,10,16,1)

    def squash(self, input_tensor): # 论文里的squashing 非线性函数
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm+self.epsilon))
        return output_tensor


class Decoder(nn.Module):
    '''
    线性层，为了计算重构损失设置的
    '''
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_width * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        '''
        x.shape=(32,10,16,1);data:(batch_size,1,28,28);
        reconstructions=（32,1，28,28）, masked= (32,10)
        '''
        classes = torch.sqrt((x ** 2).sum(2)) # (32,10,1)
        classes = F.softmax(classes, dim=0) # (32,10,1)

        _, max_length_indices = classes.max(dim=1) #  (32,1)
        masked = Variable(torch.sparse.torch.eye(10)) # (10,10)
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1) # （32,160)
        reconstructions = self.reconstraction_layers(t) # （32,784)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height) # （32,1，28,28）
        return reconstructions, masked


class CapsNet(nn.Module):
    '''
    主网络
    网络顺序：decoder(digit_capsules(primary_capsules(conv_layer(data))))
    '''
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        '''
        data:(batch_size,1,28,28),output.shape=(32,10,16,1)
        reconstructions=（32,1，28,28）, masked= (32,10)
        '''
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        '''
        data：原始图像。(batch_size,1,28,28)
        x：(32,10,16,1)
        reconstructions：线性层的输出。（32,1，28,28）
        总loss
        '''
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        '''
        x: (32,10,16,1)
        (32,10)
        '''
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))# (32,10,1,1)

        left = F.relu(0.9 - v_c).view(batch_size, -1) # (32,10)
        right = F.relu(v_c - 0.1).view(batch_size, -1) # (32,10)

        loss = labels * left + 0.5 * (1.0 - labels) * right  # (32,10)
        loss = loss.sum(dim=1).mean()  # (1,)

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005
