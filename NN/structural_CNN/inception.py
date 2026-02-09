class Inception(nn.Module):
    # 输出了128个通道：32*4=128
    def __init__(self, in_channel):
        super(Inception, self).__init__()
        self.branch1x1 = Conv2d(in_channel, 32, kernel_size=(1, 1))

        self.branch5x5_1 = Conv2d(in_channel, 16, kernel_size=(1, 1))
        #         self.branch5x5_2=Conv2d(16,32,kernel_size=(1,5),padding=(2,0))
        self.branch5x5_2 = Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2))

        self.branch3x3_1 = Conv2d(in_channel, 16, kernel_size=(1, 1))
        #         self.branch3x3_2=Conv2d(16,24,kernel_size=(1,3),padding=(1,0))
        #         self.branch3x3_3=Conv2d(24,32,kernel_size=(1,3),padding=(1,0))
        self.branch3x3_2 = Conv2d(16, 24, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_3 = Conv2d(24, 32, kernel_size=(1, 3), padding=(0, 1))

        self.branch_pool_1 = nn.AvgPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch_pool_2 = Conv2d(in_channel, 32, kernel_size=(1, 1))

    def forward(self, input):
        branch1x1 = self.branch1x1(input)

        branch5x5 = self.branch5x5_1(input)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(input)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = self.branch_pool_1(input)
        branch_pool = self.branch_pool_2(branch_pool)
        #         print(input.shape,branch1x1.shape,branch5x5.shape,branch3x3.shape,branch_pool.shape,'*******************************')

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=(1, 5))
        self.conv2 = Conv2d(128, 64, kernel_size=(1, 5))  # 88通道数是根据数据经过Inception后得到的
        self.conv3 = Conv2d(128, 32, kernel_size=(1, 5))
        #         self.conv4=Conv2d(32,16,kernel_size=(1,5))

        self.incep1 = Inception(in_channel=32)
        self.incep2 = Inception(in_channel=64)

        self.maxpool = MaxPool2d((1, 2), ceil_mode=True)
        self.linear = Linear(4000, 12)

    #         self.linear2=Linear(128,12)

    def forward(self, input):
        in_size = input.size(0)
        # output=F.sigmoid(self.maxpool(self.conv1(input)))
        output = self.maxpool(self.conv1(input))
        #         output=F.softmax(output)
        output = self.incep1(output)
        # output = F.tanh(self.maxpool(self.conv2(output)))
        output = self.maxpool(self.conv2(output))
        output = self.incep2(output)
        output = self.maxpool(self.conv3(output))
        output = F.softmax(output)
        #         output=self.maxpool(self.conv4(output))
        output = output.view(in_size, -1)  # 将数据形状改为(batch_size,-1),然后送入全连接层
        output = self.linear(output)
        #         output=self.linear2(output)
        return output