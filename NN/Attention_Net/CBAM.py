import torch
from torch import nn
class channel_attention(nn.Module):
    '''
    构建通道注意力
    '''
    def __init__(self,channel,ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(nn.Linear(channel,channel//ratio,False),
                              nn.ReLU(),
                              nn.Linear(channel//ratio,channel,False))
        # 或者用conv来生成通道注意力权重
        # self.fc = nn.Sequential(nn.Conv2d(channel,channel//ratio, 1, bias=False),
        #                         nn.ReLU(),
        #                         nn.Conv2d(channel//ratio,channel, 1, bias=False))
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        b,c,h,w=x.size()
        avg=self.avg_pool(x).view([b,c])
        max=self.max_pool(x).view([b,c])

        avg_fc=self.fc(avg)
        max_fc=self.fc(max)
        out=max_fc+avg_fc
        out=self.sigmoid(out).view([b,c,1,1]) # 通道注意力分数
        return x*out

class spacial_attention(nn.Module):
    def __init__(self,kernal_size=(7,7)):
        super(spacial_attention, self).__init__()
        self.conv=nn.Conv2d(2,1,stride=(1,1),
                            kernel_size=kernal_size,padding=kernal_size[0]//2,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        max_pool_out,_=torch.max(x,dim=1,keepdim=True)  # 保持维度不变
        mean_pool_out=torch.mean(x,dim=1,keepdim=True)  # 保持维度不变
        pool_out=torch.cat([max_pool_out,mean_pool_out],dim=1)
        out=self.conv(pool_out)
        out=self.sigmoid(out)  # 生成空间注意力分数
        return out*x

class CBAM(nn.Module):
    def __init__(self,channel,ratio=16,kernal_size=(7,7)):
        super(CBAM, self).__init__()
        self.channel_attention=channel_attention(channel,ratio)
        self.spacial_attention=spacial_attention(kernal_size)

    def forward(self,x):
        out=self.channel_attention(x)
        out=self.spacial_attention(out)
        return out

if __name__ == '__main__':
    model=CBAM(512)
    print(model)
    input=torch.ones([2,512,25,25])
    output=model(input)
    print(output.size())