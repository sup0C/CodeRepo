import math
import torch
from torch import nn
class ECANet(nn.Module):
    def __init__(self,channel,gamma=2,b=1):
        super(ECANet, self).__init__()
        kernal_size=int(abs((math.log(channel,2)+b)/gamma))
        # print(kernal_size)
        kernal_size=kernal_size if kernal_size%2 else kernal_size+1
        # print(kernal_size)
        padding=kernal_size//2
        # print(padding)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernal_size,padding=padding,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # x: input features with shape [b, c, h, w]
        b,c,h,w=x.size()
#        a = self.avg_pool(x)
#       print(a.size())
        avg=self.avg_pool(x).view([b,1,c])
        print(avg.size())
        out=self.conv(avg)
        print(out.size())
        out=self.sigmoid(out).view([b,c,1,1])
        print(out.size())
        return out*x

if __name__ == '__main__':
    model=ECANet(512)
    print(model)
    input=torch.ones([2,512,26,26])
    output=model(input)
    print(output.size())