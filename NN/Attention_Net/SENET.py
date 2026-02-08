from torch import nn
class SENet(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SENet, self).__init__()
        self.ave_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(channel,channel//reduction,bias=False),
                              nn.ReLU(inplace=True),
                              nn.Linear(channel//reduction,channel,bias=False),
                              nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.ave_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)   # y.expand_as(x)指将y的尺寸改变为和x相同的尺寸。