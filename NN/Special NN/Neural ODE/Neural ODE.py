# 定义ODE的动力学函数
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    '''
    这里f(h, t, θ)是个小神经网络，它描述了隐藏状态如何随时间变化。
    '''
    def __init__(self):
        super().__init__()
        # 定义参数化f_theta(h)的神经网络
        self.net = nn.Sequential(
            nn.Linear(2, 50),  # 层：从2D状态 -> 50个隐藏单元
            nn.Tanh(),  # 非线性激活以获得灵活性
            nn.Linear(50, 2)  # 层：从50个隐藏单元 -> 2D输出
        )

    def forward(self, t, h):
        """
        ODE函数的前向传播。
        参数：
            h : 当前状态，特征数为2的状态向量。（形状为[batch_size, 2]的张量）
            t : 当前时间（标量，odeint需要但这里未使用）
        返回：
            dh/dt : h的估计变化率。与h形状相同，也是大小2
        """
        return self.net(h)


h0 = torch.tensor([[2., 0.]])   # 起始点
t = torch.linspace(0, 25, 100)  # 时间步长
func = ODEFunc()   # 你的神经ODE动力学（dh/dt = f(h)）

# 求解ODE：
# 这样我们就把神经网络转换成了连续系统。
trajectory = odeint(func, h0, t)
print(trajectory.shape)  # (时间, 批次, 特征)