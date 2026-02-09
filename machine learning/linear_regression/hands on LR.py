import torch
import numpy as np
from matplotlib import pyplot as plt

# 线性回归
# 准备数据 y = 3x+0.8，准备参数
x = torch.rand([50])
learning_rate=0.01
y = 3 * x + 0.8 # 目标拟合值
# 随机化初始参数
w = torch.rand([1], requires_grad=True)
b = torch.rand([1], requires_grad=True)
print('w,b:',w,b,w.grad)

def loss_fn(y, y_predict):
    loss = (y_predict - y).pow(2).mean()
    for i in [w, b]:
        # 每次反向传播前把上一次反向传播时各参数计算的梯度置为0
        if i.grad is not None:
            i.grad.data.zero_()
    # [i.grad.data.zero_() for i in [w,b] if i.grad is not None]
    loss.backward()
    return loss.data


def optimize(learning_rate=0.01):
    # print(w.grad.data,w.data,b.data)
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


for i in range(3000):
    # 1. 计算预测值
    y_predict = x * w + b
    # 2.计算损失，把参数的梯度置为0，进行反向传播
    loss = loss_fn(y, y_predict)
    # 3. 更新参数w和b
    optimize(learning_rate)
    # 4 每隔500epoch打印loss
    if i % 500 == 0:
        print('i:',i, 'loss:', loss)
        print("w:", w,"b:", b)

# 绘制图形，观察训练结束的预测值和真实值
predict = x * w + b  # 使用训练后的w和b计算预测值
plt.scatter(x.data.numpy(), y.data.numpy(), c="r")
plt.plot(x.data.numpy(), predict.data.numpy())
plt.show()