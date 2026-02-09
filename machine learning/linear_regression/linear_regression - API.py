import torch
from torch import nn
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

# 1. 定义数据
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.rand([5000, 1])
y = x * 3 + 12
x,y = x.to(device),y.to(device)

# 2 .定义模型
class Lr(nn.Module):
    def __init__(self):
        super(Lr, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


# 3. 实例化模型，loss，和优化器
model = Lr().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
# 4. 训练模型
for i in range(30000):
    out = model(x)  # 3.1 获取预测值
    loss = criterion(y, out)  # 3.2 计算损失
    optimizer.zero_grad()  # 3.3 梯度归零
    loss.backward()  # 3.4 计算梯度
    optimizer.step()  # 3.5 更新梯度
    if (i + 1) % 1000 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(i, 30000, loss.data))

# 5. 模型评估
model.eval()  # 设置模型为评估模式，即预测模式。`model.train(mode=True)` 表示设置模型为训练模式
x = torch.rand([500, 1])
y = x * 3 +12
x,y = x.to(device),y.to(device)
predict = model(x)
predict = predict.cpu().detach().numpy() #转化为numpy数组
plt.scatter(x.cpu().data.numpy(),y.cpu().data.numpy(),c="r")
plt.plot(x.cpu().data.numpy(),predict,)
plt.show()