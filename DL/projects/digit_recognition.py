import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision

# 60000个训练样本和10000个测试样本，图片尺寸都是黑白图，大小为28X28，共十类。
train_batch_size = 64
test_batch_size = 1000
img_size = 28


def get_dataloader(train=True):
    '''
    获取训练和测试的dataloader
    :param train:
    :return:
    '''
    assert isinstance(train, bool), "train 必须是bool类型"
    # 准备数据集，其中0.1307，0.3081为MNIST数据的均值和标准差，这样操作能够对其进行标准化
    # 因为MNIST只有一个通道（黑白图片）,所以元组中只有一个值
    dataset = torchvision.datasets.MNIST(r'./datasets/number_recognition', train=train, download=False,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize((0.1307,), (0.3081,)), ]))
    # 准备数据迭代器
    batch_size = train_batch_size if train else test_batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = self.fc1(x)  # [batch_size,28]
        x = F.relu(x)  # [batch_size,28]
        x = self.fc2(x)  # [batch_size,10]
        return F.log_softmax(x, dim=-1)

    def generate_equation(self):
        # 自定义函数来生成函数表达式字符串
        # 获取第一层和第二层的权重和偏置
        w1 = self.fc1.weight.item()
        b1 = self.fc1.bias.item()

    w2 = self.fc2.weight.item()
    b2 = self.fc2.bias.item()
    # 生成函数表达式字符串
    equation = f"f(x) = log_softmax({w2:.4f} * relu({w1:.4f} * x + {b1:.4f}) + {b2:.4f})"
    return equation


def train(epoch):
    mode = True
    mnist_net.train(mode=mode)
    train_dataloader = get_dataloader(train=mode)
    print('len(train_dataloader.dataset)[所有训练数据],len(train_dataloader)[batch的数量]', \
          len(train_dataloader.dataset), len(train_dataloader))
    for idx, (data, target) in enumerate(train_dataloader):
        # data,target的size为batchsize，即为64。
        # idx为第几个batch
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output, target)  # 对数似然损失
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            # 每隔100个batch记录一次损失，保存
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))
            train_loss_list.append(loss.item())  # 当前batch的loss
            train_count_list.append(idx * train_batch_size + (epoch - 1) * len(train_dataloader))
            # torch.save(mnist_net.state_dict(),"model/mnist_net.pkl")
            # torch.save(optimizer.state_dict(), 'results/mnist_optimizer.pkl')


def test():
    test_loss = 0
    correct = 0
    mnist_net.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in test_dataloader:
            output = mnist_net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 累加计算所有损失
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()  # 累加获取一个epoch中所有预测正确的样本
    test_loss /= len(test_dataloader.dataset)  # 每个样本的平均损失
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    mnist_net = MnistNet()
    optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)
    # criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    train_loss_list = []
    train_count_list = []
    test()
    # 模型训练5个epoch
    for i in range(5):
        train(i)
        print('-' * 35, 'test', '-' * 35)
        test()
    # 记录最终的函数表达式
    func = mnist_net.generate_equation()
    print(func)
    # 可视化结果(以下代码运行不了)
    plt.scatter(x, y, label='Original Data')
    plt.plot(x, predicted, label=func, color='red')
    plt.legend()
    plt.show()