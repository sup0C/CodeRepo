import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import *
from keras.models import Model
import random


class Network(object):

    def __init__(self, sizes):  # 网络初始化，size为网络从第一层输出层神经元数目如[748,30,10]
        self.num_layers = len(sizes)  # 层次数，如上例返回3
        self.sizes = sizes  # 各层神经元数量，和size相等
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # 创建从第二层开始的各层的偏置，符合高斯分布
        # biases中包含了第2，3，。。等层的偏置值
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # 创建第1层到第2层，第2层到第3层，。。权值，设size=[784,30,10],则2个矩阵大小分别为30*784， 10*30
        # 权值按照高斯概率分布

    def feedforward(self, a):  # 前馈计算，给定输入a,得到神经网络的输出
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):  # 例如输入是数字7的748*1的列向量，两层，第一层30*748，第二层10*30
            a = sigmoid(np.dot(w, a) + b)  # 则循环第一次输出计算大小为30*784的权值矩阵和748*1的乘积+偏置，得到30*1作为下层输入，
        return a  # 返回前馈后的激活输出

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)  # 训练集的大小
        for j in range(epochs):
            random.shuffle(training_data)  # 对训练集列表中的数据随机排序，打乱,然后从里面以min_batch_size为一批选取训练数据集合
            # 例如总的数据为40000，每个批次400，则mini_batches中存放了100个大小为400的数据集
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # 对每个数据集，调用更新函数，计算一次正向传播和一次反向传播后的数据变动
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))  # 输出字符串“第j代完成”，其中j分别取0，1，2.。。

    def update_mini_batch(self, mini_batch, eta):
        # 利用梯度下降更新权值和偏置，其中代价函数对权值和偏导的计算使用反向传播得到
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 初始化代价函数对偏置b的偏导，全部取0
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 初始化代价函数对权值w的偏导，全部取0
        for x, y in mini_batch:  # 对于mini_batch中的每个x,y，其中x是28*28图片数据，y是10*1类别向量，计算：
            # 即将每个batch中每个数据产生的w_i、b_i的梯度先各自相加起来，然后进行梯度更新(w=w-lr*sum(delta_w))
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # 利用后向传播函数计算偏置和权值的梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):  # 后向传播函数计算梯度
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 清零
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 清零
        # feedforward 前向传播计算输出
        activation = x
        activations = [x]  # list to store all the activations, layer by layer #每一层的带权输出，即激活值
        zs = []  # list to store all the z vectors, layer by layer #每一层的带权输入存放在列表zs中
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)  # 将当前层的激活值，也就是输出值放入到列表中
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # 求解最后输出层误差，输出层的就是
        nabla_b[-1] = delta  # 最后一层的偏置
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 最后一层的权值
        for l in range(2, self.num_layers):  # 后向传播，计算每一层偏置和权值的调整也就是梯度，从-2个算起
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)  # argmax返回最大输出的索引值
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)  # 二次代价函数对激活输出的导数等于输出的激活值-y


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = [x_train[i].reshape(784, 1) / 255 for i in range(x_train.shape[0])]  # 对6000个图像转换成784*1列向量并归一化，放在列表中
y_train = to_categorical(y_train, 10)  # 将类别标签转换为只有0，1的表示。如4转换成【0001000000】
y_train = [y_train[i].reshape(10, 1) for i in range(y_train.shape[0])]  # y_train中的类别标签也转换称为10*1的列向量
training_data = list(zip(x_train, y_train))  # 每个图像和对应的类型对应起来形成训练数据列表
x_test = [x_test[i].reshape(784, 1) for i in range(x_test.shape[0])]  # 测试数据集中的图像转换为784*列向量，但是标签不需要转换
test_data = list(zip(x_test, y_test))
net = Network([784, 30, 10])
net.SGD(training_data, 30, 200, 1.0, test_data=test_data)