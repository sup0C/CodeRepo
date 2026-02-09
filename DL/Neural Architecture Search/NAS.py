import torch.nn as nn
import torch.optim as optim

# 1 搜索空间定义
search_space = {
     # 架构结构
    'num_hidden_layers': [1, 2, 3, 4, 5],
    'hidden_layer_size': [32, 64, 128, 256, 512],
    'activation_function': ['ReLU', 'LeakyReLU', 'Tanh'],

     # 超参数
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'optimizer': ['Adam', 'SGD', 'RMSprop'],
    'dropout_rate': [0.0, 0.2, 0.4, 0.6]}

activation_map = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Tanh': nn.Tanh}

optimizer_map = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop}

# 2 架构评估
import torch

def evaluate_architecture(architecture, X_train, y_train, X_val, y_val, num_epochs=50):
    # 初始化模型、优化器和损失函数
    model = build_model(architecture)
    criterion = nn.MSELoss()

    optimizer_class = optimizer_map[architecture['optimizer']]
    optimizer = optimizer_class(model.parameters(), lr=architecture['learning_rate'])

    # 训练模型
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 使用验证数据集验证模型
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    return val_loss.item()


import torch.nn as nn

class ArchitectureController(nn.Module):
    def __init__(self, search_space):
        super(ArchitectureController, self).__init__()
        self.search_space = search_space
        self.keys = list(search_space.keys())
        self.vocab_size = [len(search_space[key]) for key in self.keys]
        self.num_actions = len(self.keys)
        self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=1)
        self.policy_heads = nn.ModuleList([nn.Linear(64, vs) for vs in self.vocab_size])

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        logits = [head(output.squeeze(0)) for head in self.policy_heads]
        return logits, hidden