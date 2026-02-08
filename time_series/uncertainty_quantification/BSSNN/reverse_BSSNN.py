import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# 数据集准备
class BayesianDataset(Dataset):
    def __init__(self, df, input_cols, target_col):
        self.inputs = df[target_col].values  # y作为输入
        self.targets = df[input_cols].values  # X作为目标

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(0),  # y作为输入
            torch.tensor(self.targets[idx], dtype=torch.float32),  # X作为目标
        )

# 神经网络模型
class BayesianPredictXGivenY(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianPredictXGivenY, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # 预测多个X值

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 回归的线性输出
        return x

# 训练和评估函数
def train_model(model, loader, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for y_batch, x_batch in loader:
            optimizer.zero_grad()
            preds = model(y_batch)  # 前向传播
            loss = criterion(preds, x_batch)  # 计算损失
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for y_batch, x_batch in loader:
            pred = model(y_batch)
            preds.append(pred.numpy())
            targets.append(x_batch.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    mse = mean_squared_error(targets, preds) # 计算均方误差
    return preds, targets, mse


if __name__ == '__main__':
    # 1 读取数据
    data = pd.read_csv("binary_data.csv")
    input_cols = ["x1", "x2", "x3", "x4"]
    target_col = "y"

    # 2 将数据分为训练集和验证集
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = BayesianDataset(train_df, input_cols, target_col)
    val_dataset = BayesianDataset(val_df, input_cols, target_col)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3 初始化
    input_size = 1  # 单一输入：y
    hidden_size = 64
    output_size = len(input_cols)  # 预测所有X变量。共4个
    model = BayesianPredictXGivenY(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # 4 训练模型
    print("训练贝叶斯预测X|Y模型...")
    train_model(model, train_loader, optimizer, criterion, epochs=100)

    # 5 测试
    print("评估模型...")
    nn_preds, nn_targets, nn_mse = evaluate_model(model, val_loader)
    print(f"\n评估指标:\n  均方误差 (MSE): {nn_mse:.4f}")

    # 打印一些样本预测
    print("\n样本预测 (P(X|Y)):")
    for i in range(5):
         print(f"真实X: {nn_targets[i]}, 预测X: {nn_preds[i]}")