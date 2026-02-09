import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 数据集准备
class BayesianDataset(Dataset):
    def __init__(self, df, input_cols, target_col):
        self.inputs = df[input_cols].values
        self.targets = df[target_col].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),)

# 神经网络模型
class BayesianInspiredNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        '''

        '''
        super(BayesianInspiredNNModel, self).__init__()
        # 计算联合概率P(y, x1, x2, ...)
        self.fc1_joint = nn.Linear(input_size, hidden_size)
        self.relu_joint = nn.ReLU()
        self.fc2_joint = nn.Linear(hidden_size, 1)

        # 计算边际概率 P(x1, x2, ...)
        self.fc1_marginal = nn.Linear(input_size, hidden_size)
        self.relu_marginal = nn.ReLU()
        self.fc2_marginal = nn.Linear(hidden_size, 1)

        # 最终sigmoid激活函数用于概率  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 联合概率计算  
        joint = self.relu_joint(self.fc1_joint(x))
        joint = self.fc2_joint(joint)  # (batch,1)。输出: P(y, x1, x2, ...)的logit

        # 边际概率计算  
        marginal = self.relu_marginal(self.fc1_marginal(x))
        marginal = self.fc2_marginal(marginal)  # (batch,1)。输出: P(x1, x2, ...)的logit

        # 贝叶斯除法: P(y|x1, x2, ...) = P(y, x1, x2, ...) / P(x1, x2, ...)
        # 对数空间除法。这里是隐式的对数空间除法，因为输入是概率，而不是概率的对数
        conditional = joint - marginal # (batch,1)。
        return self.sigmoid(conditional)  # 概率分数  


# 训练和评估函数
def train_model(model, loader, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(x_batch).squeeze() # x_batch.shape=(batch,feature_num)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")


def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            pred = model(x_batch).squeeze()
            preds.append(pred.numpy())
            targets.append(y_batch.numpy())
    preds = np.hstack(preds)
    targets = np.hstack(targets)
    return preds, targets


def calculate_auc_ks(preds, targets):
    '''
    preds:网络预测。(sample_num,)
    targets:实际独热向量。(sample_num,)
    '''
    auc = roc_auc_score(targets, preds)
    fpr, tpr, _ = roc_curve(targets, preds)
    ks = max(tpr - fpr)
    return auc, ks


def logistic_regression_benchmark(train_df, val_df, input_cols, target_col):
    X_train = train_df[input_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[input_cols].values
    y_val = val_df[target_col].values

    # 训练逻辑回归
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # 预测概率
    preds = log_reg.predict_proba(X_val)[:, 1]  # 获取正类的概率

    # 计算AUC和KS
    auc, ks = calculate_auc_ks(preds, y_val)
    return auc, ks


if __name__ == '__main__':
    # 1 加载数据
    data = pd.read_csv("binary_data.csv")
    input_cols = ["x1", "x2", "x3"]
    target_col = "y"

    # 2 将数据分为训练集和验证集
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = BayesianDataset(train_df, input_cols, target_col)
    val_dataset = BayesianDataset(val_df, input_cols, target_col)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3 初始化和训练模型
    input_size = len(input_cols) # 3
    hidden_size = 64

    model = BayesianInspiredNNModel(input_size, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    # 4 训练
    print("训练贝叶斯神经网络...")
    train_model(model, train_loader, optimizer, criterion, epochs=100)

    # 5 测试
    print("评估模型...")
    nn_preds, nn_targets = evaluate_model(model, val_loader)
    nn_auc, nn_ks = calculate_auc_ks(nn_preds, nn_targets)
    print(f"\n该方法性能指标:\n  AUC: {nn_auc:.4f}\n  KS: {nn_ks:.4f}")

    # 6 评估逻辑回归
    print("评估逻辑回归...")
    lr_auc, lr_ks = logistic_regression_benchmark(train_df, val_df, input_cols, target_col)
    print(f"\n逻辑回归性能:\n  AUC: {lr_auc:.4f}\n  KS: {lr_ks:.4f}")