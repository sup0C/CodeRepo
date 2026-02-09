#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
该版本是使用自定义的dataloader处理数据
'''

import torch
from torch import nn
import torch.nn.functional as F 
from torch.optim import Adam

import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import util
from datetime import date
import argparse
from progressbar import *

class Gaussian(nn.Module):

    def __init__(self, hidden_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size，默认为1。因为就一个均值
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size*args.n_layers, 1)
        self.sigma_layer = nn.Linear(hidden_size*args.n_layers, 1)
        # self.mu_layer = nn.Linear(hidden_size, 1)
        # self.sigma_layer = nn.Linear(hidden_size, 1)
        self.sigma=nn.Softplus()
        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, h):
        '''
        为每个batch学习一个mu和sigma
        Args:
            h:  h为神经网络隐藏层输出 (batch, hidden_size)。
        Returns:  (batch, output_size)
        '''
        _, hidden_size = h.size()
        # sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6 # 建模高斯分布的标准差. (batch, 1)
        sigma_t = self.sigma(self.sigma_layer(h))  # 建模高斯分布的标准差. (batch, 1)

        mu_t = self.mu_layer(h) # .squeeze(0) # 建模高斯分布的均值。 (batch, 1)
        return mu_t, sigma_t

class NegativeBinomial(nn.Module):

    def __init__(self, input_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, 1)
        self.sigma_layer = nn.Linear(input_size, 1)
    
    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t

def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample。基于生成的mu和sigma，进行高斯采样
    Args:
    mu (array like):(num_ts, 1)=(batch_size, 1)
    sigma (array like): standard deviation。(num_ts, 1)
    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    return：
    (num_ts, 1)
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    # 创建一个均值为mu，标准差为sigma的正态分布
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    # ypred = gaussian.sample(mu.size()) # 从正态分布中采样(num_ts, 1)个值
    ypred = gaussian.sample() # 从正态分布中采样(num_ts, 1)个值
    return ypred

def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred

class DeepAR(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()
         # input_size：特征数
        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size+input_size, hidden_size, \
                num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size)
        self.likelihood = likelihood
    
    def forward(self, X, y, Xf):
        '''
        Args:
            X和xf合起来就是全视野的协变量；y是历史视野的目标值。该模型的目的是基于这三个值去预测未来视野的目标值y_pred
        X (array like): shape (num_time_series, num_obs_to_train, num_features)
                             (batch_size, num_obs_to_train, num_features)
        y (array like): shape (num_time_series, num_obs_to_train)
        Xf (array like): shape (num_time_series, out_horizon, num_features)
                          (batch_size, out_horizon, num_features)
                        此处的num_ts也就是batch_size
        Return:
        y_pred： (num_ts, out_horizon).这些预测值都是高斯采样采出来的
        mu (array like): shape (batch_size, num_obs_to_train + out_horizon)
        sigma (array like): shape (batch_size, num_obs_to_train + out_horizon)
        '''
        if isinstance(X, type(np.empty(2))): # np.empty(2)是创建一个形状为 (2,) 的空数组
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, num_obs_to_train, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None # h和c是LSTM的隐藏层和memory层
        # 遍历每个时间点
        for s in range(num_obs_to_train + output_horizon):  # num_obs_to_train，为历史序列长度，output_horizon为预测长度
            if s < num_obs_to_train:  # Encoder
                # 1.1 ynext选取
                # 选取方法1 - 选取了当前时刻的y和协变量x对下一步y进行估计
                # NOTE：原文是使用当前时刻的协变量x和历史时刻y。
                ynext = y[:, s].view(-1, 1) # 取当前时刻的一个股票值. (num_ts,1)

                # 选取方法2 - 使用当前时刻的协变量x和历史时刻的y对当前y进行估计。（更贴合原文）
                # if s == 0: ynext = torch.zeros((num_ts, 1)).to(device)
                # else: ynext = y[:, s-1].view(-1, 1) # 取上一时刻的真实值. (num_ts,1)

                #  在前num_obs_to_train时间步先对可观测数据进行编码。即使用真实数据去生成下一个数据
                yembed = self.input_embed(ynext).view(num_ts, -1) # (num_ts,embedding_size)
                x = X[:, s, :].view(num_ts, -1) #  取当前时刻的特征。(num_ts,num_features)
            else: # decoder
                # 当上面对ynext的选取是使用选取方法2时，下面这一行需要开启，结果也会更好些。否则不用开启，因为训练思路不一样。
                # if s == num_obs_to_train: ynext = y[:, s - 1].view(-1, 1)  # (num_ts,1) # 预测的第一个时间点取上一时刻的真实值
                # 1.2 在num_obs_to_train时间步之后进行解码预测未来。即使用网络在当上个时间步生成的未来一个时间步的数据y进行embedding
                # 当循环的时间步s超出了num_obs_to_train，则使用上一个时间步中高斯采样的y作为输入
                yembed = self.input_embed(ynext).view(num_ts, -1)  # (num_ts,embedding_size)
                x = Xf[:, s-num_obs_to_train, :].view(num_ts, -1) # (num_ts,num_features)
            x = torch.cat([x, yembed], dim=1) # (num_ts, num_features + embedding)
            inp = x.unsqueeze(1) # (num_ts,1, num_features + embedding)
            if h is None and c is None:
                #  2 将x和y在cat之后送入LSTM
                out, (h, c) = self.encoder(inp) # h_size=c_size=(num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            # 方法1 - 计算mu和sigma的方法1，此处是取所有层的特征去计算mu和sigma
            hs = h.permute(1, 2, 0).contiguous().view(h.shape[1], -1) # (num_ts, num_layers*hidden_size)
            # 方法2 - 计算mu和sigma的方法2，此处是取最后一个seq在最后一个lstm单元的输出
            # hs = h[-1, :, :] # (num_ts, hidden_size)
            # hs = F.relu(hs) # (num_ts, hidden_size)
            # 3 将LSTM的输出送入MLP层为每个batch的数据学习其对应的mu和sigma
            # 为每个时间步上的数据都预测了自己的mu和sigma
            mu, sigma = self.likelihood_layer(hs) #  (batch_size, 1)
            mus.append(mu.view(-1, 1)) # 将在每个时间步上学到的mu和sigma保存起来
            sigmas.append(sigma.view(-1, 1))
            # 4 对未来值进行高斯采样，基于当前的mu和sigma从高斯分布中采样一个y值
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma) #(num_ts, 1)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # 5 在num_obs_to_train时间步之后对采样的未来值进行记录
            # if without true value, use prediction
            if s >= num_obs_to_train and s < output_horizon + num_obs_to_train: #在预测区间内
                ypred.append(ynext)
        # 输出
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1) # (num_ts, out_horizon)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)  #(num_ts, num_obs_to_train + out_horizon)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)  #(num_ts, num_obs_to_train + out_horizon)
        return ypred, mu, sigma
    
# def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
#     '''
#     数据处理方法1配套
#     Args:
#     将训练的样本分割为训练过程中可以观测，和不可以观测到的两对x、y。
#     X (array like): shape (num_samples, num_periods, num_features).
#                 (series_num,:all_seq_len*train_ratio,features_num)
#     y (array like): shape (num_samples, num_periods)
#                  (series_num,:all_seq_len*train_ratio)
#     num_obs_to_train (int):训练的历史窗口长度
#     seq_len (int): sequence/encoder/decoder length。即预测的长度
#     batch_size (int).批次。
#
#     Returns:
#         从所有训练数据中划分的input和target数据
#         其中，X_train_batch和y_train_batch是长度为num_obs_to_train的训练过程中可以观测到的x和y
#         Xf和yf是长度为seq_len（也就是output_horizon）的训练过程中可以观测到的x和不可以观测到的y
#             X_train_batch, y_train_batch, Xf, yf.
#             X_train_batch和xf合起来在这个时间尺度上的x，deepar方法假设其可以一直被观测到
#
#             X_train_batch.shape=(batch_size, num_obs_to_train, num_features)
#             y_train_batch.shape=(batch_size, num_obs_to_train)
#             Xf.shape= (batch_size, output_horizon, num_features)
#             yf.shape= (batch_size, output_horizon)
#
#     '''
#     num_ts, num_periods, _ = X.shape
#     if num_ts < batch_size:
#         batch_size = num_ts
#     # 从给定序列中选择一个随机数
#     t = random.choice(range(num_obs_to_train, num_periods-seq_len))
#     # Chooses batch_size unique random elements from a population sequence or set
#     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     # 从num_ts条数据中随机选择batch_size条   》》》 这里的batch_size采样什么意思？num_samples到底代表什么
#     batch = random.sample(range(num_ts), batch_size)
#     # 从划分的batch_size的训练数据中分别随机选取batch_size段长度为num_obs_to_train的数据序列作为一个batch的训练样本
#     X_train_batch = X[batch, t-num_obs_to_train:t, :] # (batch_size, num_obs_to_train, num_features)
#     y_train_batch = y[batch, t-num_obs_to_train:t]
#     # 就是训练数据的target
#     Xf = X[batch, t:t+seq_len,:] # (batch_size, output_horizon, num_features)
#     yf = y[batch, t:t+seq_len]  # (batch_size, output_horizon)
#     return X_train_batch, y_train_batch, Xf, yf

def train(X, Y,args):
    '''
    Args:
    - X (array like): (series_num,num_obs_to_train+seq_len,features_num)，
                    series_num也是num_samples
            此处共2个特征，其分别为：(今天的第几个小时,星期几)
    - Y (array like): shape (num_samples, num_obs_to_train+seq_len)
            股票数据。all_seq_len是数据的总长度
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon。
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''
    num_ts, num_periods, num_features = X.shape
    # 1 构建deepAR模型对象
    model = DeepAR(num_features, args.embedding_size, 
        args.hidden_size, args.n_layers, args.lr, args.likelihood).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(0)
    # 2 划分训练和测试数据。select sku with most top n quantities。
    # 数据处理方法1 - 原始代码方法
    '''
    x_train.shape=(series_num,:all_seq_len*train_ratio,features_num)
    x_test.shape=(series_num,all_seq_len*train_ratio:,features_num)
    '''
    # Xtr, ytr, Xte, yte = util.train_test_split(X, y, train_ratio=0.7)
    # Xte,yte=torch.from_numpy(Xte).float().to(device),torch.from_numpy(yte).float().to(device)
    # 数据处理方法2
    # SPLIT TRAIN TEST - 划分训练和测试样本
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, Y,
                                          test_size=0.3,
                                          random_state=0,
                                          shuffle=False)
    print("X_train:{},y_train:{}".format(Xtr.shape, ytr.shape)) # (num_samples*(1-test_size),num_obs_to_train+seq_len,features_num)
    print("X_test:{},y_test:{}".format(Xte.shape, yte.shape))  # (num_samples*(test_size),num_obs_to_train+seq_len,features_num)


    # 数据处理方法2配套- 构造Dtaloader
    Xtr = torch.as_tensor(torch.from_numpy(Xtr), dtype=torch.float32) # (num_samples*(1-test_size),num_obs_to_train+seq_len,features_num)
    Xte = torch.as_tensor(torch.from_numpy(Xte), dtype=torch.float32).to(device) # (num_samples*(test_size),num_obs_to_train+seq_len,features_num)
    ytr = torch.as_tensor(torch.from_numpy(ytr), dtype=torch.float32) # (num_samples*(1-test_size),num_obs_to_train+seq_len)
    yte = torch.as_tensor(torch.from_numpy(yte), dtype=torch.float32).to(device) # (num_samples*(test_size),num_obs_to_train+seq_len)

    # 3 是否对y，即股票数据进行标准化
    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
        # yscaler = StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    if yscaler is not None: # apply 标准化
        ytr = yscaler.fit_transform(ytr) #  (num_samples*(1-test_size),num_obs_to_train+seq_len)

    # 建立dataloader
    train_dataset = torch.utils.data.TensorDataset(Xtr, ytr)  # 训练集dataset
    train_Loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

    # 4 training
    losses = []
    cnt = 0 # 记录进行了多少个step
    seq_len = args.seq_len # 预测的长度，即output_horizon
    num_obs_to_train = args.num_obs_to_train # 训练的历史窗口长度
    progress = ProgressBar()
    for epoch in progress(range(args.num_epoches)):
        epoch_loss = 0
        # print("Epoch {} starts...".format(epoch))
        # for step in range(args.step_per_epoch):
        for input_data,target_data in train_Loader:
            # 4.1 数据处理方法1配套 - 从所有训练数据中划分网络模型的input和target
            # Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, seq_len, args.batch_size)
            # Xtrain_tensor = torch.from_numpy(Xtrain).float().to(device)
            # ytrain_tensor = torch.from_numpy(ytrain).float().to(device) # (batch_size, num_obs_to_train)
            # Xf = torch.from_numpy(Xf).float().to(device)
            # yf = torch.from_numpy(yf).float().to(device) #  (batch_size, output_horizon)
            # 4.1 数据处理方法2配套
            input_data = input_data.to(device)  # (batch_size, num_obs_to_train+seq_len, num_features)
            target_data = target_data.to(device)  # (batch_size, num_obs_to_train+seq_len)
            Xtrain_tensor = input_data[:, :num_obs_to_train, :].float()  # (batch_size, num_obs_to_train, num_features)
            ytrain_tensor = target_data[:, :num_obs_to_train].float()  # (batch_size, num_obs_to_train)
            Xf = input_data[:, -seq_len:, :].float()  # (batch_size, seq_len, num_features)
            yf = target_data[:, -seq_len:].float()  # (batch_size, seq_len)

            # 4.2 进行训练
            # ypred: (num_ts, out_horizon).这些预测值都是高斯采样采出来的
            # mu、sigma: (batch_size, num_obs_to_train + out_horizon)
            ypred, mu, sigma = model(Xtrain_tensor, ytrain_tensor, Xf)
            # ypred_rho = ypred
            # e = ypred_rho - yf
            # loss = torch.max(rho * e, (rho - 1) * e).mean()
            # 4.3 算损失
            ## gaussian loss
            ytrain_tensor = torch.cat([ytrain_tensor, yf], dim=1) # (batch_size, num_obs_to_train+output_horizon)
            if args.likelihood == "g":
                loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            elif args.likelihood == "nb":
                loss = util.negative_binomial_loss(ytrain_tensor, mu, sigma)
            # 4.4 反向传播更新参数
            losses.append(loss.item())
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    # test
    # select skus with most top K (num_samples*(test_size),num_obs_to_train+seq_len,features_num)
    # Xte = (batch_size, num_obs_to_train+seq_len, num_features)
    # yte =  (batch_size, num_obs_to_train+seq_len)
    X_test = Xte[:, :num_obs_to_train, :].float()  # (batch_size, num_obs_to_train, num_features)
    Xf_test = Xte[:, -seq_len:, :].float()  # (batch_size, seq_len, num_features)
    y_test = yte[:, :num_obs_to_train].float()  # (batch_size, num_obs_to_train)
    yf_test = yte[:, -seq_len:].float()  # (batch_size, seq_len)

    if yscaler is not None:
        y_test = yscaler.transform(y_test) # 用训练集的标准化参数对测试集进行标准化
    result = []
    n_samples = args.sample_size # 采样个数
    # 蒙特卡洛采样
    # 循环n_samples次，也即是使用相同的输入让模型预测N次，然后求在N次结果上的分位数等等。
    # p50相当于是n_samples次预测结果的平均。n_samples越大，预测结果越稳定。
    for _ in tqdm(range(n_samples)):
        y_pred, _, _ = model(X_test, y_test, Xf_test) # ypred:(num_samples, output_horizon)，这里num_samples就是batch_size
        # y_pred = y_pred
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)  # 用训练集的标准化参数对测试集进行反标准化
        # result.append(y_pred.reshape((-1, 1)))
        result.append(y_pred[:, :, np.newaxis].cpu().detach().numpy())  # (num_samples, output_horizon,1)
    result = np.concatenate(result, axis=2)  # (num_samples, output_horizon, n_samples)
    #  Compute the q-th quantile of the data along the specified axis.
    # 计算所有采样的序列（n_samples条）在同个seq位置上的分位数
    p90 = np.quantile(result, 0.9, axis=2) # (num_samples, output_horizon)
    p50 = np.quantile(result, 0.5, axis=2) # (num_samples, output_horizon)
    p10 = np.quantile(result, 0.1, axis=2) # (num_samples, output_horizon)
    yf_test=yf_test.cpu().detach().numpy()

    # 评价指标
    SMAPE = util.SMAPE(yf_test, p50) # 计算SMAPE指标。越接近0越好。
    print("P50 SMAPE: {}".format(SMAPE))
    mape = util.MAPE(yf_test, p50) # 计算MAPE指标。越接近0越好。
    print("P50 MAPE: {}".format(mape))
    PICP=util.calculate_PICP(yf_test,result)
    print("PICP:", PICP)
    PINAW=util.calculate_PINAW(yf_test,result)
    print("PINAW:",PINAW)
    cwc=util.calculate_CWC(PICP,PINAW)
    print("cwc:", cwc)

    # 可视化
    i = -1  # 选取其中一条序列进行可视化
    if args.show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot([k + num_obs_to_train for k in range(seq_len)], p50[i,:], "r-") # 绘制50%分位数曲线
        # 绘制10%-90%分位数阴影
        plt.fill_between(x=[k + num_obs_to_train  for k in range(seq_len)], \
            y1=p10[i,:], y2=p90[i,:], alpha=0.5)
        plt.title('Prediction uncertainty')
        yplot = yte[i, :].cpu().detach().numpy() #真实值 (1, seq_len+num_obs_to_train)
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["P50 forecast", "P10-P90 quantile","true"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(num_obs_to_train, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=100) # epoch次数
    # parser.add_argument("--step_per_epoch", "-spe", type=int, default=5) # 每个epoch中训练几个step。可以不用该参数
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=5) # 网络层数
    parser.add_argument("--hidden_size", "-hs", type=int, default=256)
    parser.add_argument("--embedding_size", "-es", type=int, default=10)  # 将上一时刻的真实值编码为embedding_size长度
    parser.add_argument("--likelihood", "-l", type=str, default="g") # likelihood to select, "g" or "nb"
    parser.add_argument("--seq_len", "-sl", type=int, default=60) # 预测的未来窗口长度
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=168) # 训练的历史窗口长度
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", default=True, action="store_true")
    parser.add_argument("--run_test", "-rt", default=True,action="store_true") # 是否需要测试
    parser.add_argument("--standard_scaler", "-ss", default=True,action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true") # los归一化
    parser.add_argument("--mean_scaler", "-ms", action="store_true") # 均值归一化
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=300) # 测试时的蒙特卡洛采样次数。在 Deep Factors/DeepAR 中训练后要采样的样本量，默认为 100

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.run_test:
        data_path = util.get_data_path() # 文件所在路径
        data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour.csv"), parse_dates=["date"]) # 尝试解析date列为时间格式
        data["year"] = data["date"].apply(lambda x: x.year) # 提取年份
        data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek) # # dayofweek获取此实例所表示的日期是星期几
        # 1 选取指定时间区间的数据，（1440,5）。特征分别为：date（年月日）、hour、MT 200（实际股票数据）、year、day of week。
        data = data.loc[(data["date"].dt.date >= date(2014, 1, 1)) & (data["date"].dt.date <= date(2014, 3, 1))]
        print(data.shape)
        # 可视化原始数据
        # plt.figure(figsize=(16, 4))
        # plt.plot(data['MT_200'])
        # plt.show()
        print(data.head())
        # 2 提取数据两个特征。用小时和星期几去预测未来seq_len长度的股票。
        features = ["hour", "day_of_week"]
        # hours = pd.get_dummies(data["hour"])
        # dows = pd.get_dummies(data["day_of_week"])
        hours = data["hour"]
        dows = data["day_of_week"]
        X = np.c_[np.asarray(hours), np.asarray(dows)] # shape=(seq_len,feature_num)=(1440,2)，2个特征分别为：(今天的第几个小时,星期几)
        num_features = X.shape[1] # 2。特征数
        num_periods = len(data) # 1440。数据总长度
        X = np.asarray(X).reshape((num_periods, num_features)) # (all_seq_len,features_num),series_num表示输入数据为几元数据
        # 这里如果要预测多元y变量，需要修改滑动采样代码
        y = np.asarray(data["MT_200"]).reshape((num_periods)) # (all_seq_len)
        # X = np.tile(X, (10, 1, 1))
        # y = np.tile(y, (10, 1))
        # 数据滑动采样
        width = args.num_obs_to_train + args.seq_len
        X_data = util.sliding_window(X, width, multi_vector=True)  # (len-width+1,width,features)
        Y_data = util.sliding_window(y, width, multi_vector=False).squeeze(-1)  # (len-width+1,width,1)
        print("x的维度为：", X_data.shape)
        print("y的维度为：", Y_data.shape)

        # 3 开始训练
        losses= train(X_data, Y_data, args)
        print("loss.shape",np.array(losses).shape,max(losses),min(losses))
        if args.show_plot:
            plt.figure()
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
