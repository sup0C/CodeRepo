#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
DeepAR Model (Pytorch Implementation)
Paper Link: https://arxiv.org/abs/1704.04110
Author: Jing Wang (jingw2@foxmail.com)
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

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t

class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)
    
    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t

def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log 
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample(mu.size())
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
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood
    
    def forward(self, X, y, Xf):
        '''
        Args:
        X (array like): shape (num_time_series, seq_len, input_size)
        y (array like): shape (num_time_series, seq_len)
        Xf (array like): shape (num_time_series, horizon, input_size)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, seq_len, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1)
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = X[:, s, :].view(num_ts, -1)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = Xf[:, s-seq_len, :].view(num_ts, -1)
            x = torch.cat([x, yembed], dim=1) # num_ts, num_features + embedding
            inp = x.unsqueeze(1)
            if h is None and c is None:
                out, (h, c) = self.encoder(inp) # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]
            hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(hs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # if without true value, use prediction
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma
    
def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf

def train(X, y,args):
    '''
    Args:
    - X (array like): (series_num,seq_len,features_num)，series_num也是num_samples
            此处共2个特征，其分别为：(今天的第几个小时,星期几)
    - y (array like): shape (num_samples, seq_len)
            股票数据
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''
    num_ts, num_periods, num_features = X.shape
    # 构建deepAR模型对象
    model = DeepAR(num_features, args.embedding_size, 
        args.hidden_size, args.n_layers, args.lr, args.likelihood)
    optimizer = Adam(model.parameters(), lr=args.lr)
    random.seed(2)
    # select sku with most top n quantities。划分数据。
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    # print("X_train:{},y_train:{}".format(Xtr.shape,ytr.shape))
    # print("X_test:{},y_test:{}".format(Xte.shape,yte.shape))
    losses = []
    cnt = 0
    # 是否对y，即股票数据进行标准化
    yscaler = None
    if args.standard_scaler:
        yscaler = util.StandardScaler()
    elif args.log_scaler:
        yscaler = util.LogScaler()
    elif args.mean_scaler:
        yscaler = util.MeanScaler()
    if yscaler is not None: # apply 标准化
        ytr = yscaler.fit_transform(ytr)

    # training
    seq_len = args.seq_len # 也即是
    num_obs_to_train = args.num_obs_to_train # 训练的历史窗口长度
    progress = ProgressBar()
    for epoch in progress(range(args.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(args.step_per_epoch):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, seq_len, args.batch_size)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()  
            yf = torch.from_numpy(yf).float()
            ypred, mu, sigma = model(Xtrain_tensor, ytrain_tensor, Xf)
            # ypred_rho = ypred
            # e = ypred_rho - yf
            # loss = torch.max(rho * e, (rho - 1) * e).mean()
            ## gaussian loss
            ytrain_tensor = torch.cat([ytrain_tensor, yf], dim=1)
            if args.likelihood == "g":
                loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            elif args.likelihood == "nb":
                loss = util.negative_binomial_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
    
    # test 
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len-num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len-num_obs_to_train:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    if yscaler is not None:
        y_test = yscaler.transform(y_test)
    result = []
    n_samples = args.sample_size
    for _ in tqdm(range(n_samples)):
        y_pred, _, _ = model(X_test, y_test, Xf_test)
        y_pred = y_pred.data.numpy()
        if yscaler is not None:
            y_pred = yscaler.inverse_transform(y_pred)
        result.append(y_pred.reshape((-1, 1)))
    
    result = np.concatenate(result, axis=1)
    p50 = np.quantile(result, 0.5, axis=1)
    p90 = np.quantile(result, 0.9, axis=1)
    p10 = np.quantile(result, 0.1, axis=1)
    
    mape = util.MAPE(yf_test, p50)
    print("P50 MAPE: {}".format(mape))
    mape_list.append(mape)

    if args.show_plot:
        plt.figure(1, figsize=(20, 5))
        plt.plot([k + seq_len + num_obs_to_train - seq_len \
            for k in range(seq_len)], p50, "r-")
        plt.fill_between(x=[k + seq_len + num_obs_to_train - seq_len for k in range(seq_len)], \
            y1=p10, y2=p90, alpha=0.5)
        plt.title('Prediction uncertainty')
        yplot = yte[-1, -seq_len-num_obs_to_train:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(seq_len + num_obs_to_train - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses, mape_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=1000)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=3) # 网络层数
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--embedding_size", "-es", type=int, default=64)
    parser.add_argument("--likelihood", "-l", type=str, default="g") # likelihood to select, "g" or "nb"
    parser.add_argument("--seq_len", "-sl", type=int, default=7) # 预测的未来窗口长度
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=1) # 训练的历史窗口长度
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--run_test", "-rt", default=True,action="store_true") # 是否需要测试
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true") # los归一化
    parser.add_argument("--mean_scaler", "-ms", action="store_true") # 均值归一化
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--sample_size", type=int, default=100) # 在 Deep Factors/DeepAR 中训练后要采样的样本量，默认为 100

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.run_test:
        data_path = util.get_data_path() # 文件所在路径
        data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour.csv"), parse_dates=["date"]) # 尝试解析date列为时间格式
        data["year"] = data["date"].apply(lambda x: x.year) # 提取年份
        data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek) # # dayofweek获取此实例所表示的日期是星期几
        # 1 选取指定时间区间的数据，（1440,5）。特征分别为：date（年月日）、hour、MT 200（实际数据）、year、day of week。
        data = data.loc[(data["date"].dt.date >= date(2014, 1, 1)) & (data["date"].dt.date <= date(2014, 3, 1))]
        print(data.shape)
        plt.figure(figsize=(16, 4))
        plt.plot(data['MT_200'])
        print(data.head())
        # 2 提取数据两个特征。用小时和星期几去预测未来seq_len长度的电量。
        features = ["hour", "day_of_week"]
        # hours = pd.get_dummies(data["hour"])
        # dows = pd.get_dummies(data["day_of_week"])
        hours = data["hour"]
        dows = data["day_of_week"]
        X = np.c_[np.asarray(hours), np.asarray(dows)] # shape=(seq_len,feature_num)=(1440,2)，2个特征分别为：(今天的第几个小时,星期几)
        num_features = X.shape[1] # 2。特征数
        num_periods = len(data) # 1440。总样本数
        X = np.asarray(X).reshape((-1, num_periods, num_features)) # (series_num,seq_len,features_num),series_num表示输入数据为几元数据
        y = np.asarray(data["MT_200"]).reshape((-1, num_periods)) # (series_num,seq_len)
        # X = np.tile(X, (10, 1, 1))
        # y = np.tile(y, (10, 1))
        # 3 开始训练
        losses, mape_list = train(X, y, args)
        if args.show_plot:
            plt.plot(range(len(losses)), losses, "k-")
            plt.xlabel("Period")
            plt.ylabel("Loss")
            plt.show()
