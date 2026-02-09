#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
Utility functions
'''
import torch 
import numpy as np
import os
import random

def get_data_path():
    folder = os.path.dirname(__file__)
    return os.path.join(folder, "data")

def RSE(ypred, ytrue):
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
            np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-6
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
        / mean_y))

def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-6
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
        / ytrue))


def calculate_PICP(y_true, y_pred, alpha=None):
    """
    计算区间预测的评价指标
    参数:
        y_true: 真实值，shape为(n_samples,output_horizon)
        y_pred: 区间预测值，shape为(num_samples, output_horizon, n_samples)
                num_samples是预测几元序列，n_samples是测试时蒙特卡洛的采样数
        alpha: 置信水平，取值范围为(0, 1)
    返回:
        PICP: 区间预测置信度，float
    """
    PICP = 0
    test_samples, seq_len = y_true.shape
    p90 = np.quantile(y_pred, 0.9, axis=2) # (num_samples, output_horizon)
    p10 = np.quantile(y_pred, 0.1, axis=2) # (num_samples, output_horizon)

    for i in range(test_samples):
        count = 0
        for j in range(seq_len):
            if y_true[i, j] > p10[i, j] and y_true[i, j] < p90[i, j]:
                count += 1
        picp = count / seq_len
        PICP += picp
    PICP = PICP / test_samples
    return PICP

def calculate_PINAW(y_true, y_pred, alpha=None):
    """
    计算区间预测的评价指标
    参数:
        y_true: 真实值，shape为(n_samples,output_horizon)
        y_pred: 区间预测值，shape为(num_samples, output_horizon, n_samples)
                num_samples是预测几元序列，n_samples是测试时蒙特卡洛的采样数
        alpha: 置信水平，取值范围为(0, 1)
    返回:
        PINAW(PI normalized averaged width) 用于定义区间的狭窄程度，在保证准确性的前提下越小越好
    """
    PINAW = 0
    test_samples, seq_len = y_true.shape
    p90 = np.quantile(y_pred, 0.9, axis=2) # (num_samples, output_horizon)
    p10 = np.quantile(y_pred, 0.1, axis=2) # (num_samples, output_horizon)

    for i in range(test_samples):
        width = 0
        true_max = np.max(y_true[i, :])
        true_min = np.min(y_true[i, :])
        for j in range(seq_len):
            width += (p90[i, j] - p10[i, j])
        width /= seq_len
        pinaw = (width / (true_max - true_min))
        PINAW += pinaw
    PINAW = PINAW / test_samples

    return PINAW


# 滑动窗口
def sliding_window(input_data, width, multi_vector=True):
    '''
    对数据进行滑动采样
    Args:
        input_data:所有数据。 (all_length,features)或者 (length)
                    num_samples代表几元序列
                    width = num_obs_to_train + output_horizon
        width:窗口宽度
        multi_vector:是否是多元序列

    Returns:
        data：(all_length-width+1,width,features)，最后的all_length-width+1就是num_samples,
    '''
    if multi_vector:  # 二维 (length,features)
        length, features = input_data.shape
    else:  # 一维 (length)
        input_data = input_data[:, np.newaxis]  # (length,1)
        length, features = input_data.shape

    x = input_data[ 0:width, :]  # (width,features)
    x = x[np.newaxis,  :, :]  # (1,num_samples,width,features)
    for i in range(length - width):
        i += 1
        tmp = input_data[ i:i + width, :]  # (width,features)
        tmp = tmp[np.newaxis,  :, :]  # (1,width,features)
        x = np.concatenate([x, tmp], 0)  # (i+1,width,features)
    return x



def calculate_CWC(PICP, PINAW, eta=90,u=0.9):
    """
    计算区间预测的评价指标
    参数:
        PICP
        PINAW
        eta: 惩罚系数，取值在50-100
    返回:
        CWC:越小越好
    """
    error = np.exp(-eta * (PICP - u))
    if PICP >= u:
        gamma = 0
    else:
        gamma = 1
    CWC = PINAW * (1 + gamma * error)

    return CWC


def crps(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)
    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)
    y_pred = np.sort(y_pred, axis=0)  # (3,60)
    diff = y_pred[1:] - y_pred[:-1]  # 一阶差分
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples ** 2
    return np.average(per_obs_crps, weights=sample_weight)


def calculate_crps(y_true, y_pred, alpha=None):
    """
    计算区间预测的评价指标
    参数:
        y_true: 真实值，shape为(n_samples,output_horizon)
        y_pred: 区间预测值，shape为(num_samples, output_horizon, n_samples)
                num_samples是预测几元序列，n_samples是测试时蒙特卡洛的采样数
        alpha: 置信水平，取值范围为(0, 1)
    返回:
        PINAW(PI normalized averaged width) 用于定义区间的狭窄程度，在保证准确性的前提下越小越好
    """
    test_samples, seq_len = y_true.shape
    p90 = np.quantile(y_pred, 0.9, axis=2) # (num_samples, output_horizon)
    p50 = np.quantile(y_pred, 0.5, axis=2) # (num_samples, output_horizon)
    p10 = np.quantile(y_pred, 0.1, axis=2) # (num_samples, output_horizon)

    CRPS = 0
    for i in range(test_samples):
        this_pred = np.concatenate([p90[i, None, :], p50[i, None, :], p10[i, None, :]],
                                   axis=0)  # (3,output_horizon)
        this_true = y_true[i, :]  # (1,n_samples)
        c = crps(this_true, this_pred)
        CRPS += c
    CRPS = CRPS / test_samples

    return CRPS

# def train_test_split(X, y, train_ratio=0.7):
#     '''
#     数据处理方法1配套
#     :param X: X.shape=(series_num,all_seq_len,features_num)，series_num也是num_samples
#     :param y:Y.shape=(num_samples, all_seq_len)
#     :param train_ratio:
#     :return:划分后的训练数据
#             X_train=(series_num,:all_seq_len*train_ratio,features_num)
#             X_test=(series_num,all_seq_len*train_ratio:,features_num)
#
#     '''
#     num_ts, num_periods, num_features = X.shape
#     train_periods = int(num_periods * train_ratio)
#     random.seed(2)
#     Xtr = X[:, :train_periods, :]
#     ytr = y[:, :train_periods]
#     Xte = X[:, train_periods:, :]
#     yte = y[:, train_periods:]
#     return Xtr, ytr, Xte, yte

class StandardScaler:
    # Z-score标准化。Z-score标准化是一种将数据转换为均值为0、标准差为1的标准正态分布的方法。
    def fit_transform(self, y):
        '''
        Args:
            y: (num_samples*(1-test_size),num_obs_to_train+seq_len)
        Returns:
        '''
        self.mean = torch.mean(y) # 2900
        self.std = torch.std(y) + 1e-6  # 850
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std

class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max
    
    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MeanScaler:
    
    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean
    
    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean

class LogScaler:
    # log标准化
    def fit_transform(self, y):
        return np.log1p(y) # log1p = log(x+1)      即ln(x+1)
    
    def inverse_transform(self, y):
        return np.expm1(y) # expm1 = exp(x)-1

    def transform(self, y):
        return np.log1p(y)


def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    计算高斯分布的负对数似然，也就是优化网络时的损失。
    Args:
    z (tensor):在所有时间步上的真实数据。
        true observations, shape (num_ts, num_periods)
        (batch_size, num_obs_to_train+output_horizon)
    mu (tensor):
        mean, shape (num_ts, num_periods)
        (batch_size, num_obs_to_train+output_horizon)
    sigma (tensor):
        standard deviation, shape (num_ts, num_periods)
        (batch_size, num_obs_to_train+output_horizon)

    likelihood: 
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2)) # 高斯概率密度函数

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2) # 高斯分布的对数似然
    '''
    # 高斯分布的负对数似然。
    # 手动实现
    # negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6 # 原始loss
    # negative_likelihood = torch.log(sigma ) + (z - mu) ** 2 / (2 * sigma ** 2) + 6 # 修改后的
    # return negative_likelihood.mean()
    # API实现
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(z)
    loss=-torch.mean(likelihood)
    return loss

def negative_binomial_loss(ytrue, mu, alpha):
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
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()

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
