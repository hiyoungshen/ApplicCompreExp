import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from datetime import datetime
from colorama import Fore, Back

import warnings

warnings.filterwarnings("ignore")

from MyScore import myScore

# 这是训练函数，分为train和val
# train时前向传播后向更新参数
# val时只计算损失函数
def train(net, data_iter, phase, criterion, optimizer=None, device=None):
    y_true = []
    y_pred = []
    mean_loss = []
    is_grad = True if phase == "train" else False
    with torch.set_grad_enabled(is_grad):
        net.train()
        for step, (X, y) in enumerate(data_iter):
            X = X.to(device)
            y = y.to(device)
            out = net(X)
            loss = criterion(out, y)  # 计算损失
            mean_loss.append(loss.item())

            if phase == "train":
                optimizer.zero_grad()  # optimizer 0
                loss.backward()  # back propragation
                optimizer.step()  # update the paramters

            # 将每一个step的结果加入列表，最后统一生产这一个epoch的指标
            # 添加预测值和真实类标签
            y_pred.extend(out.detach().cpu().squeeze().numpy().tolist())
            y_true.extend(y.detach().cpu().squeeze().numpy().tolist())

    # 全量样本的rmse和平均loss
    rmse = myScore(y_true, y_pred)
    mean_loss = np.mean(mean_loss)
    # 保留4位小数
    rmse = np.round(rmse, 4)
    mean_loss = np.round(mean_loss, 4)
    return mean_loss, rmse
