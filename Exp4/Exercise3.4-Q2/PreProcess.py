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


def preProcess(all_features: pd.DataFrame):
    # print(all_features.dtypes != 'object')
    # print(all_features.dtypes[all_features.dtypes != 'object'])
    # print(all_features.dtypes[all_features.dtypes != 'object'].index)
    # 数值型数据减均值除方差标准化
    numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )

    #  标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 标称数据直接onehot
    print(f"预处理之前数据: {all_features.shape}")
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(f"预处理之后数据: {all_features.shape}")
    return all_features
