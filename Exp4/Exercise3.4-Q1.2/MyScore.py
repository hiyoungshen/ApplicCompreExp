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

# 使用rmse作为自定义得分函数，这也是比赛的判定标准
def myScore(y_true, y_pred):
    y_pred[y_pred==np.NaN]=0
    rmse = mean_squared_error(np.log1p(y_true), np.log1p(y_pred), squared=False)
    return rmse
