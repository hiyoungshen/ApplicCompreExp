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


def pre_process(all_features: pd.DataFrame):
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
    print(f"预处理之前数据形状: {all_features.shape}")
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(f"预处理之后数据形状: {all_features.shape}")
    return all_features


# 生成数据集类
class myDataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.data[idx, :], self.label[idx]


# 使用rmse作为自定义得分函数，这也是比赛的判定标准
def custom_score(y_true, y_pred):
    rmse = mean_squared_error(np.log1p(y_true), np.log1p(y_pred), squared=False)
    return rmse

# 网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(331, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        # 初始化权重
        def _weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.apply(_weight_init) # 初始化参数
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 这是训练函数，分为train和val
# train时前向传播后向更新参数
# val时只计算损失函数
def train(net, data_iter, phase, criterion, optimizer=None):
    y_true = []
    y_pred = []
    mean_loss = []
    is_grad = True if phase == 'train' else False
    with torch.set_grad_enabled(is_grad):
        net.train()
        for step, (X, y) in enumerate(data_iter):
            X = X.to(device)
            y = y.to(device)
            out = net(X)
            loss = criterion(out, y) # 计算损失
            mean_loss.append(loss.item())
            
            if phase == 'train':
                optimizer.zero_grad() # optimizer 0
                loss.backward() # back propragation
                optimizer.step() # update the paramters

            # 将每一个step的结果加入列表，最后统一生产这一个epoch的指标  
            # 添加预测值和真实类标签
            y_pred.extend(out.detach().cpu().squeeze().numpy().tolist())
            y_true.extend(y.detach().cpu().squeeze().numpy().tolist())

    # 全量样本的rmse和平均loss
    rmse = custom_score(y_true, y_pred)
    mean_loss = np.mean(mean_loss)
    # 保留4位小数
    rmse = np.round(rmse, 4)
    mean_loss = np.round(mean_loss, 4)
    return mean_loss, rmse



if __name__ == "__main__":
    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")
    # 第一列为id，最后一列为标签，把这两列取出后合并训练集与测试集
    # print(train_df.iloc[0, 1])
    # print(all_features)
    all_features = pd.concat((train_df.iloc[:, 1:-1], test_df.iloc[:, 1:]))
    all_features = pre_process(all_features)

    train_num = train_df.shape[0]  # 训练集总样本
    train_data = all_features.iloc[:train_num, :]  # 训练集样本
    test_data = all_features.iloc[train_num:, :]  # 测试集样本
    # 8, 2 分训练集，验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_data, train_df.iloc[:, -1], test_size=0.2, random_state=42
    )

    # 转化为tensor
    train_features = torch.tensor(train_features.values, dtype=torch.float)
    val_features = torch.tensor(val_features.values, dtype=torch.float)
    test_features = torch.tensor(test_data.values, dtype=torch.float)

    # 类标签需要加一维, 为了作为损失函数的输入
    # eg:[100,]--->[100, 1]
    train_labels = torch.tensor(train_labels.values, dtype=torch.float)
    train_labels = train_labels.unsqueeze(1)
    val_labels = torch.tensor(val_labels.values, dtype=torch.float)
    val_labels = val_labels.unsqueeze(1)
    print(f"训练集数据: {train_features.shape}")
    print(f"验证集数据: {val_features.shape}")
    print(f"测试集数据: {test_features.shape}")


    train_dataset = myDataset(train_features, train_labels)
    val_dataset = myDataset(val_features, val_labels)

    # 变为迭代器
    train_iter = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_iter = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net = net.to(device) # 将网络和损失函数转化为GPU或CPU

    criterion = torch.nn.MSELoss() # 损失函数为MSE
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, weight_decay=0)

    epochs = 100
    print(f'{datetime.now()} Begin training...')
    for epoch in tqdm(range(epochs)):
        train_mean_loss, train_score = train(net=net, 
                                            data_iter=train_iter, 
                                            phase='train', 
                                            criterion=criterion, 
                                            optimizer=optimizer)
        
        val_mean_loss, val_score = train(net=net, 
                                        data_iter=train_iter, 
                                        phase='val', 
                                        criterion=criterion, 
                                        optimizer=None)
        if epoch%10  == 0:
            print(Fore.CYAN + Back.BLACK, end='')
            tqdm.write(f'Epoch: {epoch} Train loss: {train_mean_loss} Val loss: {val_mean_loss}', end=' ')
            tqdm.write(f'Train score: {train_score} Val score: {val_score}')

    print(f'{datetime.now()} 训练结束...')