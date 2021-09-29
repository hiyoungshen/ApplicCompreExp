import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from datetime import datetime
from colorama import Fore, Back

import warnings

warnings.filterwarnings("ignore")

import argparse

from PreProcess import preProcess
from MyDataset import MyDataset
from MyNet import MyNet
from MyScore import myScore
from train import train

parser = argparse.ArgumentParser(
        description='Regression of house price prediction in kaggle. ')
parser.add_argument("--trainfile", type=str, default = "./data/train.csv", help="train file in format .csv")
parser.add_argument("--testfile", type=str, default = "./data/test.csv", help="test file in format .csv")
parser.add_argument("--save", action = "store_true", default = False, help="test file in format .csv")
args = parser.parse_args()

trainfile=args.trainfile
testfile=args.testfile
save=args.save

if __name__ == "__main__":
    train_data_origin = pd.read_csv(trainfile)
    test_data_origin = pd.read_csv(testfile)
    # 第一列为id，最后一列为标签，把这两列取出后合并训练集与测试集
    # print(train_data.iloc[0, 1])
    # print(all_features)
    all_features = pd.concat((train_data_origin.iloc[:, 1:-1], test_data_origin.iloc[:, 1:]))
    all_features = preProcess(all_features)

    train_num = train_data_origin.shape[0]  # 训练集总样本
    train_data = all_features.iloc[:train_num, :]  # 训练集样本
    test_data = all_features.iloc[train_num:, :]  # 测试集样本
    # 8, 2 分训练集，验证集
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_data, train_data_origin.iloc[:, -1], test_size=0.2, random_state=42
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
    print(f"训练集: {train_features.shape}")
    print(f"验证集: {val_features.shape}")
    print(f"测试集: {test_features.shape}")

    train_dataset = MyDataset(train_features, train_labels)
    val_dataset = MyDataset(val_features, val_labels)

    train_iter = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4
    )
    val_iter = DataLoader(
        dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MyNet()
    net = net.to(device)

    criterion = torch.nn.MSELoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.005, weight_decay=1)

    epochs = 100
    writer = SummaryWriter()
    print(f"{datetime.now()} Begin training...")
    for epoch in tqdm(range(epochs)):
        train_mean_loss, train_score = train(
            net=net,
            data_iter=train_iter,
            phase="train",
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_mean_loss, val_score = train(
            net=net,
            data_iter=train_iter,
            phase="val",
            criterion=criterion,
            optimizer=None,
            device=device,
        )
        writer.add_scalar("Loss/train", train_mean_loss, epoch)
        writer.add_scalar("Loss/val", val_mean_loss, epoch)
        writer.add_scalar("Score/train", train_score, epoch)
        writer.add_scalar("Score/val", val_score, epoch)
        if epoch % 10 == 0:
            print(Fore.CYAN + Back.BLACK, end="")
            tqdm.write(
                f"Epoch: {epoch} Train loss: {train_mean_loss} Val loss: {val_mean_loss}",
                end=" ",
            )
            tqdm.write(f"Train score: {train_score} Val score: {val_score}")
    writer.close()
    print(f"{datetime.now()} End training...")
    if save:
        torch.save(net, "net.pth")
    else:
        torch.load(net, "net.pth")

    #  提交结果
    submission = pd.read_csv('./data/sample_submission.csv')
    predict = net(test_features)
    predict = predict.detach().squeeze().numpy()
    submission['SalePrice'] = predict
    submission.to_csv('torch_submission.csv', index=False)

