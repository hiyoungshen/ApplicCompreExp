# 题目2
身高分析预测

1. 由于身高信息难以采集，采用类似问题的房价数据集进行处理。
## 数据采集
1. kaggle比赛房价数据集采集。
## 数据存储
1. 存储为CSV数据集。

## 数据清洗
1. preProcess.py
2. 数值型数据减均值除方差标准化
3. 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
4. 然后进行one hot编码。
## 数据标记抽取
1. 数据集划分为训练集和验证集。
