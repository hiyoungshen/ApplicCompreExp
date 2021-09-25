import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: 数据采集
df = pd.read_csv('labelled_data.csv')
X = df.drop('label',axis=1)   
y = df['label']                         # normal & anomalous


# Step 2: 缺失值处理
#   Step 2.1: 判断各变量中是否存在缺失值
print(df.isnull().any(axis = 0))
#   Step 2.2: 各变量中缺失值的数量
print(df.isnull().sum(axis = 0))
#   Step 2.3: 各变量中缺失值的比例
print(df.isnull().sum(axis = 0)/df.shape[0])
#   Step 2.4: 需要添加inplace参数
X.fillna('-1',inplace=True)
print(X.head(100)) 

cat_cols = X.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print(f"col name: {col}, N Unique: {X[col].nunique()}")
for col in cat_cols:
    X[col]=X[col].astype("category")
    X[col]=X[col].cat.codes
print(X.head())


# Step 3: 使用PCA进行降维操作
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X_std = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_std)
X_pca = np.vstack((X_pca.T, y)).T
df_pca = pd.DataFrame(X_pca, columns=['1st_Component','2_nd_Component','label'])
print(df_pca.head())

plt.figure(figsize=(8,8))
sns.scatterplot(data=df_pca,hue='label',x='1st_Component',y='2_nd_Component')
plt.savefig('Scatter Program_PCA.jpg', dpi=300)
plt.show()