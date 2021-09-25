import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('labelled_data.csv')
X = df.drop('label',axis=1)   
y = df['label'] 
# ---------------------------  缺失值处理 ---------------------------
print(df.isnull().any(axis = 0)) 
print(df.isnull().sum(axis = 0)) 
print(df.isnull().sum(axis = 0)/df.shape[0]) 
X.fillna('-1',inplace=True) 
print(X.head(100)) 
# ------------------------------------------------------------------
cat_cols = X.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print(f"col name: {col}, N Unique: {X[col].nunique()}")
for col in cat_cols:
    X[col]=X[col].astype("category")
    X[col]=X[col].cat.codes
print(X.head())
# ------------------------------------------------------------------
# 使用PCA进行降维操作
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
X_std = StandardScaler().fit_transform(X)
X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(X_std)
X_tsne = np.vstack((X_tsne.T, y)).T

df_tsne = pd.DataFrame(X_tsne, columns=['Dim1','Dim2','label'])
print(df_tsne.head())

plt.figure(figsize=(8,8))
sns.scatterplot(data=df_tsne,hue='label',x='Dim1',y='Dim2')
plt.savefig('Scatter Program_TSNE600000.jpg', dpi=300)
plt.show()