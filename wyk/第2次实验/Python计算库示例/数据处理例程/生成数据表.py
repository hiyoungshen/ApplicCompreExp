import numpy as np
import pandas as pd
df = pd.DataFrame(pd.read_csv('name.csv', header=1))
df = pd.DataFrame(pd.read_excel('name.xlsx'))

# 数据表信息查看
df.shape
df.info 
df.dtypes
df['B'].dtype
df.isnull()
df.head()
df.tail()

# 数据表清洗
#   1. 用数字0填充空值  
    df.fillna(value=0)
#   2. 使用列price的均值对NA进行填充
    df['prince'].fillna(df['price'].mean())
#   3. 清除city字段的字符空格
    df['city'] = df['city'].map(str.strip)
#   4. 大小写转换
    df['city'] = df['city'].str.lower()
#   5. 更改数据格式
    df['price'].astype('int')
#   6. 更改列名称
    df.rename(columns={'category': 'category-size'})
#   7. 数据替换
    df['city'].replace('sh', 'shanghai')


# 数据预处理：数据表合并、设置索引列、按照特定的值排序
# 数据提取：  按索引提取单行的数值、重设索引、设置日期为索引、提取前三个字符并生成数据
# 数据筛选：  使用与或非、query函数进行筛选
# 数据汇总：  对所有的列进行计数汇总
# 数据统计：  数据采样、计算标准差、协方差和相关系数
# 数据输出：  分析后的数据可以输出为xlsx格式和csv格式


