# Pandas主要功能：
#     1) 具备对齐功能的数据结构DataFrame, Series
#     2) 集成时间序列功能
#     3) 提供丰富的数学运算操作
#     4) 灵活处理缺失数据

import pandas as pd
s = pd.Series([1, 2, 3, np.nan, 6, 8])

pd.DataFrame({'one':pd.Series([1,2,3],   index=['a','b','c']),
              'two':pd.Series([1,2,3,4], index=['b','a','c','d'])})
