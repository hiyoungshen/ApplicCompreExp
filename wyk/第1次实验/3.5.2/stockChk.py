# Question 2: 实时股价
#             可以查询股票当前价格。用户可以设定数据刷新频率，程序会用绿色和红色的箭头表示股价走势。

import efinance as ef
stock_code = '600519'   # 股票代码
stock_freq = 1          # 刷新频率
df = ef.stock.get_quote_history(stock_code, klt=stock_freq)
df.to_csv(f'stock_{stock_code}.csv', encoding='UTF-8-sig', index=None)



# 参考教程：https://www.zhihu.com/question/438404653