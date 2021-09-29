from time import time
import efinance as ef
from datetime import datetime
from pyecharts import options as opts
from pyecharts.charts import Kline

stock_code = '600519' # 贵州茅台
freq = 1
status = {stock_code: 0}

df = ef.stock.get_quote_history(stock_code, klt=freq)
now = str(datetime.today()).split('.')[0] # 现在的时间
df.to_csv(f'{stock_code}.csv', encoding='utf-8-sig', index=None)
print(f'已在 {now}, 将股票: {stock_code} 的行情数据存储到文件: {stock_code}.csv 中！')
if len(df) == status[stock_code]:
    print(f'{stock_code} 已收盘')
status[stock_code] = len(df)


# 生成日k线的html
k_lines_data=df.loc[:, ['开盘', '收盘', '最高', '最低']]
data=k_lines_data.to_numpy()
data=data.tolist()
time_data=df.loc[:, ['日期']]
time_data=time_data.to_numpy()
time_data=time_data.tolist()
time_data=[time_[0] for time_ in time_data]
# print(time_data)
c = (
    Kline()
    .add_xaxis(time_data)
    .add_yaxis("kline", data)
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(is_scale=True),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        datazoom_opts=[opts.DataZoomOpts(pos_bottom="-2%")],
        title_opts=opts.TitleOpts(title="Kline-DataZoom-slider-Position"),
    )
    .render("日K线.html")
)