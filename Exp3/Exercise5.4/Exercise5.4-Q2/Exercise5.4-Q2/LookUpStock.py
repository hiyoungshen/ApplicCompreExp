# 导入 efinance 如果没有安装则需要通过执行命令: pip install efinance 来安装
import efinance as ef
import time
from datetime import datetime
from pyecharts import options as opts
from pyecharts.charts import Kline
# 股票代码
stock_code = '600519'
# 数据间隔时间为 1 分钟
freq = 1
status = {stock_code: 0}

while 1:
    # 获取最新一个交易日的分钟级别股票行情数据
    df = ef.stock.get_quote_history(
        stock_code, klt=freq)
    # 现在的时间
    now = str(datetime.today()).split('.')[0]
    # 将数据存储到 csv 文件中
    df.to_csv(f'{stock_code}.csv', encoding='utf-8-sig', index=None)
    print(f'已在 {now}, 将股票: {stock_code} 的行情数据存储到文件: {stock_code}.csv 中！')
    if len(df) == status[stock_code]:
        print(f'{stock_code} 已收盘')
        break
    status[stock_code] = len(df)
    print('暂停 60*60 秒')
    # time.sleep(60*60)
    # print('-'*10)

    # 生成日k线的html
    k_lines_data=df.loc[:, ['开盘', '收盘', '最高', '最低']]
    data_np=k_lines_data.to_numpy()
    data=data_np.tolist()
    time_data=df.loc[:, ['日期']]
    time_data=time_data.to_numpy()
    time_data=time_data.tolist()
    time_data=[time_[0] for time_ in time_data]
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

    # k_lines_data
    the_opening_quotation = data_np[:, 0]
    closing_quotation = data_np[:, 1]
    highest_price = data_np[:, 2]
    lowest_price = data_np[:, 3]
    # print(the_opening_quotation)
    import matplotlib.pyplot as plt
    # print(time_data.flatten())
    plt.figure(figsize=(9, 3))
    plt.plot(time_data, the_opening_quotation)
    plt.title("The opening quotation")
    x_major_locator = plt.MultipleLocator(50)
    #把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator=plt.MultipleLocator(10)
    #把y轴的刻度间隔设置为10，并存在变量里
    ax=plt.gca()
    #ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    #把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # plt.autoscale()
    # plt.annotate("maxmin",xy=(1,1.0),xytext=(1,0.8),color="blue",weight="bold",arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color="b"))
    # plt.suptitle('Line chart')
    plt.savefig("the_opening_quotation.png")
    plt.show()

print('全部股票已收盘')