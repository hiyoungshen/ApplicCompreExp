# 向量量化  scipy.cluster
# 数学常量  scipy.constants
# 快速傅里叶变换    scipy.fftpack
# 积分      scipy.integrate
# 插值      scipy.interpolate
# 数据输入输出      scipy.io
# 线性代数          scipy.linalg
# N维图像           scipy.ndimage
# 正交距离回归      scipy.odr
# 优化算法          scipy.optimize
# 信号处理          scipy.signal
# 稀疏矩阵          scipy.sparse
# 空间数据结构和算法 scipy.spatial
# 特殊数学函数      scipy.special
# 统计函数          scipy.stats

# Example 1: 最小二乘拟合
import numpy as np
import pylab as pl
from scipy import io as spio
from scipy.optimize import leastsq

def func(x, p):
    A, k, theta = p
    return A * np.sin(2 * np.pi * k * x + theta)

def residuals(p, y, x):
    return y - func(x, p)

x = np.linspace(0, -2*np.pi, 100)
A, k, theta = 10, 0.34, np.pi/6 # 真实数据的函数参数
y0 = func(x, [A, k, theta]) # 真实数据
y1 = y0 + 2 * np.random.randn(len(x)) # 加入噪声之后的实验数据 
p0 = [7, 0.2, 0] # 第一次猜测的函数拟合参数

plsq = leastsq(residuals, p0, args=(y1, x))
print (u"真实参数:", [A, k, theta]) 
print (u"拟合参数", plsq[0]) # 实验数据拟合后的参数
pl.plot(x, y0, label=u"真实数据")
pl.plot(x, y1, label=u"带噪声的实验数据")
pl.plot(x, func(x, plsq[0]), label=u"拟合数据")
pl.legend()
pl.show()


# Example 2: 函数最小值