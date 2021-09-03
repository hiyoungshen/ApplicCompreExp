import numpy as np
# Numpy的原生方法：
#     1) np.arange
#     2) np.ones
#     3) np.zeros

mm  = np.array((1,1,1))
nn  = np.array((1,2,3))
sum = mm + nn
sum = sum ** 2

mul = np.array([[1,2,3], [1,1,1]])
a1  = np.array([1,2,3])
a2  = np.array([3,4,5])
a12 = a1 * a2

ss = np.mat([1,2,3])
mm = np.matrix([1,2,3])
mulMat = mm * ss.T

# Numpy随机抽样库：
#   1) 随机纯小数函数rand
#   2) 随机正态分布纯小数函数randn
#   3) 随机整数函数randint
#   4) 随机整数函数random_integers
#   5) 随机浮点数random_sample
#   6) 随机数函数choice
#   7) 概率密度分布

# Numpy数学函数：
#   1) 三角函数 np.sin(x)  | np.arcsin(x) | np.hypot(x1,x2) | np.degrees(x) | np.radians(x) | np.deg2rad(x) | np.rad2deg(x)
#   2) 双曲函数 np.sinh(x) | np.arcsinh(x)
#   3) 数值修约
#   4) 求和、求积、差分
#   5) 指数和对数
#   6) 算术运算
#   7) 矩阵和向量积