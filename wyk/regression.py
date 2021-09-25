import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression


data = [[0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
        [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
        [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
        [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
        [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
        [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]]

dataMat = np.array(data)
X = dataMat[:, 0:1]
y = dataMat[:, 1]

model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(X, y)
print('Coefficient Matrix: ')
print(model.coef_)
print('Linear Regression Model: ')
print(model)

predicted = model.predict(X)
plt.scatter(X, y, marker='o')
plt.plot(X, predicted, c='pink')
plt.xlabel("x")
plt.ylabel("y")
plt.show()