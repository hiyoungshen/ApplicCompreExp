

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

X, y = make_blobs(n_samples=125, centers=2, cluster_std=0.60, random_state=0)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=20, random_state=0)
plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter')
plt.show()

svc = SVC(kernel='linear')
svc.fit(train_X, train_y)

plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap='winter')
ax = plt.gca()
xlim = ax.get_xlim()
ax.scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap='winter')
w=svc.coef_[0]
a=-w[0]/w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a*xx - (svc.intercept_[0]/w[1])
plt.plot(xx, yy)
plt.show()


pred_y = svc.predict(test_X)
confusion_matrix(test_y, pred_y)
print(confusion_matrix())