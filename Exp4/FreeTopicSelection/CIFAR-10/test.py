import numpy as np
import scipy as sp
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import tensorflow as tf
import keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import layers,regularizers,models,backend,utils,optimizers
import os
from keras.models import model_from_json,load_model
from keras.datasets import cifar10


model = load_model('./VGG16model/VGG16weights.best.h5')

(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
img_row,img_col,channel = 32,32,3
mnist_input_shape = (img_row,img_col,channel)

#将数据维度进行处理
X_test = test_images.reshape(test_images.shape[0],img_row,img_col,channel)

X_test = X_test.astype("float32")

## 进行归一化处理
X_test /= 255
print(test_labels[1])
# one-hot独热码
test_labels = utils.to_categorical(test_labels,10)
error=[]
for i in range(500):
    predict_y=model.predict(X_test[i].reshape((1,32,32,3)))
    result=np.argmax(predict_y)
    ans=np.argmax(test_labels[i])
    if(i==1):
        print(result)
    if(result!=ans):
        error.append(i)
print(len(error))
num_sqrt = len(error) ** 0.5
num=int(num_sqrt)
plt.figure(figsize=(5, 5))
plt.subplots_adjust(hspace=0.1)
for n in range(len(error)):
    plt.subplot(num+1,num+1, n+1)
    plt.imshow(train_images[error[n]])
    plt.axis('off')
_ = plt.suptitle("Error Example")
plt.show()


    
