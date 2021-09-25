import numpy as np
import os
import scipy as sp
import tensorflow as tf
import keras
import matplotlib.mlab as mlab
import keras.initializers as k_init
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,normalization
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import layers,regularizers,models,backend,utils,optimizers
from keras.datasets import cifar10

num_classes=10

(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
plt.figure(figsize=(5, 3))
plt.subplots_adjust(hspace=0.1)
for n in range(15):
    plt.subplot(3, 5, n+1)
    plt.imshow(train_images[n])
    plt.axis('off')
_ = plt.suptitle("CIFAR-10 Example")
plt.show()
img_row,img_col,channel = 32,32,3
input_shape = (img_row,img_col,channel)
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape,test_images.shape)

train_labels = utils.to_categorical(train_labels,num_classes)
test_labels = utils.to_categorical(test_labels,num_classes)

print("data ready ")

checkpoint_dir = 'CNNmodel/CNNweights.best.h5'
if os.path.exists(checkpoint_dir):
    print('checkpoint exists, Load weights from %s\n'%checkpoint_dir)
    model = load_model('./CNNmodel/CNNweights.best.h5')
    eval = model.evaluate(test_images,test_labels, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100) )
else:
    print('No checkpoint found')
    model = models.Sequential()
    model.add(Conv2D(128,kernel_size=(3,3),activation="relu",input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())#Flatten层用来将输入“压平”，即把多维的输入一维化，
    model.add(Dense(512,activation="relu"))#全连接层
    model.add(Dropout(0.2))
    model.add(Dense(512,activation="relu"))#全连接层
    model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizers.Adam(lr=0.003,decay=1e-7),metrics=['accuracy'])
model.summary()
#保存模型
filepath="CNNmodel/CNNweights.best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max',period=1)
#tensorboard
callbacks_list = [checkpoint,TensorBoard(log_dir="./logs/CNN")]
#炼丹
model.fit(train_images,train_labels,batch_size=32,epochs=10,verbose=1,validation_data=(test_images,test_labels),shuffle=True,callbacks=callbacks_list)
print("Training finished \n")