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
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import layers,regularizers,models,backend,utils,optimizers
from keras.datasets import mnist
import keras.backend.tensorflow_backend as KTF

#显存按需占用
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)


num_classes=10

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

img_row,img_col,channel = 28,28,1

mnist_input_shape = (img_row,img_col,1)

#将数据维度进行处理
X_train = X_train.reshape(X_train.shape[0],img_row,img_col,channel)
X_test = X_test.reshape(X_test.shape[0],img_row,img_col,channel)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

#进行归一化处理
X_train  /= 255
X_test /= 255

Y_train = utils.to_categorical(Y_train,num_classes)
Y_test = utils.to_categorical(Y_test,num_classes)

print(X_train.shape)
print(Y_train.shape)
print("data ready ")
#模型准备
checkpoint_path = './DNNmodel/DNNweights.best.h5'
if os.path.exists(checkpoint_path):
    print('checkpoint exists, Load weights from %s\n'%checkpoint_path)
    model = load_model('./DNNmodel/DNNweights.best.h5')
    eval = model.evaluate(X_test, Y_test, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100) )
else:
    print('No checkpoint found')
    model = models.Sequential()
    model.add(layers.Dense(units=1024,input_shape=mnist_input_shape,activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.003,decay=1e-7), metrics=['accuracy'])
model.summary()
#保存模型
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,save_best_only=True,mode='max',period=1)
#tensorboard
callbacks_list = [checkpoint,TensorBoard(log_dir="./logs/DNN")]
#炼丹开始
history=model.fit(X_train, Y_train, batch_size=32, epochs=10, shuffle=True, verbose=1,validation_data=(X_test,Y_test),callbacks=callbacks_list)
print("Training finished \n")