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
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,normalization,BatchNormalization,Activation
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras import layers,regularizers,models,backend,utils,optimizers
from keras.datasets import cifar10
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

num_classes=10

(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
print(train_images.shape)
"""
plt.figure(figsize=(5, 3))
plt.subplots_adjust(hspace=0.1)
for n in range(15):
    plt.subplot(3, 5, n+1)
    plt.imshow(train_images[n])
    plt.axis('off')
_ = plt.suptitle("CIFAR-10 Example")
plt.show()
"""
img_row,img_col,channel = 32,32,3
input_shape = (img_row,img_col,channel)
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape,test_images.shape)

train_labels = utils.to_categorical(train_labels,num_classes)
test_labels = utils.to_categorical(test_labels,num_classes)
datagen_train = ImageDataGenerator(width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True)
 
datagen_train.fit(train_images)
print("data ready ")

checkpoint_dir = 'VGG16model/VGG16weights.best.h5'
if os.path.exists(checkpoint_dir):
    print('checkpoint exists, Load weights from %s\n'%checkpoint_dir)
    model = load_model('./VGG16model/VGG16weights.best.h5')
    eval = model.evaluate(test_images,test_labels, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100) )
else:
    print('No checkpoint found')
    model = models.Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',input_shape=(32,32,3),activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    #layer2 32*32*64
    model.add(Conv2D(64, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer3 16*16*64
    model.add(Conv2D(128, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer4 16*16*128
    model.add(Conv2D(128, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer5 8*8*128
    model.add(Conv2D(256, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer6 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer7 8*8*256
    model.add(Conv2D(256, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer8 4*4*256
    model.add(Conv2D(512, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer9 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer10 4*4*512
    model.add(Conv2D(512, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #layer11 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer12 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    #layer13 2*2*512
    model.add(Conv2D(512, (3, 3), padding='same',activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #layer14 1*1*512
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
    model.add(BatchNormalization())
    #layer15 512
    model.add(Dense(512,activation="relu"))
    model.add(BatchNormalization())
    #layer16 512
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizers.Adam(lr=0.003,decay=1e-7),metrics=['accuracy'])
model.summary()
#保存模型
filepath="VGG16model/VGG16weights.best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,save_best_only=True,mode='max',period=1)
#tensorboard
callbacks_list = [checkpoint,TensorBoard(log_dir="./logs/VGG16")]
#炼丹
#model.fit(train_images,train_labels,batch_size=32,epochs=20,verbose=1,validation_data=(test_images,test_labels),shuffle=True,callbacks=callbacks_list)
history=model.fit_generator(datagen_train.flow(train_images, train_labels, batch_size=32),steps_per_epoch=train_images.shape[0] //32,epochs = 100,verbose=1,callbacks=callbacks_list,validation_data=(test_images, test_labels),validation_steps=test_images.shape[0]//32)

