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
from keras.datasets import cifar10
import keras.backend.tensorflow_backend as KTF

#显存按需占用
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
def load_batch(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
        data = d['data']
        labels = d['labels']
        data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_data(path ='cifar-10-batches-py'):
    from keras import backend as K

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
num_classes=10

(train_images,train_labels),(test_images,test_labels) = load_data(path='/home/team04/cifar-10-batches-py')
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

train_labels = keras.utils.to_categorical(train_labels,num_classes)
test_labels = keras.utils.to_categorical(test_labels,num_classes)

print("data ready ")
#模型准备
checkpoint_path = './DNNmodel/DNNweights.best.h5'
if os.path.exists(checkpoint_path):
    print('checkpoint exists, Load weights from %s\n'%checkpoint_path)
    model = load_model('./DNNmodel/DNNweights.best.h5')
    eval = model.evaluate(test_images, test_labels, verbose=0)
    print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" % (eval[0], eval[1] * 100) )
else:
    print('No checkpoint found')
    model = models.Sequential()
    model.add(layers.Dense(units=512,input_shape=input_shape,activation='relu'))
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
history=model.fit(train_images, train_labels, batch_size=32, epochs=5, shuffle=True, verbose=1,validation_data=(test_images,test_labels),callbacks=callbacks_list)
print("Training finished \n")