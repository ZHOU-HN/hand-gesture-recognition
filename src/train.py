#训练网络

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv

np.set_printoptions(threshold=np.inf)
size = 64

def read_directory(directory_path):
    data_x=[]
    data_y=[]
    for filename in os.listdir(directory_path):
        img = cv.imread(directory_path+'/'+filename,)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
        #_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        #img[int(size * 0.97):size, :] = 0
        data_x.append(np.array(img))
        data_y.append(int(filename.split('_')[1][1]))
    return np.array(data_x),np.array(data_y)

x_train,y_train = read_directory("./datasets2")
x_test,y_test = read_directory("./ceshi2")
#x_test,y_test = read_directory("E:/PycharmProjects/hand-identify/CNN-ver3/ceshi3")

x_train = x_train / 255.0
x_test = x_test / 255.0
print("x_train.shape", x_train.shape)
print("x_test.shape",x_test.shape)
x_train = x_train.reshape(x_train.shape[0], size, size, 1)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], size, size, 1)
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)

print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)

'''
#数据增强
image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=.02,
    height_shift_range=.02,
    horizontal_flip=False,
    zoom_range=0
)
image_gen_train.fit(x_train)
'''

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(6, activation='softmax')
        #self.d3 = Dropout(0.2)
        #self.f3 = Dense(5,activation="softmax")

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        #x = self.d3(x)
        #y = self.f3(x)
        return y


model = Baseline()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/Baseline.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test,y_test), validation_freq=1,
                    callbacks=[cp_callback])

#history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=200, validation_split=0.05, validation_freq=1,
#                    callbacks=[cp_callback])

#history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=128), epochs=10,validation_data=(x_test, y_test),
#          validation_freq=1,callbacks=[cp_callback])
model.summary()

'''
# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
'''
###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
'''
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
'''