#一个简单的应用，使用训练好的网络进行正向推理

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import cv2 as cv

model_save_path = './checkpoint_zhn/Baseline.ckpt'

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

model.load_weights(model_save_path)
preNum = int(input("input the number of test pictures:"))
ceshi_path = "./ceshi2"

for i in range(preNum):
    image_path = str(input("the name of test picture:"))
    img = cv.imread(ceshi_path+"/"+image_path)


#    img1 = cv.resize(img,(250, 250), interpolation=cv.INTER_CUBIC)
#    plt.figure()
#    plt.imshow(img1, cmap='gray')
#    plt.title("gray")

    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img2 = cv.resize(img_gray,(64, 64), interpolation=cv.INTER_AREA)
    plt.figure()
    plt.imshow(img2,cmap='gray')
    plt.title("thresh")

    img_thresh1 = img2
    img_thresh1 = img_thresh1 / 255.0
    img_thresh1 = img_thresh1.reshape(64, 64, 1)
    x_predict = img_thresh1[tf.newaxis, ...]
    result = model.predict(x_predict)
#    print('各类归一化概率:')
#    print(result)
    pred = tf.argmax(result, axis=1)

    print('\n')
    print("预测结果：")
    tf.print(pred)

    plt.pause(1)
    plt.close()