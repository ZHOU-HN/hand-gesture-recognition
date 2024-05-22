# 显示原始图像和增强后的图像
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2 as cv
import os

size = 64

def read_directory(directory_path):
    data_x=[]
    data_y=[]
    for filename in os.listdir(directory_path):
        img = cv.imread(directory_path+'/'+filename,)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
        #_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
        img[int(size*0.97):size,:] = 0
        data_x.append(np.array(img))
        data_y.append(int(filename.split('_')[1][1])-1)
    return np.array(data_x),np.array(data_y)

#x_train,y_train = read_directory("E:/PycharmProjects/hand-identify/DataSet/datasets2")
x_train,y_train = read_directory("./datasets2")
print("x_train[0].shape", x_train[0].shape)
x_train = x_train.reshape(x_train.shape[0], size, size, 1)  # 给数据增加一个维度，使数据和网络结构匹配


image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=.05,
    height_shift_range=.05,
    horizontal_flip=False,
    zoom_range=0.02
)
image_gen_train.fit(x_train)

print("xtrain",x_train.shape)
x_train_subset1 = np.squeeze(x_train[:12])
print("xtrain_subset1",x_train_subset1.shape)
print("xtrain",x_train.shape)
x_train_subset2 = x_train[:12]  # 一次显示12张图片
print("xtrain_subset2",x_train_subset2.shape)


fig = plt.figure(figsize=(20, 2))
plt.set_cmap('gray')

# 显示原始图片
for i in range(0, len(x_train_subset1)):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train_subset1[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
#plt.show()

# 显示增强后的图片
fig = plt.figure(figsize=(20, 2))
for x_batch in image_gen_train.flow(x_train_subset2, batch_size=12, shuffle=False):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i + 1)
        ax.imshow(np.squeeze(x_batch[i]))
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break

