#通过手势识别玩游戏

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import cv2
import Dino

size = 64

#皮肤检测
def A(img):

    YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
    (y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5,5), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
    #_, skin = cv2.threshold(cr1, 132, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(img,img, mask = skin)
    return res

def B(img):

    #binaryimg = cv2.Canny(Laplacian, 50, 200) #二值化，canny检测
    h = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    contour = h[0]
    contour = sorted(contour, key = cv2.contourArea, reverse=True)#已轮廓区域面积进行排序
    #contourmax = contour[0][:, 0, :]#保留区域面积最大的轮廓点坐标
    bg = np.ones(dst.shape, np.uint8) *255#创建白色幕布
    ret = cv2.drawContours(bg,contour[0],-1,(0,0,0),3) #绘制黑色轮廓
    return ret


model_save_path = './checkpoint_zhn/Baseline.ckpt'
#np.set_printoptions(precision=1)

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

if __name__ == "__main__":
    device_id = 0
    '''
    device_id {
              0:电脑摄像头
              1:手机摄像头}
    '''
    if device_id == 0:
        cap = cv2.VideoCapture(0)  # 读取摄像头
    elif device_id == 1:
        url = "http://192.168.137.11:8080/video"  # 手机摄像头的URL（根据IP Webcam应用显示的URL修改）尾缀video不动
        cap = cv2.VideoCapture(url)  # 打开视频流
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    model = Baseline()

    model.load_weights(model_save_path)

    Test_Game = Dino.Game()
    Test_Dino = Dino.DinoAgent(Test_Game)


    while(True):

        ret, frame = cap.read()
        frame_shape = frame.shape

        # 下面提取roi区域可以根据自己的电脑进行调节
        src = cv2.resize(frame, (frame_shape[1] // 2, frame_shape[0] // 2), interpolation=cv2.INTER_CUBIC)  # 窗口大小
        src_shape = src.shape

        p1 = [0, 0]
        p2 = [0, 0]
        if device_id == 0:
            tmp = min(src_shape[1] // 3, src_shape[0] // 3)
        elif device_id == 1:
            tmp = min(src_shape[1] // 2, src_shape[0] // 2)
        p1[0] = src_shape[1] // 2 - tmp if src_shape[1] // 2 - tmp >= 0 else 0
        p1[1] = src_shape[0] // 2 - tmp if src_shape[0] // 2 - tmp >= 0 else 0
        p2[0] = src_shape[1] // 2 + tmp if src_shape[1] // 2 + tmp < src_shape[1] else src_shape[1] - 1
        p2[1] = src_shape[0] // 2 + tmp if src_shape[0] // 2 + tmp < src_shape[0] else src_shape[0] - 1

        cv2.rectangle(src, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0))  # 框出截取位置
        cv2.imshow("src", src)
        roi = src[p1[1]:p2[1], p1[0]:p2[0]]  # 获取手势框图

        res = A(roi)  # 进行肤色检测
        cv2.imshow("0",roi)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
        Laplacian = cv2.convertScaleAbs(dst)

        contour = B(Laplacian)#轮廓处理
        cv2.imshow("2",contour)


    #送入模型预测
        img = cv2.resize(contour, (size, size), interpolation=cv2.INTER_AREA)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        #img[int(size * 0.97):size, :] = 0
        cv2.imshow("img", img)
        img = img/255.0
        img = img.reshape(size, size, 1)
        x_predict = img[tf.newaxis, ...]
        result = model.predict(x_predict)

        pred = np.argmax(result, axis=1)
        print("\r识别结果:{}".format(pred),end='')

        if (pred == 5):
            Test_Dino.jump()
       # print("预测结果:{}".format(pred + 1))
        key = cv2.waitKey(25)# & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
