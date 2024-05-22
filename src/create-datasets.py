#采集数据集

import cv2
import numpy as np

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


if __name__ == '__main__':
    device_id = 0
    '''
    device_id {
              0:电脑摄像头
              1:手机摄像头}
    '''

    i = 1502 #存图开始的第一个索引


    if device_id == 0:
        cap = cv2.VideoCapture(0)  # 读取摄像头
    elif device_id == 1:
        url = "http://192.168.137.11:8080/video"  # 手机摄像头的URL（根据IP Webcam应用显示的URL修改）尾缀video不动
        cap = cv2.VideoCapture(url)  # 打开视频流

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
        cv2.imshow("roi",roi)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
        Laplacian = cv2.convertScaleAbs(dst)

        contour = B(Laplacian)#轮廓处理
        #contour = cv2.fastNlMeansDenoising(contour,None,10,7,21)
        cv2.imshow("contour",contour)

        #_,img = cv2.threshold(contour,127,255,cv2.THRESH_BINARY)

        #img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        img = cv2.resize(contour, (100, 100), interpolation=cv2.INTER_AREA)
        img = 255-img
        #print(img)
        #print(type(img))
        cv2.imshow("img",img)
        key = cv2.waitKey(50)# & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('./datasets2/' + str(i) + '_01_zhn' + '.jpg',img)
            #cv2.imwrite('E:\\opencv\\hand_5.jpg',255-contour)
            print("保存成功+",i)
            i = i + 1
    cap.release()
    cv2.destroyAllWindows()