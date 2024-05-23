# 手势识别

记录曾经的一个CV demo项目。

## 功能

通过电脑摄像头或手机摄像头进行实时拍摄，使用神经网络对手势进行识别，最终通过识别到的手势动作游玩chrome浏览器里的Dino恐龙小游戏。

<font color=blue>效果如下：</font>

![image](https://github.com/ZHOU-HN/hand-gesture-recognition/blob/main/playGame-short-bar.gif)

## 环境

+ python 3.7.7

+ tensorflow 2.1.0
+ opencv-python 4.4.0.42

### 文件说明

+ `create-datasets.py`：保存摄像头图像，采集数据构成数据集

+ `train.py` ：训练网络

+ `finnal-hand-detection.py`：使用网络进行手势识别

+ `playGame`：通过手势识别交互play game

