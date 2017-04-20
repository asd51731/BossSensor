#!/usr/bin/python
# encoding: utf-8
import os

import cv2
import datetime
import shutil
import time

from boss_train import Model
from image_show import show_image
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw

#detectFaces()返回图像中所有人脸的矩形坐标（矩形左上、右下顶点）
#使用haar特征的级联分类器haarcascade_frontalface_default.xml，在haarcascades目录下还有其他的训练好的xml文件可供选择。
#注：haarcascades目录下训练好的分类器必须以灰度图作为输入。
def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


#保存人脸图
def saveFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        #将人脸保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = "read_pic/_faces/"
        count = 0
        for (x1,y1,x2,y2) in faces:
            file_name = os.path.join(save_dir,str(count)+image_name.split(".")[0].split("/")[-1] +".jpg")
            Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name)
            count+=1


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # 采集多长时间
    time_read = 10

    starttime = datetime.datetime.now()
    endtime = datetime.datetime.now()
    pic_path = "read_pic/pic/"
    shutil.rmtree(pic_path)
    os.mkdir(pic_path)
    save_dir = "read_pic/_faces/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    while (endtime - starttime).seconds < time_read:
        endtime = datetime.datetime.now()
        _, frame = cap.read()
        millis = int(round(time.time() * 1000))
        pic_name = pic_path + str(millis) + ".jpg"
        plt.imsave(pic_name,frame)
        saveFaces(pic_name)
        # plt.show()
        print("save : " + str(pic_name) )



