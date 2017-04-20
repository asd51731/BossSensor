# -*- coding:utf-8 -*-
import Image
import os

import cv2

from boss_train import Model
from matplotlib import pyplot as plt


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
        # if os.path.exists(save_dir):
        #     shutil.rmtree(save_dir)
        # os.mkdir(save_dir)
        count = 0
        for (x1,y1,x2,y2) in faces:
            file_name = os.path.join(save_dir,str(count)+image_name.split(".")[0].split("/")[-1] +".jpg")
            Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name)
            count+=1


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    model = Model()
    model.load()
    while True:
        _, frame = cap.read()
        pic_name = "pic.jpg"
        plt.imsave(pic_name, frame)
        faces = detectFaces(pic_name)
        if faces:
            rs = 1
            for (x1, y1, x2, y2) in faces:
                face_name = "face.jpg"
                Image.open(pic_name).crop((x1, y1, x2, y2)).save(face_name)
                image = cv2.imread(face_name)
                rs = model.predict(image)

                if rs == 0:
                    print("boss come ...")
                else:
                    print("boss not come ...")

    #キャプチャを終了
    cap.release()
    cv2.destroyAllWindows()
