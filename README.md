# 说明
程序修改自https://github.com/Hironsan/BossSensor.git,用来自己测试一下

# 环境
本程序可以在python2.7下跑通,有些警告,但是可以运行.
照片采集使用笔记本自带摄像头(效果不好~~)
tensorflow:0.12
需要安装opencv,hdf5,keras等,keras后端使用tensorflow
使用os:ubuntu15.04


# 程序说明
* camera_reader.py : 用于从摄像头拍摄照片,设置为采集10s,将采集的照片存放到read_pic/pic,opencv识别出的人脸存放在read_pic/_faces
* boss_train.py : 训练模型,要识别的人脸图片放在data/boss下,其他人脸图片放在data/other下
* camera_reader.py : 利用摄像头识别人脸,如果是输出"boss come ...",不是输出"boss not come ..."
* model : 存放训练好的模型
* read_pic : 存放摄像头采集的目标图片,pic存放整个图片,_faces存放pic中图片识别出的人脸,处于提交目录的目的,在里面放了一个图片

# 步骤
1. 运行pic_reader.py采集要识别的人脸图片
2. 将read_pic/_faces/* 转移到 data/boss下,将其他训练的人脸数据存放到 data/other下
3. 运行boss_train.py 进行模型训练
4. 运行camera_reader.py,查看模型预测情况