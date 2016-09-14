# -*- coding:utf-8 -*-
import threading
from datetime import datetime
import cv2

from boss_train import Model
from image_show import show_image

"""
class FaceThread(threading.Thread):
    def __init__(self, frame):
        super(FaceThread, self).__init__()
        self._cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
        self._frame = frame

    def run(self):
        # グレースケール変換
        self._frame_gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

        # カスケード分類器の特徴量を取得する
        self._cascade = cv2.CascadeClassifier(self._cascade_path)

        # 物体認識（顔認識）の実行
        self._facerect = self._cascade.detectMultiScale(self._frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))

        if len(self._facerect) > 0:
            print('face detected')
            self._color = (255, 255, 255) # 白
            for self._rect in self._facerect:
                # 検出した顔を囲む矩形の作成
                cv2.rectangle(self._frame, tuple(self._rect[0:2]),tuple(self._rect[0:2] + self._rect[2:4]), self._color, thickness=2)

                x, y = self._rect[0:2]
                width, height = self._rect[2:4]
                image = self._frame[y: y + height, x: x + width]
                predict(image)
                cv2.imwrite("test.jpg", self._frame[y: y + height, x: x + width])

            # 現在の時間を取得
            self._now = datetime.now().strftime('%Y%m%d%H%M%S')
            # 認識結果の保存
            self._image_path = self._now + '.jpg'
            cv2.imwrite(self._image_path, self._frame)

# カメラをキャプチャ開始
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # frameを表示
    cv2.imshow('camera capture', frame)

    if(threading.activeCount() == 1):
        th = FaceThread(frame)
        th.start()

    # 10msecキー入力待ち
    k = cv2.waitKey(10)
    # Escキーを押されたら終了
    if k == 27:
        break

# キャプチャを終了
cap.release()
cv2.destroyAllWindows()
"""

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    model = Model()
    model.load()
    while True:
        _, frame = cap.read()

        # グレースケール変換
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        # 物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for rect in facerect:
                # 検出した顔を囲む矩形の作成
                #cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

                x, y = rect[0:2]
                width, height = rect[2:4]
                #image = frame[y - 50: y + height, x: x + width + 50]
                image = frame[y - 50: y + height, x: x + width]
                #image = frame[y: y + height, x: x + width]
                #now = datetime.now().strftime('%Y%m%d%H%M%S')
                #cv2.imwrite(now + '.jpg', image)
                result = model.predict(image)
                if result == 0:  # boss
                    print('Boss is approaching')
                    show_image()
                else:
                    print('Not boss')

        #10msecキー入力待ち
        k = cv2.waitKey(100)
        #Escキーを押されたら終了
        if k == 27:
            break

    #キャプチャを終了
    cap.release()
    cv2.destroyAllWindows()