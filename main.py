from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import queue
import time

import numpy as np
from PyQt5.QtGui import QStandardItemModel

from untitled import *
import logging
import sys
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread,  QWaitCondition, QMutex, QFileInfo
from csv_operation import cvs_write2
from perceptual_hashing_algorithm import is_similar

draw_enabled = False
qmut_1 = QMutex()  # 创建线程锁
qmut_2 = QMutex()
framing = queue.Queue()   # 先进先出队列
reference_substance = queue.Queue()  # 对比相似度的参照图片
image = None

class MyWin(QMainWindow, Ui_train_window):
    def __init__(self, parent=None):
        super(MyWin, self).__init__(parent)
        self.setupUi(self)
        self.mylabel = MyLabel
        self.cap = cv2.VideoCapture()#视频流
        self.timer_camera = QtCore.QTimer()#定义定时器，用于控制显示视频的帧率
        self.signal = 0 #用于区分timer_camera是因为视频被关闭而stop，还是因为暂停动作而stop


        self.sld_duration.customSliderClicked.connect(self.slider_moved)  # 鼠标点击滑动条，接收mousePressEvent发送的信号后触发

        self.btn_select.clicked.connect(self.open)
        self.timer_camera.timeout.connect(self.Video_Play)
        self.btn_play_pause.clicked.connect(self.playPause)
        self.tag.clicked.connect(self.taggo)
        self.clear.clicked.connect(self.clear_clear)

        self.pushButton_start.clicked.connect(self.start)   # 开始
        self.pushButton_stop.clicked.connect(self.stop_thread)  # 结束

    def taggo(self):
        # 开启画矩形框的功能
        QMessageBox.information(self, "提示", "开启标记功能", QMessageBox.Yes)
        self.mylabel.marking = True

    def open(self):
        self.view.clear()  #每次想选择新视频时，显示界面都清空一下
        if self.timer_camera.isActive() == False and self.signal == 0:
            selectFileName, _ = QFileDialog.getOpenFileName(self, '选择文件', './')
            fileInfo = QFileInfo(selectFileName)
            baseName = fileInfo.fileName()
            self.LabCurMedia.setText(baseName)
            if selectFileName == '':
                QMessageBox.information(self, "提示", "请选择文件", QMessageBox.Yes)
            elif (selectFileName.lower().endswith(('.mp4', '.avi', 'flv')) == False):
                QMessageBox.warning(self, '警告', '请输入正确的视频格式', QMessageBox.Yes)
            else:
                flag = self.cap.open(selectFileName)
                if flag:  # 检测视频是否被打开
                    self.timer_camera.start(30)
                    QMessageBox.information(self, "提示", "请点击'开始标记'并选择目标区域", QMessageBox.Yes)
                    self.btn_select.setText('关闭')
        else:
            self.signal = 0
            self.btn_play_pause.setText('暂停')
            self.timer_camera.stop()
            self.cap.release()
            self.btn_select.setText('选择上传')
        # self.Ratio = self.cap.get(cv2.CAP_PROP_FPS)
        # self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @staticmethod
    def format_time(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def Video_Play(self):
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.sld_duration.setMaximum(self.total_frames)
        if self.total_frames < 0:
            self.sld_duration.setEnabled(False)
        else:
            self.sld_duration.setEnabled(True)
        flag, self.image = self.cap.read()  # 从视频流中读取.当flag为False时，表示视频位于最后一帧。
        if flag:  # 检测视频是否播放完毕
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            img = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                               QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.view.setPixmap(QtGui.QPixmap.fromImage(img))  # 往显示视频的Label里 显示QImage
            self.view.setScaledContents(True)  # 使画面完全展示+比例自适应。

            # 更新进度条和当前播放时间
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.sld_duration.setValue(self.current_frame)

            self.total_time = int(self.total_frames / self.cap.get(cv2.CAP_PROP_FPS))
            self.current_time = int(self.current_frame / self.cap.get(cv2.CAP_PROP_FPS))

            self.total_time_str = self.format_time(self.total_time)
            self.current_time_str = self.format_time(self.current_time)

            framing.put((self.image, self.current_time_str))#抽帧
            self.lab_duration.setText(f"{self.current_time_str} / {self.total_time_str}")
            self.setWindowTitle(f"Progress: {self.current_frame}/{self.total_frames}")

            # 在每一帧之后等待一段时间，以控制视频的播放速度
            cv2.waitKey(30)

        else:
            self.btn_select.setText('播放完毕')

    def playPause(self):
        ret, self.image = self.cap.read()
        if ret:  # 只有在视频被选中且播放时，这个按钮被按时才有效果
            if self.timer_camera.isActive() == False:
                self.timer_camera.start(40)
                self.btn_play_pause.setText('暂停')
            else:
                self.signal = 1
                self.timer_camera.stop()
                self.btn_play_pause.setText('播放')

    def slider_moved(self, value):  # 触发事件写在该函数中
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        print(f'{value}')

    def clear_clear(self):
        self.mylabel.bboxPointList.clear()

    def start(self):
        self.thread_screen = Screen()  # 创建图片分析线程类对象
        self.thread_screen.sinOut.connect(self.add_message)
        # self.thread_camera = ReadCamera(self.username, self.password, self.ip)  # 创建读取摄像头线程类对象
        self.pushButton_start.setEnabled(False)
        self.thread_screen.start()

    def add_message(self, inf):
        self.textBrowser.append(inf)

    # 停止线程
    def stop_thread(self):
        if  self.thread_screen:
            self.thread_screen.terminate()
            self.thread_screen = None
            self.pushButton_start.setEnabled(True)


    def onLabelAdded(self, label):
        self.listView.insertRow(self.listView.rowCount())
        index = self.model.index(self.listViewrowCount() - 1)
        self.listView.setData(index, label)

class Screen(QThread):  # 逐帧检测

    sinOut = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._isPause = False
        self.cond = QWaitCondition()
        self.mutex = QMutex()
        self.ui = Ui_train_window()
        self.mylabel = MyLabel()

        self.start_time = None
        self.black_screen_duration = 0
        self.all_black = True
        self.specific_duration = 15

        self.black_screen_dict = {}  # 记录黑屏开始时间的字典

        self.frames_folder = 'frames/'
        self.output_file = 'output.mp4'
        self.frame_rate = 30
        self.current_frame_num = 0
        self.last_frame = None
        self.last_frame_time = None
        self.black_screen_start_time = None
        self.black_screen_end_time = None

    def run(self):
        image: object = framing.get()[0]
        # reference_substance.put(image)
        self.prev_brightness = None
        self.flash_detected = False
        self.threshold = 20
        self.is_saved = False  # 标记是否已保存当前帧
        labels = []
        start_time = None
        while True:
            self.mutex.lock()  # 上锁
            if self._isPause:
                self.cond.wait(self.mutex)
            if framing.empty() is not True:
                frame = framing.get()  #一帧一帧取
                self.box = frame[0]
                self.time = frame[1]
                for x1, y1, x2, y2, text in self.mylabel.bboxPointList:
                    # 在帧中截取矩形框内的图像
                    self.box_frame = self.box[y1:y2, x1:x2]
                    print(self.box_frame)
                    is_black = self.black_check(self.box_frame, self.box)
                    print(is_black)
                    if is_black:  # 如果黑屏
                        if start_time is None:  # 如果黑屏刚开始
                            start_time = self.time  # 记录开始时间
                    else:  # 如果不是黑屏
                        if start_time is not None:
                            end_time = self.time  # 记录结束时间
                            labels.append((start_time, end_time))  # 将开始和结束时间添加到标签列表中
                            start_time = None  # 重置开始时间

                    if start_time is not None:  # 如果还存在未结束的黑屏
                        end_time = self.time  # 记录结束时间
                        labels.append((start_time, end_time))  # 将开始和结束时间添加到标签列表中



                # self.flashdetect()
            self.mutex.unlock()  # 解锁

    # 线程暂停
    def pause(self):
        self._isPause = True

    # 线程恢复
    def resume(self):
        self._isPause = False
        self.cond.wakeAll()

    def detect_flash(self):
        # 计算帧的亮度
        brightness = cv2.mean(self.box_frame)[0]

        return brightness

    def black_check(self, box_frame, box):

        box_frame_hsv = cv2.cvtColor(box_frame, cv2.COLOR_BGR2HSV)
        box_hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)


        mean_brightness_box = np.mean(box_hsv[:, :, 2])
        mean_brightness_frame = np.mean(box_frame_hsv[:, :, 2])

        if mean_brightness_box <= mean_brightness_frame:
            return True
        else:
            return False



    def flashdetect(self, frame):
        # 初始化上一帧的亮度值
        prev_brightness = None
        # 预处理帧
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算当前帧的亮度值
        brightness = cv2.mean(gray_frame)[0]
        # 检查是否有明显变化
        if prev_brightness is not None:
            if abs(brightness - prev_brightness) > threshold:
                # 这里可以发现闪屏情况，进行相应操作
                if brightness < prev_brightness:
                    # 闪黑屏的处理逻辑
                    pass
                else:
                    # 闪白屏的处理逻辑
                    pass

        # 更新上一帧的亮度值
        prev_brightness = brightness


    def jump_to_progressbar(self):
        self.ui.sld_duration.setValue(50)

    def on_label_clicked(self, index):
        # 获取被点击的标签项的行号
        row = index.row()

        # 获取对应的开始时间和结束时间
        start_time, end_time = self.label_list[row]

        # 设置进度条的范围和当前值
        self.progress_slider.setRange(0, 100)  # 假设进度条范围是0-100
        self.progress_slider.setValue(int((start_time / self.get_total_duration()) * 100))



if __name__ == "__main__":
        # 适应高DPI设备
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        # 适应Windows缩放
        QtGui.QGuiApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QtWidgets.QApplication(sys.argv)
        ui = MyWin()
        ui.show()
        sys.exit(app.exec_())

