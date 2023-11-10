from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import queue
import threading
import time

import numpy as np
from PyQt5.QtGui import QStandardItemModel, QImage, QPixmap
from pandas.core.common import all_none

from label import *
import logging
import sys
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread,  QWaitCondition, QMutex, QFileInfo
# from csv_operation import cvs_write2
# from perceptual_hashing_algorithm import is_similar

draw_enabled = False
qmut_1 = QMutex()  # 创建线程锁
qmut_2 = QMutex()
framing = queue.Queue()   # 先进先出队列
black_durations =queue.Queue()  # 记录每个矩形框内的黑屏帧
k = 0

class MyWin(QMainWindow, Ui_train_window):
    def __init__(self, parent=None):
        super(MyWin, self).__init__(parent)
        self.setupUi(self)
        self.mylabel = MyLabel
        self.cap = cv2.VideoCapture()#视频流
        self.timer_camera = QtCore.QTimer()#定义定时器，用于控制显示视频的帧率
        self.timer_camera.timeout.connect(self.Video_Play)
        self.signal = 0 #用于区分timer_camera是因为视频被关闭而stop，还是因为暂停动作而stop
        self.is_playing = True
        self.end_time = None

        self.sld_duration.customSliderClicked.connect(self.slider_moved)  # 鼠标点击滑动条，接收mousePressEvent发送的信号后触发
        self.listWidget.itemClicked.connect(self.list_item_clicked)

        self.btn_select.clicked.connect(self.open)

        self.btn_play_pause.clicked.connect(self.playPause)
        self.tag.clicked.connect(self.taggo)
        self.not_tag.clicked.connect(self.notgo)
        self.clear.clicked.connect(self.clear_clear)
        # self.pushButton_start.clicked.connect(self.start)   # 开始
        self.pushButton_stop.clicked.connect(self.stop_thread)  # 结束


    def list_item_clicked(self, item):
        start, end = item.data(1)  # item对象索引为1的数据
        start = start * 1000
        end = end * 1000
        print(start, end)
        # 设置视频的起始时间和结束时间
        self.cap.set(cv2.CAP_PROP_POS_MSEC, start)#以毫秒为单位
        self.end_time = end  # 保存结束时间

    def process_frame(self):
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        count = -1
        printed_label_1 = []
        i = 0
        label_1 = []
        key_statuses = []
        none_index = 0
        start_time_1 = None
        end_time_1 = None
        nested_dict = {}
        while True:
            count += 1  # 值计数，用于切片列表索引
            self.frame_list = black_durations.get()
            n = len(self.frame_list)
            for i in range(n):
                nested_dict[f"框{i + 1}"] = nested_dict.get(f"框{i + 1}", []) + [self.frame_list[i]]
            for key in nested_dict:#键是有几个框
                value = nested_dict.get(key)#值是每个框的列表
                print(len(value))
                sliced_list = value[none_index:]#先从0开始切片
                if sliced_list and sliced_list[-1] != None: #只判断新进来的那一个
                    if start_time_1 is None:
                        start_time_1 = sliced_list[-1]
                        label_1.append(start_time_1)
                        key_statuses.append(False)
                    if start_time_1 is not None:
                        end_time_1 = sliced_list[-1]
                        if len(label_1) >= 2:
                            label_1[1] = end_time_1
                        else:
                            label_1.append(end_time_1)
                        key_statuses.append(False)
                        second_time = self.time2second(label_1)
                    duration = second_time[1] - second_time[0]
                    if duration >= 3 and self.total_frames == self.current_frame:
                        item_text = f'{key}一直黑屏到播放结束'
                        item = QListWidgetItem(item_text)
                        item.setData(i + 1, (second_time[0], second_time[1]))
                        self.listWidget.addItem(item)
                        key_statuses.append(False)
                        none_index = count
                        start_time_1 = None
                        end_time_1 = None
                        label_1 = []
                if sliced_list and sliced_list[-1] == None:
                    if start_time_1 is not None and end_time_1 is not None:
                        second_time = self.time2second(label_1)
                        duration = second_time[1] - second_time[0]
                        if duration >= 3:
                            print(f"{key}长时间黑屏")
                            # 将键的状态添加到列表中
                            key_statuses.append(True)
                        if duration < 3:
                            item_text = f'{key}黑屏时间小于3秒'
                            item = QListWidgetItem(item_text)
                            item.setData(k + 1, (second_time[0], second_time[1]))
                            self.listWidget.addItem(item)
                            key_statuses.append(False)
                    none_index = count
                    start_time_1 = None
                    end_time_1 = None

            if key_statuses and all(key_statuses):
                print(key_statuses)
                if label_1 not in printed_label_1:
                    print(label_1)
                    printed_label_1.append(label_1)
                    item_text = f"{label_1}所有目标长时间黑屏"
                    item = QListWidgetItem(item_text)
                    item.setData(k + 1, (second_time[0], second_time[1]))
                    self.listWidget.addItem(item)
                    none_index = count
                    start_time_1 = None
                    end_time_1 = None
                    label_1 = []
                    key_statuses = []
            else:
                key_statuses = []


    def time2second(self, start_end_time):
        # 将输入的时间转换成秒['00:10:40', '00:10:47'] - > [600, 900]
        second_time = []
        start_hour, start_minute, start_second = map(int, start_end_time[0].split(":"))
        end_hour, end_minute, end_second = map(int, start_end_time[1].split(":"))
        start_seconds = start_hour * 3600 + start_minute * 60 + start_second
        end_seconds = end_hour * 3600 + end_minute * 60 + end_second
        if end_seconds < start_seconds:
            return -1
        second_time.append(start_seconds)
        second_time.append(end_seconds)
        return second_time

    def taggo(self):
        # 开启画矩形框的功能
        QMessageBox.information(self, "提示", "开启标记功能", QMessageBox.Yes)
        self.mylabel.marking = True

    def notgo(self):
        # 关闭画矩形框的功能
        QMessageBox.information(self, "提示", "关闭标记功能", QMessageBox.Yes)
        self.mylabel.marking = False

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
                    ret, frame = self.cap.read()  # 读取视频的下一帧
                    if ret:
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = rgb_image.shape
                        bytesPerLine = c * w  # 计算每行的字节数
                        image = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)  # 创建 QImage 对象
                        pixmap = QPixmap.fromImage(image)  # 创建 QPixmap 对象
                        self.view.setPixmap(pixmap)  # 在界面上显示视频帧
                        self.view.setScaledContents(True)  # 使画面完全展示+比例自适应。
                    self.play_pause()  # 添加这一行来自动播放视频
                    # self.timer_camera.start(30)
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

    def play_pause(self):
        if self.is_playing:
            self.timer_camera.stop()
            self.is_playing = False
            self.btn_play_pause.setText('开始检测')
        else:
            if not self.cap.isOpened():
                QMessageBox.warning(self, '警告', '请先选择视频文件', QMessageBox.Yes)
                return
            self.timer_camera.start(30)
            self.is_playing = True
            self.btn_play_pause.setText('暂停')

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

            framing.put((self.image, self.current_time_str, self.current_frame))#抽帧
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
                self.thread_screen = Screen()  # 创建图片分析线程类对象
                self.thread_screen.sinOut.connect(self.add_message)
                self.thread_screen.start()
                thread = threading.Thread(target=self.process_frame)
                thread.start()
                print("haha")
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


    def add_message(self, inf):
        self.textBrowser.append(inf)

    # 停止线程
    def stop_thread(self):
        if self.thread_screen:
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
        self.labels = []
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
        while True:
            self.mutex.lock()  # 上锁
            if self._isPause:
                self.cond.wait(self.mutex)
            if framing.empty() is not True:
                frame = framing.get()  #一帧一帧取
                self.box = frame[0]
                self.time = frame[1]
                self.current_frame = frame[2]
                self.black_frames = []
                for x1, y1, x2, y2, text in self.mylabel.bboxPointList:
                    # 在帧中截取矩形框内的图像
                    self.box_frame = self.box[y1:y2, x1:x2]
                    is_black = self.black_check(self.box_frame, self.box)
                    if is_black:
                        self.black_frames.append(self.time)
                    else:
                        self.black_frames.append(None)
                black_durations.put(self.black_frames)
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

        if mean_brightness_box < mean_brightness_frame:
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




if __name__ == "__main__":
        # 适应高DPI设备
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        # 适应Windows缩放
        QtGui.QGuiApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QtWidgets.QApplication(sys.argv)
        ui = MyWin()
        ui.show()
        sys.exit(app.exec_())

