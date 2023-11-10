# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'label.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QFont, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QSlider, QLabel, QInputDialog, QWidget


class MySlider(QSlider):  # 继承QSlider
    customSliderClicked = pyqtSignal(int)  # 创建信号

    def __init__(self, parent=None):
        super(QSlider, self).__init__(parent)

    def mousePressEvent(self, QMouseEvent):  # 重写的鼠标点击事件
        super().mousePressEvent(QMouseEvent)
        pos = QMouseEvent.pos().x() / self.width()
        self.setValue(round(pos * self.maximum()))#设定滑动条滑块位置为鼠标点击处

        self.customSliderClicked.emit(self.value())  # 发送信号

class MyLabel(QLabel):
    marking = False #标记功能是否开启
    flag = False
    x1 = 0  # 左上角坐标
    y1 = 0
    x2 = 0  # 右下角坐标
    y2 = 0
    x2_realtime = 0  # 鼠标当前位置的坐标
    y2_realtime = 0

    bboxPointList = []  # 用来存放bbox左上和右下坐标及label，每个元素以(x1,y1,x2,y2,text)的形式储存
    labelList = []  # 存放label，会展示在旁边的listview中。
    defaultLabelId = 0

    drawLabelFlag = -1  # 是否加了一个框，因为弹出的输入label名字的对话框，可以点取消而不画Bbox

    def mousePressEvent(self, event):
        self.flag = True
        if self.marking:
            self.x1 = event.x()
            self.y1 = event.y()

    def mouseReleaseEvent(self, event):
        self.flag = False
        if self.marking:
            self.x2 = event.x()
            self.y2 = event.y()
            self.x2_realtime = self.x1
            self.y2_realtime = self.y1  # 这样就不用画出实时框了
            text, ok = QInputDialog().getText(QWidget(), '添加Label', '输入label:')
            if ok and text:
                text = self.getSpecialLabel(text)  # 这个函数是为了标签名不重复
                self.savebbox(self.x1, self.y1, self.x2, self.y2, text)
                self.labelList.append(text)
                self.drawLabelFlag *= -1  # 将标记变为正，表示画了
            elif ok:
                self.defaultLabelId += 1
                defaultLabel = 'label' + str(self.defaultLabelId)
                self.savebbox(self.x1, self.y1, self.x2, self.y2, defaultLabel)  # 这个函数在下面有解释
                self.labelList.append(defaultLabel)
                self.drawLabelFlag *= -1
            event.ignore()  # 将信号同时发给父部件，标签管理。

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter()
        painter.begin(self)
        for point in self.bboxPointList:
            rect = QRect(point[0], point[1], abs(point[0]-point[2]), abs(point[1]-point[3]))
            painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.drawRect(rect)
            painter.drawText(point[0], point[1], point[4])
        # 实时显示
        rect_realtime = QRect(self.x1, self.y1, abs(self.x1-self.x2_realtime), abs(self.y1-self.y2_realtime))
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        painter.drawRect(rect_realtime)
        painter.end()

    def mouseMoveEvent(self, event):
        if self.marking:
            if self.flag:
                self.x2_realtime = event.x()
                self.y2_realtime = event.y()
                self.update()

    def savebbox(self, x1, y1, x2, y2, text):
        bbox = (x1, y1, x2, y2, text)  # 两个点的坐标以一个元组的形式储存，最后一个元素是label
        self.bboxPointList.append(bbox)


    def getSpecialLabel(self, text):
        # 获得不重名的label
        index = 0
        text_new = text
        for label in self.labelList:
            if text == label.split(' ')[0]:
                index += 1
                text_new = text + ' ' + str(index)
        return text_new

class Ui_train_window(object):
    def setupUi(self, train_window):
        train_window.setObjectName("train_window")
        train_window.resize(787, 516)
        train_window.setMinimumSize(QtCore.QSize(750, 500))
        train_window.setMaximumSize(QtCore.QSize(800, 600))
        self.btn_select = QtWidgets.QPushButton(train_window)
        self.btn_select.setGeometry(QtCore.QRect(10, 10, 81, 31))
        self.btn_select.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_select.setAutoRepeatDelay(311)
        self.btn_select.setObjectName("btn_select")
        self.btn_play_pause = QtWidgets.QPushButton(train_window)
        self.btn_play_pause.setGeometry(QtCore.QRect(590, 460, 71, 31))
        self.btn_play_pause.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_play_pause.setAutoRepeatDelay(311)
        self.btn_play_pause.setObjectName("btn_play_pause")
        self.tag = QtWidgets.QPushButton(train_window)
        self.tag.setGeometry(QtCore.QRect(140, 460, 81, 31))
        self.tag.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tag.setAutoRepeatDelay(311)
        self.tag.setObjectName("tag")
        self.groupBox = QtWidgets.QGroupBox(train_window)
        self.groupBox.setGeometry(QtCore.QRect(10, 50, 571, 361))
        self.groupBox.setObjectName("groupBox")
        # self.view = QtWidgets.QLabel(self.groupBox)
        self.view = MyLabel(self.groupBox)
        self.view.setGeometry(QtCore.QRect(10, 20, 551, 331))
        self.view.setStyleSheet("border:2px solid lightgray")
        self.view.setObjectName("view")
        self.horizontalLayoutWidget = QtWidgets.QWidget(train_window)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 420, 561, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # self.sld_duration = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.sld_duration = MySlider(self.horizontalLayoutWidget)
        self.sld_duration.setOrientation(QtCore.Qt.Horizontal)
        self.sld_duration.setObjectName("sld_duration")
        self.horizontalLayout.addWidget(self.sld_duration)
        self.lab_duration = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.lab_duration.setObjectName("lab_duration")
        self.horizontalLayout.addWidget(self.lab_duration)
        self.clear = QtWidgets.QPushButton(train_window)
        self.clear.setGeometry(QtCore.QRect(360, 460, 81, 31))
        self.clear.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.clear.setAutoRepeatDelay(311)
        self.clear.setObjectName("clear")
        self.pushButton_stop = QtWidgets.QPushButton(train_window)
        self.pushButton_stop.setGeometry(QtCore.QRect(710, 460, 71, 31))
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.LabCurMedia = QtWidgets.QLabel(train_window)
        self.LabCurMedia.setGeometry(QtCore.QRect(110, 20, 441, 16))
        self.LabCurMedia.setObjectName("LabCurMedia")
        self.listWidget = QtWidgets.QListWidget(train_window)
        self.listWidget.setGeometry(QtCore.QRect(590, 50, 191, 391))
        self.listWidget.setObjectName("listWidget")
        self.not_tag = QtWidgets.QPushButton(train_window)
        self.not_tag.setGeometry(QtCore.QRect(250, 460, 81, 31))
        self.not_tag.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.not_tag.setAutoRepeatDelay(311)
        self.not_tag.setObjectName("not_tag")

        self.retranslateUi(train_window)
        QtCore.QMetaObject.connectSlotsByName(train_window)

    def retranslateUi(self, train_window):
        _translate = QtCore.QCoreApplication.translate
        train_window.setWindowTitle(_translate("train_window", "Form"))
        self.btn_select.setText(_translate("train_window", "打开文件"))
        self.btn_play_pause.setText(_translate("train_window", "开始检测"))
        self.tag.setText(_translate("train_window", "开始标记"))
        self.groupBox.setTitle(_translate("train_window", "预览画面"))
        self.view.setText(_translate("train_window", "                                    Waiting for video....."))
        self.lab_duration.setText(_translate("train_window", "00:00:00"))
        self.clear.setText(_translate("train_window", "清除标记"))
        self.pushButton_stop.setText(_translate("train_window", "结束检测"))
        self.LabCurMedia.setText(_translate("train_window", "文件名称显示"))
        self.not_tag.setText(_translate("train_window", "关闭标记"))
