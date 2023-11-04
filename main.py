import sys
import pandas as pd
import os
import threading
from dialog import *
from serialports import *
from datetime import datetime
from time import sleep
from PyQt5.QtWidgets import QMainWindow, QDialog, QTableWidgetItem, QMessageBox, QFileDialog, \
    QHeaderView, QLabel, QFormLayout, QDialogButtonBox
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal,  QDir, QDateTime, QFile


#接收线程
class SerialReceiver(QThread):

    received_data = pyqtSignal(str)
    received_data_finish = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.serial = None
        self.data_buffer = None

    def run(self):
        while True:
            if self.serial is not None:
                datas = self.serial.readline()
                hex_data = binascii.hexlify(datas).decode('utf-8')
                if hex_data:
                    self.data_buffer = hex_data.strip()  # 更新命令缓冲区
                    self.received_data.emit(hex_data)  # 发送接收到的原始数据信号
                    logging.info(f"接收数据: {hex_data}")

class DialogInfo(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(DialogInfo, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_go.setText("确认")
        self.pushButton_go.clicked.connect(self.accept)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.ser = SerialPorts() #串口
        self.receiver = SerialReceiver()#接收线程
        self.comboBox_com.addItems(self.ser.ports_list)#COM
        self.setpushButton(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        #串口设置
        self.radioButton_mik3.toggled.connect(self.updateComboBoxOptions)#为radioButton绑定信号槽函数
        self.radioButton_mip.toggled.connect(self.updateComboBoxOptions) #为radioButton绑定信号槽函数
        self.radioButton_mik53D.toggled.connect(self.updateComboBoxOptions) #为radioButton绑定信号槽函数
        self.radioButton_mik5.toggled.connect(self.updateComboBoxOptions) #为radioButton绑定信号槽函数
        self.pushButton_connect.clicked.connect(self.connect)#连接串口
        self.pushButton_disconnect.clicked.connect(self.disconnect)#断开串口
        #面版区域
        self.pushButton_clear.clicked.connect(self.clear)#清除
        self.buttom_sendout.clicked.connect(self.commandsendout)#手动发送
        self.buttom_senddata.clicked.connect(self.commandsendout2)#快捷发送
        #菜单操作
        self.comboBox.currentIndexChanged.connect(self.comboBoxAIndexChanged) #双框关联(1下拉框+1选择框)
        self.pushButton_act_3.clicked.connect(self.button_clicked)#执行下拉循环
        self.pushButton_start.clicked.connect(self.starttest) #下拉选择框发送
        self.pushButton_reset.clicked.connect(self.on_pushButton_reset_clicked)#初始化设备
        self.pushButton_insert.clicked.connect(self.insert) #脚本录制新增
        self.pushButton_create.clicked.connect(self.create) #脚本录制开始
        #批量导入执行
        self.pushButton_import.clicked.connect(self.importexcel)#导入excel
        self.pushButton_ouput_3.clicked.connect(self.exportData)#导出excel
        self.pushButton_stop_3.clicked.connect(self.stop) #停止执行
        self.pushButton_send_3.clicked.connect(self.table_send)#表格发送
        #面板按键
        self.pushButton_search.clicked.connect(self.search)#menu
        self.pushButton_ok.clicked.connect(self.ok)#ok
        self.pushButton_left.clicked.connect(self.left)#left
        self.pushButton_right.clicked.connect(self.right)#right
        self.pushButton_up.clicked.connect(self.up)#up
        self.pushButton_down.clicked.connect(self.down)#down
        self.pushButton_back.clicked.connect(self.back)#back
        self.pushButton_record.clicked.connect(self.record)#录像
        self.pushButton_capture.clicked.connect(self.capture)#抓图
        self.pushButton_freeze.clicked.connect(self.freeze)#冻结
        self.pushButton_compare.clicked.connect(self.device)#获取设备参数
        self.pushButton_mode.clicked.connect(self.mode) #mode


    #检查校验和并进行接收解析
    def parse_data(self, strs):
        data, data_int = self.ser.datacut(strs)
        if data_int[-1] != sum(data_int[:-1]) % 256:
            logging.info(f"校验和{data_int[-1]}不等于{sum(data_int[:-1])},校验失败")
            self.showtext(f"校验结果：F")
            logging.info(f"校验结果：F")
        if data_int[-1] == sum(data_int[:-1]) % 256 and strs[6:8] != 'ff':
            logging.info(f"校验和{data_int[-1]}等于{sum(data_int[:-1])},校验成功")
            self.showtext(f"校验结果：P\n")
            logging.info(f"校验结果：P")
        if data_int[-1] == sum(data_int[:-1]) % 256 and strs[6:8] == 'ff':
            logging.info(f"校验和{data_int[-1]}等于{sum(data_int[:-1])},但包含ff校验失败")
            self.showtext(f"校验结果：F")
            logging.info(f"校验结果：F")
        if strs[6:8] == 'ff':
            if strs[9] == '0':
                self.showtext("未知错误\n")
                logging.info(f"未知错误")
            elif strs[9] == '1':
                self.showtext("校验失败\n")
                logging.info("校验失败")
            elif strs[9] == '2':
                self.showtext("命令(数据1)非法\n")
                logging.info("命令(数据1)非法")
            elif strs[9] == '3':
                self.showtext("命令(数据2)非法\n")
                logging.info("命令(数据2)非法")
            elif strs[9] == '4':
                self.showtext("不支持操作\n")
                logging.info("不支持操作")
        if strs[:6] == '550640':
            if strs[6:8] == 'b5':
                if strs[9] == '1':
                    self.showtext("U盘当前状态：录像中\n")
                    logging.info("U盘当前状态：录像中")
                if strs[9] == '2':
                    self.showtext("U盘当前状态：保存中\n")
                    logging.info("U盘当前状态：保存中")
                if strs[9] == '3':
                    self.showtext("U盘当前状态：拍照中\n")
                    logging.info("U盘当前状态：拍照中")
                if strs[9] == '4':
                    self.showtext("U盘当前状态：空闲\n")
                    logging.info("U盘当前状态：空闲")
                if strs[9] == '0':
                    self.showtext("U盘当前状态：未插入\n")
                    logging.info("U盘当前状态：未插入")
        if strs[6:8] == 'b2':
            if strs[9] == '1':
                self.showtext("录像开始\n")
                logging.info("录像开始")
            if strs[9] == '3':
                self.showtext("录像结束\n")
                logging.info("录像结束")
        if strs[6:8] == 'b3':
            if strs[9] == '2':
                self.showtext("抓图成功\n")
                logging.info("抓图成功")
        if strs[6:8] == 'b4':
            if strs[9] == '4':
                self.showtext("聚焦完成\n")
                logging.info("聚焦完成")
        if strs[:6] == "550740":
            print("尝试获取U盘容量！")
            Hex = "0x"
            bytehex = bytearray()
            for k in range(8, 10):
                Hex += strs[k]
                bytehex.append(strs[k])
            dec = int(Hex, 16)
            num1 = str(dec)
            print("尝试获取U盘容量！", num1)
            self.showtext(f"当前U盘剩余容量:{num1}\n")
            logging.info(f"当前U盘剩余容量:{num1}\n")
            if int(num1) <= 1 and strs[10] == '0' and strs[11] == '0':
                dialog = QDialog(self)
                dialog.setWindowTitle("提示")
                l = QLabel(dialog)
                l.setText("\n   当前U盘容量不足1G，请注意！")
                dialog.setWindowModality(Qt.NonModal)
                dialog.resize(200, 40)
                dialog.show()
    #串口连接
    def connect(self):
        sleep(0.5)
        com = self.comboBox_com.currentText()
        baund = int(self.comboBox_baund.currentText())
        bytesize = int(self.comboBox_bytesize.currentText())
        parity = self.comboBox_parity.currentText()
        stopbits = int(self.comboBox_stopbits.currentText())
        xonxoff = self.comboBox_xonxoff.currentText()
        if parity == 'None':
            parity = 'N'
        if xonxoff == 'False':
            xonxoff = False
        else:
            xonxoff = True
        logging.info(com)
        try:
            self.ser.opencom(com, baund, bytesize, parity, stopbits, xonxoff)
            if self.ser.serial.isOpen():
                self.statusbar.showMessage(f"串口{self.ser.serial.name}连接成功", 0)
                self.setpushButton(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1)
            else:
                self.statusbar.showMessage(f"串口连接失败", 0)
            self.receiver.serial = self.ser.serial
            self.receiver.received_data.connect(self.showtext2)
            # self.receiver.received_data_finish.connect(self.start_execution)
            self.receiver.ser = self.ser
            self.receiver.start()
        except:
            self.statusbar.showMessage(f"串口连接失败", 0)

    #串口断开
    def disconnect(self):
        self.receiver.terminate()
        self.ser.closecom()
        if self.ser.serial.isOpen():
            self.statusbar.showMessage(f"串口{self.ser.serial.name}断开失败", 0)
        else:
            self.statusbar.showMessage(f"串口断开连接", 0)
            self.setpushButton(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    #设置按钮
    def setpushButton(self, connect=2, disconnect=2, starttest=2, search=2, clear=2, importexcel=2, ok=2,
                      up=2, left=2, right=2, down=2, back=2, freeze=2, record=2, capture=2, mode=2, device=2,
                      send_3=2, ouput_3=2, stop_3=2, buttom_sendout=2, buttom_senddata=2, pushButton_act_3=2):
        if connect != 2:
            self.pushButton_connect.setEnabled(connect)
        if disconnect != 2:
            self.pushButton_disconnect.setEnabled(disconnect)
        if starttest != 2:
            self.pushButton_start.setEnabled(starttest)
        if search != 2:
            self.pushButton_search.setEnabled(search)
        if clear != 2:
            self.pushButton_clear.setEnabled(clear)
        if importexcel != 2:
            self.pushButton_import.setEnabled(importexcel)
        if ok != 2:
            self.pushButton_ok.setEnabled(ok)
        if up != 2:
            self.pushButton_up.setEnabled(up)
        if left != 2:
            self.pushButton_left.setEnabled(left)
        if right != 2:
            self.pushButton_right.setEnabled(right)
        if down != 2:
            self.pushButton_down.setEnabled(down)
        if back != 2:
            self.pushButton_back.setEnabled(back)
        if freeze != 2:
            self.pushButton_freeze.setEnabled(freeze)
        if record != 2:
            self.pushButton_record.setEnabled(record)
        if capture != 2:
            self.pushButton_capture.setEnabled(capture)
        if mode != 2:
            self.pushButton_mode.setEnabled(mode)
        if device != 2:
            self.pushButton_compare.setEnabled(device)
        if send_3 != 2:
            self.pushButton_send_3.setEnabled(send_3)
        if ouput_3 != 2:
            self.pushButton_ouput_3.setEnabled(ouput_3)
        if stop_3 != 2:
            self.pushButton_stop_3.setEnabled(stop_3)
        if buttom_sendout != 2:
            self.buttom_sendout.setEnabled(buttom_sendout)
        if buttom_sendout != 2:
            self.buttom_senddata.setEnabled(buttom_senddata)
        if pushButton_act_3 != 2:
            self.pushButton_act_3.setEnabled(pushButton_act_3)
    #下拉选择框发送
    def starttest(self):
        if self.pushButton_connect.isEnabled():
            self.statusbar.showMessage("串口未连接", 0)
        else:
            command1 = "0x55"
            command2 = "0x06"
            command3 = "0x20"
            command4 = self.comboBox.currentData()
            selected_items = self.comboBox_2.get_selected()
            for item, value in selected_items.items():
                print(f"已选中的选项：{item}，对应的值：{value}")
                command5 = value
                nums = (int(0x55) + int(0x06) + int(0x20) + int(command4, 16) + int(value, 16)) % 256#把十六进制转换成了十进制并进行相加
                result = str(hex(nums))[2:]
                send_data = str(command1)[2:] + str(command2)[2:] + str(command3)[2:] + str(command4)[2:] + str(command5)[2:] + result
                # 获取当前日期和时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_data_split = ' '.join(send_data[i:i + 2] for i in range(0, len(send_data), 2))
                self.showtext(f"[{current_time}]发送{self.comboBox.currentText()}指令:{send_data_split}")
                self.ser.send(send_data)
                logging.info(f"发送选择下拉框指令:{send_data}")
                sleep(2)
                self.showtext("")
    """""            
    #下拉循环发送
    def start_execution(self):
        if self.pushButton_connect.isEnabled():
            self.statusbar.showMessage("串口未连接", 0)
        else:
            command1 = "0x55"
            command2 = "0x06"
            command3 = "0x20"
            command4 = self.comboBox.currentData()
            selected_items = self.comboBox_2.get_selected()
            loop_count = int(self.lineEdit_time_2.text())
            interval_time = float(self.lineEdit_time.text())
            if self.pushButton_act_3.text() == "停止":
                self.pushButton_act_3.setText("循环发送")
                self.is_stopped = True  # 当点击停止按钮时，将停止状态设置为 True
                self.statusbar.showMessage("循环已停止", 0)
            if self.pushButton_act_3.text() == "循环发送":
                self.pushButton_act_3.setText("停止")
                for i in range(loop_count):
                    for item, value in selected_items.items():
                        if self.is_stopped1:
                            break
                        command5 = value
                        nums = (int(0x55) + int(0x06) + int(0x20) + int(command4, 16) + int(value, 16)) % 256
                        result = str(hex(nums))[2:]
                        send_data = str(command1)[2:] + str(command2)[2:] + str(command3)[2:] + str(command4)[2:] + str(
                            command5)[2:] + result
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.receiver.send(send_data)
                        send_data_split = ' '.join(send_data[i:i + 2] for i in range(0, len(send_data), 2))
                        self.showtext(f"[{current_time}]发送{self.comboBox.currentText()}指令:{send_data_split}")
                        print(send_data)
                        sleep(interval_time)
                        self.showtext("")
                        logging.info(f"发送选择下拉框指令:{send_data}")
                    if self.is_stopped1:
                        break
        """""

    #下拉循环发送
    def start_execution(self):
        if self.pushButton_act_3.text() == "循环发送":
            if self.pushButton_connect.isEnabled():
                self.statusbar.showMessage("串口未连接", 0)
            else:
                self.command1 = "0x55"
                self.command2 = "0x06"
                self.command3 = "0x20"
                self.command4 = self.comboBox.currentData()
                self.selected_items = self.comboBox_2.get_selected()
                self.loop_count = int(self.lineEdit_time_2.text())
                self.interval_time = float(self.lineEdit_time.text())

                self.send_thread = threading.Thread(target=self.loop_sending)
                print("循环")
                self.send_thread.start()

                self.pushButton_act_3.setText("停止")
                self.statusbar.showMessage("开始循环发送")
    #循环发送send_thread
    def loop_sending(self):
        for i in range(self.loop_count):
            for item, value in self.selected_items.items():
                if self.stopped1:
                    break
                command5 = value
                nums = (int(0x55) + int(0x06) + int(0x20) + int(self.command4, 16) + int(value, 16)) % 256
                result = str(hex(nums))[2:]
                send_data = str(self.command1)[2:] + str(self.command2)[2:] + str(self.command3)[2:] + str(
                    self.command4)[2:] + str(command5)[2:] + result
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.ser.send(send_data)
                send_data_split = ' '.join(send_data[i:i + 2] for i in range(0, len(send_data), 2))
                self.showtext(f"[{current_time}]发送{self.comboBox.currentText()}指令:{send_data_split}")
                print(send_data)
                sleep(self.interval_time)
                logging.info(f"发送选择下拉框指令:{send_data}")
        self.pushButton_act_3.setText("循环发送")
        self.statusbar.showMessage("循环已停止")
    #双点击
    def button_clicked(self):
        if self.pushButton_act_3.text() == "循环发送":
            self.stopped1 = False  # 切换停止状态
            self.start_execution()
        elif self.pushButton_act_3.text() == "停止":
            self.stopped1 = True  # 切换停止状态
            self.pushButton_act_3.setText("循环发送")
            self.statusbar.showMessage("循环已停止")
            self.send_thread.join()
    #核心：发送函数
    def send_dataa(self, send_data, name):
        if self.pushButton_connect.isEnabled():
            self.statusbar.showMessage("串口未连接", 0)
        else:
            self.showtext(f"-----------------------------当前操作：{name}-----------------------------")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_data_split = ' '.join(send_data[i:i + 2] for i in range(0, len(send_data), 2))
            self.showtext(f"[{current_time}]发送{name}指令:{send_data_split}")
            self.ser.send(send_data)
            sleep(2)
            self.showtext("")
    #获取设备参数
    def device(self):
        send_data = '5506204b01c7'
        self.send_dataa(send_data, name="获取设备参数")
    #menu
    def search(self):
        send_data = '550620070183'
        self.send_dataa(send_data, name="menu")
    #up
    def up(self):
        send_data = '550620090185'
        self.send_dataa(send_data, name="up")
    #left
    def left(self):
        send_data = '5506200b0187'
        self.send_dataa(send_data, name="left")
    #ok
    def ok(self):
        send_data = '5506200d0189'
        self.send_dataa(send_data, name="ok")
    #right
    def right(self):
        send_data = '5506200c0188'
        self.send_dataa(send_data, name="right")
    #down
    def down(self):
        send_data = '5506200a0186'
        self.send_dataa(send_data, name="down")
    #back
    def back(self):
        send_data = '550620080184'
        self.send_dataa(send_data, name="back")
    #冻结
    def freeze(self):
        send_data = '55062001017d'
        self.send_dataa(send_data, name="冻结")
    #录像
    def record(self):
        send_data = '55062002017e'
        self.send_dataa(send_data, name="录像")
    #抓图
    def capture(self):
        send_data = '55062003017f'
        self.send_dataa(send_data, name="抓图")
    #Mode
    def mode(self):
        send_data = '5506200e018a'
        self.send_dataa(send_data, name="mode")
    #手动发送
    def commandsendout(self):
        command = self.textEdit_output.toPlainText()
        nums = [int(command[i:i + 2], 16) for i in range(0, len(command), 2)]
        result = hex(sum(nums))[2:].upper()
        send_data = command + result
        self.send_dataa(send_data, name="手动发送")
    #快捷发送
    def commandsendout2(self):
        command1 = self.lineEdit_senddata_1.text()
        command2 = self.lineEdit_senddata_2.text()
        command3 = self.lineEdit_senddata_3.text()
        command4 = self.lineEdit_senddata_4.text()
        command5 = self.lineEdit_senddata_5.text()
        nums = int(0x55) + int(0x06) + int(0x20) + int(command4, 16) + int(command5, 16)
        result = str(hex(nums))[2:]
        send_data = command1 + command2 + command3 + command4 + command5 + result
        self.send_dataa(send_data, name="快捷发送")
    #接收显示
    def showtext2(self, strs):
        for j in range(len(strs)):
            if strs[j: j + 4] == self.ser.checkhead:  # 查找返回值中与头一致的字符串
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_data_split = ' '.join(strs[i:i + 2] for i in range(0, len(strs), 2))
                self.showtext(f"[{current_time}]接收指令：{send_data_split}")
                # self.textEdit.append()
                self.parse_data(strs[j:j + 12])
    #其他显示
    def showtext(self, strs):
        self.textEdit.append(strs)
        QApplication.processEvents()
    #清空
    def clear(self):
        self.textEdit.clear()
    #导入excel
    def importexcel(self):
        file_dialog = QFileDialog()
        file_types = "All Files (*)"
        folder_path, _ = file_dialog.getOpenFileName(None, "Select File", "", file_types)
        if folder_path:
            try:
                self.statusbar.showMessage(f"配置文件导入成功", 0)
            except:
                self.statusbar.showMessage(f"配置文件导入失败", 0)
        df = pd.read_excel(folder_path, dtype=object)
        # 取消列宽的截断显示，使得列的内容能够完整地显示出来
        pd.set_option('display.max_colwidth', None)
        # 获取表格的行数和列数
        num_rows, num_cols = df.shape
        # 设置QTableWidget的行列数
        self.tableWidget.setRowCount(num_rows)
        self.tableWidget.setColumnCount(num_cols)
        #设置表头
        self.tableWidget.setHorizontalHeaderLabels(df.columns)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)#题栏自适应大小的设置
        # 将数据加载到QTableWidget中
        for row in range(num_rows):
            for col in range(num_cols):
                # item = QTableWidgetItem()
                item = QTableWidgetItem(str(df.iloc[row, col]))
                self.tableWidget.setItem(row, col, item)
    #导出excel
    def exportData(self):
        # 获取表格的行数和列数
        num_rows = self.tableWidget.rowCount()
        num_cols = self.tableWidget.columnCount()

        # 将表格数据存储到 DataFrame 中
        data = []
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                item = self.tableWidget.item(i, j)
                if item is not None:
                    row.append(item.text())
                else:
                    row.append('')
            data.append(row)

        df = pd.DataFrame(data)

        header = []
        for j in range(num_cols):
            header.append(self.tableWidget.horizontalHeaderItem(j).text())
        df.columns = header

        # 弹出文件选择对话框，选择保存的Excel文件路径
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx)")

        if file_path:
            # 将数据保存为Excel文件
            df.to_excel(file_path, index=False)
            self.statusbar.showMessage(f"数据导出成功！", 0)
    #返回单选选项
    def getSelectedRadioButton(self):
        if self.radioButton_mik3.isChecked():
            return "radioButton_mik3"
        if self.radioButton_mik5.isChecked():
            return "radioButton_mik5"
        if self.radioButton_mik53D.isChecked():
            return "radioButton_mik53D"
        if self.radioButton_mip.isChecked():
            return "radioButton_mip"
    #单选选项改变下拉框
    def updateComboBoxOptions(self, is_radioButton_mik3_checked):
        self.comboBox.clear()
        selected_option = self.getSelectedRadioButton()
        if is_radioButton_mik3_checked:
            options = [("色调", "0x10"), ("白平衡模式", "0x11"), ("数字降噪", "0x12"), ("电子放大", "0x13"), ("电子缩小", "0x14")
                       , ("亮度", "0x15"), ("锐度", "0x16"), ("饱和度", "0x17"), ("对比度", "0x18"), ("伽马", "0x19"), ("暗区改善", "0x1a")
                       , ("高亮抑制", "0x1b"), ("去摩尔纹", "0x1c"), ("图像翻转", "0x1d"), ("双镜显示", "0x1f"), ("U盘操作", "0x20")
                       , ("手柄按键1", "0x21"), ("手柄按键2", "0x22"), ("手柄按键3", "0x23"), ("手柄按键4", "0x24"), ("语言选择", "0x25")
                       , ("视频质量", "0x26"), ("图片质量", "0x27"), ("增益", "0x28"), ("分辨率设置", "0x29"), ("帧率设置", "0x2A")
                       , ("图片格式", "0x2B"), ("脚踏配置", "0x2C"),  ("日期-年", "0x40"), ("日期-月", "0x41")
                       , ("日期-日", "0x42"), ("日期-时", "0x43"), ("日期-分", "0x44"), ("场景切换", "0x45"), ("去雾", "0x46")
                       , ("防红溢出", "0x47"), ("细节滤镜", "0x48"), ("画中画", "0x49"), ("彩条", "0x4A"), ("上报状态信息", "0x4B")]
        if selected_option == "radioButton_mik3":
            options = [("色调", "0x10"), ("白平衡模式", "0x11"), ("数字降噪", "0x12"), ("电子放大", "0x13"), ("电子缩小", "0x14")
                       , ("亮度", "0x15"), ("锐度", "0x16"), ("饱和度", "0x17"), ("对比度", "0x18"), ("伽马", "0x19"), ("暗区改善", "0x1a")
                       , ("高亮抑制", "0x1b"), ("去摩尔纹", "0x1c"), ("图像翻转", "0x1d"), ("双镜显示", "0x1f"), ("U盘操作", "0x20")
                       , ("手柄按键1", "0x21"), ("手柄按键2", "0x22"), ("手柄按键3", "0x23"), ("手柄按键4", "0x24"), ("语言选择", "0x25")
                       , ("视频质量", "0x26"), ("图片质量", "0x27"), ("增益", "0x28"), ("分辨率设置", "0x29"), ("帧率设置", "0x2A")
                       , ("图片格式", "0x2B"), ("脚踏配置", "0x2C"),  ("日期-年", "0x40"), ("日期-月", "0x41")
                       , ("日期-日", "0x42"), ("日期-时", "0x43"), ("日期-分", "0x44"), ("场景切换", "0x45"), ("去雾", "0x46")
                       , ("防红溢出", "0x47"), ("细节滤镜", "0x48"), ("画中画", "0x49"), ("彩条", "0x4A"), ("上报状态信息", "0x4B")]
        if selected_option == "radioButton_mik5":
            options = [("色调", "0x10"), ("白平衡模式", "0x11"), ("数字降噪", "0x12"), ("电子放大/缩小", "0x13"), ("亮度", "0x15")
                       , ("锐度", "0x16"), ("饱和度", "0x17"), ("对比度", "0x18"), ("伽马", "0x19"), ("暗区改善", "0x1a")
                       , ("高亮抑制", "0x1b"), ("去摩尔纹", "0x1c"), ("图像翻转", "0x1d"), ("双镜显示", "0x1f"), ("U盘格式化", "0x20")
                       , ("手柄按键1", "0x22"), ("手柄按键2", "0x23"), ("手柄按键3", "0x24"), ("语言选择", "0x25")
                       , ("视频质量", "0x26"), ("图片质量", "0x27"), ("增益", "0x28"), ("分辨率帧率设置", "0x29")
                       , ("图片格式", "0x2B"), ("脚踏配置", "0x2C"), ("手动白平衡RGAIN", "0x2D"), ("手动白平衡BGAIN", "0x2E"), ("日期-年", "0x40")
                       , ("日期-月", "0x41"), ("日期-日", "0x42"), ("日期-时", "0x43"), ("日期-分", "0x44"), ("场景切换", "0x45"), ("去雾", "0x46")
                       , ("防红溢出", "0x47"), ("细节滤镜", "0x48"), ("画中画", "0x49"), ("显示模式", "0x4A"), ("主屏显示", "0x4B"), ("融合色彩", "0x4C")
                       , ("荧光增益", "0x4D"), ("荧光亮度", "0x4E"), ("荧光对比度", "0x4F"), ("显彰饱和度", "0x50"), ("轮廓模式", "0x51")
                       , ("开始录像", "0x52"), ("结束录像", "0x53"), ("抓图", "0x54"), ("存储位置选择", "0x55"), ("U盘弹出", "0x56")
                       , ("硬盘导出开始", "0x57"), ("硬盘导出结束", "0x58"), ("硬盘格式化", "0x59"), ("智能曝光", "0x5A"), ("串口回传", "0x5B")
                       , ("蜂鸣器", "0x5C")]
        if selected_option == "radioButton_mik53D":
            options = [("色调", "0x10"), ("白平衡模式", "0x11"), ("数字降噪", "0x12"), ("电子放大/缩小", "0x13"), ("亮度", "0x15")
                       , ("锐度", "0x16"), ("饱和度", "0x17"), ("对比度", "0x18"), ("伽马", "0x19"), ("暗区改善", "0x1a")
                       , ("高亮抑制", "0x1b"), ("去摩尔纹", "0x1c"), ("图像翻转", "0x1d"), ("双镜显示", "0x1f"), ("U盘格式化", "0x20")
                       , ("手柄按键1", "0x22"), ("手柄按键2", "0x23"), ("手柄按键3", "0x24"), ("语言选择", "0x25")
                       , ("视频质量", "0x26"), ("图片质量", "0x27"), ("增益", "0x28"), ("分辨率帧率设置", "0x29")
                       , ("图片格式", "0x2B"), ("脚踏配置", "0x2C"), ("手动白平衡RGAIN", "0x2D"), ("手动白平衡BGAIN", "0x2E"), ("日期-年", "0x40")
                       , ("日期-月", "0x41"), ("日期-日", "0x42"), ("日期-时", "0x43"), ("日期-分", "0x44"), ("场景切换", "0x45"), ("去雾", "0x46")
                       , ("防红溢出", "0x47"), ("细节滤镜", "0x48"), ("画中画", "0x49"), ("显示模式", "0x4A"), ("主屏显示", "0x4B"), ("融合色彩", "0x4C")
                       , ("荧光增益", "0x4D"), ("荧光亮度", "0x4E"), ("荧光对比度", "0x4F"), ("显彰饱和度", "0x50"), ("轮廓模式", "0x51")
                       , ("开始录像", "0x52"), ("结束录像", "0x53"), ("抓图", "0x54"), ("存储位置选择", "0x55"), ("U盘弹出", "0x56")
                       , ("硬盘导出开始", "0x57"), ("硬盘导出结束", "0x58"), ("硬盘格式化", "0x59"), ("智能曝光", "0x5A"), ("串口回传", "0x5B")
                       , ("蜂鸣器", "0x5C")]
        if selected_option == "radioButton_mip":
            options = [("暂无", "0x00")]
            self.pushButton_mode.show()
        else:
            self.pushButton_mode.hide()

        for label, value in options:
            self.comboBox.addItem(label, value)
        self.comboBoxAIndexChanged()
    #表格导入批量发送
    def table_send(self):
        if self.tableWidget.columnCount() > 2:
            self.is_stopped = False
            th = 1
            self.tableWidget.setColumnCount(14)
            strs = ["名称", "命令", "命令", "命令", "命令", "命令", "校验位", "返回值", "返回值", "返回值", "返回值", "返回值", "校验位", "结论"]
            self.tableWidget.setHorizontalHeaderLabels(strs)
            row = self.tableWidget.rowCount()
            # 获取循环次数和发送间隔时间的输入值
            loop_count = int(self.lineEdit_cycleNum.text())
            interval_time = float(self.lineEdit_cycleTime.text())
            for z in range(loop_count):
                for i in range(row):
                    if self.tableWidget.isRowHidden(i):
                        self.tableWidget.removeRow(i)
                    if self.is_stopped:
                        break  # 如果停止状态为 True，则跳出循环
                    if th == 1:
                        name = self.tableWidget.item(i, 0).text()
                        self.tableWidget.scrollToItem(self.tableWidget.item(i, 1))
                        self.textEdit.insertPlainText(f"---------------------------{name}---------------------------")
                        hang = ""
                        for j in range(1, 6):
                            cell = self.tableWidget.item(i, j).text()
                            hang = hang + cell
                        bytehex = bytes.fromhex(hang)  # hang为十六进制字符串
                        check = 0
                        for k in range(0, 5):
                            check += bytehex[k]
                        check1 = hex(check)
                        item = QTableWidgetItem(hex(check)[2:])  # 将 check 转换为十六进制字符串并创建 QTableWidgetItem 对象
                        send_data = hang + str(check1)[2:]
                        print(f"send_data:{send_data}")
                        self.tableWidget.setItem(i, 6, item)  # 将 QTableWidgetItem 设置到指定的单元格
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        send_data_split = ' '.join(send_data[i:i + 2] for i in range(0, len(strs), 2))
                        self.showtext(f"[{current_time}]发送指令：{send_data_split}")
                        self.ser.send(send_data)
                        sleep(2)
                        self.showtext("")
                        data_recv = self.receiver.data_buffer
                        sleep(interval_time - 2)  # 循环发送间隔时间
                        #返回数据填充
                        if data_recv == None:
                            for y in range(7, 13):
                                self.tableWidget.setItem(i, y, QTableWidgetItem("null"))
                        item = QTableWidgetItem("F")
                        # 设置目标单元格的前景色为红色
                        red_color = QColor("red")
                        brush = QBrush(red_color)
                        item.setForeground(brush)
                        self.tableWidget.setItem(i, 13, item)

                        if data_recv != None and len(data_recv) == 12:
                            datacut_1 = self.ser.datacut1(data_recv)
                            print(f"datacut_1:{datacut_1}")
                            for m, item in enumerate(datacut_1):
                                self.tableWidget.setItem(i, m + 7, QTableWidgetItem(str(item)))
                            data, data_int = self.ser.datacut(data_recv)
                            if data_int[-1] != sum(data_int[:-1]) % 256:
                                item = QTableWidgetItem("F")
                                red_color = QColor("red")
                                brush = QBrush(red_color)
                                item.setForeground(brush)
                                self.tableWidget.setItem(i, 13, item)
                            if data_int[-1] == sum(data_int[:-1]) % 256 and data_recv[6:8] != 'ff':
                                item = QTableWidgetItem("P")
                                self.tableWidget.setItem(i, 13, item)
                            if data_int[-1] == sum(data_int[:-1]) % 256 and data_recv[6:8] == 'ff':
                                item = QTableWidgetItem("F")
                                red_color = QColor("red")
                                brush = QBrush(red_color)
                                item.setForeground(brush)
                                self.tableWidget.setItem(i, 13, item)

                        if data_recv != None and len(data_recv) > 24:
                            extracted_data = data_recv[-24: -12]
                            print(f"dextracted_data:{extracted_data}")
                            datacut_1 = self.ser.datacut1(extracted_data)
                            for m, item in enumerate(datacut_1):
                                self.tableWidget.setItem(i, m + 7, QTableWidgetItem(str(item)))
                            data, data_int = self.ser.datacut(extracted_data)
                            if data_int[-1] != sum(data_int[:-1]) % 256:
                                item = QTableWidgetItem("F")
                                red_color = QColor("red")
                                brush = QBrush(red_color)
                                item.setForeground(brush)
                                self.tableWidget.setItem(i, 13, item)
                            if data_int[-1] == sum(data_int[:-1]) % 256 and extracted_data[6:8] != 'ff':
                                item = QTableWidgetItem("P")
                                self.tableWidget.setItem(i, 13, item)
                            if data_int[-1] == sum(data_int[:-1]) % 256 and extracted_data[6:8] == 'ff':
                                item = QTableWidgetItem("F")
                                red_color = QColor("red")
                                brush = QBrush(red_color)
                                item.setForeground(brush)
                                self.tableWidget.setItem(i, 13, item)

                        if data_recv != None and len(data_recv) == 24:
                            extracted_data = data_recv[-12: -1]
                            print(f"dextracted_data2:{extracted_data}")
                            datacut_1 = self.ser.datacut1(extracted_data)
                            for m, item in enumerate(datacut_1):
                                self.tableWidget.setItem(i, m + 7, QTableWidgetItem(str(item)))
                            data, data_int = self.ser.datacut(extracted_data)
                            if data_int[-1] != sum(data_int[:-1]) % 256:
                                item = QTableWidgetItem("F")
                                red_color = QColor("red")
                                brush = QBrush(red_color)
                                item.setForeground(brush)
                                self.tableWidget.setItem(i, 13, item)
                            if data_int[-1] == sum(data_int[:-1]) % 256 and extracted_data[6:8] != 'ff':
                                item = QTableWidgetItem("P")
                                self.tableWidget.setItem(i, 13, item)
                            if data_int[-1] == sum(data_int[:-1]) % 256 and extracted_data[6:8] == 'ff':
                                item = QTableWidgetItem("F")
                                red_color = QColor("red")
                                brush = QBrush(red_color)
                                item.setForeground(brush)
                                self.tableWidget.setItem(i, 13, item)
            msg_box = QMessageBox(QMessageBox.Information, "完成", "命令已批量执行完毕！")
            msg_box.exec_()
    #停止执行
    def stop(self):
        self.is_stopped = True
    #初始化设备
    def on_pushButton_reset_clicked(self):
        if self.pushButton_connect.isEnabled():
            self.statusbar.showMessage("串口未连接", 0)
        else:
            self.textEdit.insertPlainText(f"初始化设备")
            self.get_reset_cmds()
            for i in range(len(self.resetCmd)):
                bytehex = self.resetCmd[i]
                numbers = bytehex.split()
                result = ''.join(numbers)
                total = 0
                for number in numbers:
                    decimal = int(number, 16)
                    total += decimal
                hex_result = hex(total)
                send_data = result + hex_result[2:]
                print(send_data)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_data_split = ' '.join(send_data[i:i + 2] for i in range(0, len(send_data), 2))
                self.showtext(f"[{current_time}]发送指令:{send_data_split}")
                self.ser.send(send_data)
                sleep(2)
                self.showtext("")
    #读取cmd初始化命令
    def get_reset_cmds(self):
        self.resetCmd = []
        # file_path = QApplication.applicationDirPath() + "/reset_cmd.txt"
        file_path = os.getcwd() + "/reset_cmd.txt"

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    str_line = line.strip()
                    self.resetCmd.append(str_line[:14])
        except FileNotFoundError:
            print("Can't open the file!")

        return self.resetCmd
    #脚本录制：新增
    def insert(self):
        if self.pushButton_create.text() == "完成":
            dialog = QDialog()
            form = QFormLayout(dialog)
            dialog.setWindowTitle("操作名称")
            form.addRow(QLabel("请输入操作名称:"))
            li = QLineEdit(dialog)
            form.addRow(li)

            buttonBox = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                Qt.Horizontal, dialog
            )
            form.addRow(buttonBox)
            buttonBox.accepted.connect(dialog.accept)
            buttonBox.rejected.connect(dialog.reject)

            # print(self.cmdStr_en)

            if dialog.exec() == QDialog.Accepted:
                self.tableWidget_panel.setColumnCount(2)
                strs = ["名称", "操作"]
                self.tableWidget_panel.setHorizontalHeaderLabels(strs)
                header = self.tableWidget_panel.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.Stretch)   #适应面板宽度
                curRow = self.tableWidget_panel.rowCount()
                self.tableWidget_panel.insertRow(curRow)
                name = QTableWidgetItem()
                cmds = QTableWidgetItem()

                name.setText(li.text())
                # cmds.setText(self.cmdStr_en)

                self.tableWidget_panel.setItem(curRow, 0, name)
                self.tableWidget_panel.setItem(curRow, 1, cmds)
                # SPIS_write(cell_index, li.text(), cmdStr_en)
                # print(self.cmdStr_en)
    #脚本录制：开始
    def create(self):
        # dirName = os.path.join(QCoreApplication.applicationDirPath(), "SPIS")
        current_dir = os.getcwd()  # 获取当前目录的路径
        dirName = os.path.join(current_dir, "SPIS")  # 新文件夹的路径

        dir = QDir(dirName)
        if not dir.exists():
            dir.mkdir(dirName)

        if self.pushButton_create.text() == "开始":
            # self.cmdList.clear()
            # self.cmdStr_en = ""
            self.tableWidget_panel.clear()

            for i in range(self.tableWidget_panel.rowCount() - 1, -1, -1):
                self.tableWidget_panel.removeRow(i)

            dirName = os.path.join(
                current_dir,
                "SPIS",
                QDateTime.currentDateTime().toString("yyyy-MM-dd"),
            )
            dir = QDir(dirName)
            if not dir.exists():
                dir.mkdir(dirName)

            if dir.exists():
                msg_box = QMessageBox(QMessageBox.Information, "创建Excel", "创建成功")
                msg_box.exec_()
                self.pushButton_create.setText("完成")
            else:
                msg_box = QMessageBox(QMessageBox.Information, "创建Excel", "创建失败")
                msg_box.exec_()
        else:
            dialog = QDialog(self)
            form = QFormLayout(dialog)
            dialog.setWindowTitle("文件名称")
            form.addRow(QLabel("请输入文件名称:"))
            fileName = QLineEdit(dialog)
            form.addRow(fileName)

            buttonBox = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog
            )
            form.addRow(buttonBox)
            buttonBox.accepted.connect(dialog.accept)
            buttonBox.rejected.connect(dialog.reject)

            while dialog.exec():
                if fileName.text() == "":
                    QMessageBox.critical(None, "错误", "请输入文件名称！")
                    if dialog.result() == QDialog.Rejected:
                        break
                if dialog.result() == QDialog.Accepted:
                    filePath = os.path.join(
                        current_dir,
                        "SPIS",
                        QDateTime.currentDateTime().toString("yyyy-MM-dd"),
                        fileName.text() + ".xlsx",
                    )
                    f = QFile(filePath)
                    if f.exists():
                        result = QMessageBox.question(
                            None,
                            "文件已存在",
                            "该文件已存在，是否覆盖？",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if result == QMessageBox.Yes:
                            # allCom = " ".join(str(cmd) for cmd in self.cmdList)
                            curRow = self.tableWidget_panel.rowCount()
                            self.tableWidget_panel.insertRow(curRow)
                            name = QTableWidgetItem()
                            cmds = QTableWidgetItem()

                            name.setText("全部操作代号")
                            # cmds.setText(allCom)

                            self.tableWidget_panel.setItem(curRow, 0, name)
                            self.tableWidget_panel.setItem(curRow, 1, cmds)

                            self.table_output(filePath)
                            self.pushButton_create.setText("开始")
                        else:
                            break
                    elif fileName.text() != "":
                        # allCom = " ".join(str(cmd) for cmd in self.cmdList)
                        curRow = self.tableWidget_panel.rowCount()
                        self.tableWidget_panel.insertRow(curRow)
                        name = QTableWidgetItem()
                        cmds = QTableWidgetItem()

                        name.setText("全部操作代号")
                        # cmds.setText(allCom)

                        self.tableWidget_panel.setItem(curRow, 0, name)
                        self.tableWidget_panel.setItem(curRow, 1, cmds)
                        self.table_output(filePath)
                        self.pushButton_create.setText("开始")

                        break
    #脚本录制表格输出
    def table_output(self, file_name):
        if file_name != "":
            data = []
            columns = ["名称", "操作"]

            for j in range(self.tableWidget_panel.rowCount()):
                row_data = []
                for k in range(self.tableWidget_panel.columnCount()):
                    item = self.tableWidget_panel.item(j, k)
                    if item is None or not item.text():
                        value = "null"
                    else:
                        value = item.text()
                    row_data.append(value)
                data.append(row_data)

            df = pd.DataFrame(data, columns=columns)
            df.to_excel(file_name, index=False)

if __name__ == '__main__':
    # 适应高DPI设备
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    # 适应Windows缩放
    QtGui.QGuiApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    mywin = MainWindow()
    mywin.show()
    sys.exit(app.exec_())
