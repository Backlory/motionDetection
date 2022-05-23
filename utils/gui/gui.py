import sys
import typing
import cv2
import numpy as np
from PyQt5.QtCore import Qt as Qt
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog, QWidget, QLabel, QMessageBox
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QPixmap, QImage
from algorithm.infer_all import Inference_all

from utils.gui.Ui_MotionDetectionSystem import Ui_MotionDetectionSystemClass


class mywindow(QMainWindow):
    def __init__(self, parent=None, flags =None) -> None:
        super().__init__()
        self.outPutType = 1
        self.algo = algorithm()
        #UI布局
        self.ui = Ui_MotionDetectionSystemClass()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowMaximizeButtonHint)
        self.setFixedSize(self.width(), self.height()) 
        self.ui.textBrowser.document().setMaximumBlockCount(8)

        #时间与显示
        self.timer = QTimer(self)
        self.current_time = QDateTime()
        self.timerCurrent = QTimer(self)
        self.timerCurrent.start(1000)
        self.label_currentTime = QLabel(self)
        self.ui.statusBar.addWidget(self.label_currentTime)


        #图像处理
        self.cap = cv2.VideoCapture()
        self.framelast = None
        self.frameNow = None
        self.frameNowProcessed = None
        
        #记录
        self.lastVideoPath = r"E:\dataset\dataset-fg-det"
        self.lastVideoPath_save = r"E:\dataset\dataset-fg-det"
        self.videolength = ""
        
        #连接
        self.timerCurrent.timeout.connect(self.timeCurrentUpdate)
        self.timer.timeout.connect(self.timerTicToc)
        # 其他链接，在ui中没链接上的
        self.ui.actionLoad_Video.triggered.connect(self.loadVideo)
        self.ui.actionClose_Video.triggered.connect(self.closeVideo)
        self.ui.actionAbout_this_app.triggered.connect(self.SLOTabout)
        
    def __del__(self):
        self.closeVideo()
        return
        
    def closeVideo(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.ui.slider_timestamp.setValue(0)
            self.ui.pushButton_playpause.setText("▶")
            self.ui.InputArea.setPixmap(QPixmap())
            self.ui.OutputArea.setPixmap(QPixmap())
        return

    def loadVideo(self):
        dyn = "*.gif *.mp4 *.avi"
        path, _ = QFileDialog.getOpenFileName(self, "Load Video...", self.lastVideoPath, dyn)
        if path:
            self.lastVideoPath = path
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if self.cap.isOpened():
                timeMSMax = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) * 1000 / self.cap.get(cv2.CAP_PROP_FPS)
                self.videolength = self.get_timestamp(timeMSMax)
                self.ui.slider_timestamp.setRange(0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.ui.slider_timestamp.setSingleStep(10)
                self.ui.slider_timestamp.setValue(0)
                self.ui.pushButton_playpause.setText("‖")
                
                temp = self.cap.get(cv2.CAP_PROP_FPS)
                self.timer.start(np.ceil(1000.0 / temp))
            else:
                self.cap.release()
                QMessageBox.warning(None, "Warning", "Open video failed!", QMessageBox.Ok)
        return



    def timerTicToc(self):
        assert(self.cap is not None)
        if self.framelast is None:
            _, self.framelast = self.cap.read()
        _, self.frameNow = self.cap.read()
        if self.frameNow is not None:
            #在输入区显示
            h,w,c = self.frameNow.shape
            qImageFrameInputDisplay = QImage(
                self.frameNow, w,h,QImage.Format_RGB888
            )
            qImageFrameInputDisplay = qImageFrameInputDisplay.scaled(self.ui.InputArea.size(),  
                Qt.IgnoreAspectRatio, 
                Qt.SmoothTransformation
            )
            self.ui.InputArea.setPixmap(QPixmap.fromImage(qImageFrameInputDisplay))
            
            #执行算法
            self.frameNowProcessed, states_algo = self.algo(self.framelast, self.frameNow)
            h,w,c = self.frameNowProcessed.shape

            #打印结果
            states = str(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + "-"
            states += states_algo
            self.ui.textBrowser.append(states)

            #在输出区显示
            qImageFrameOutputDisplay = QImage(
                self.frameNowProcessed, w, h, QImage.Format_RGB888
            )
            qImageFrameOutputDisplay = qImageFrameOutputDisplay.scaled(self.ui.OutputArea.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.OutputArea.setPixmap(QPixmap.fromImage(qImageFrameOutputDisplay))

            #时间戳、进度条控制
            timeMS = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.ui.slider_timestamp.setValue(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            timestamp = self.get_timestamp(timeMS) + "/" + self.videolength
            self.ui.label_timestamp.setText(timestamp)
        else:
            print("读取图像为空！")
            return
        self.framelast = np.array(self.frameNow, copy=True)
        return


    def get_timestamp(self, timeMS):
        '''/*将毫秒时间转化为文本时间戳00:00:00*/'''
        timeS = int(timeMS / 1000)
        timeM = 0
        timeH = 0
        timeMS = timeMS % 1000
        if (timeS >= 60):
            timeM = timeS // 60
            timeS = timeS % 60
        if (timeM >= 60):
            timeH = timeM // 60
            timeM = timeM % 60
        timestamp = str(timeH).zfill(2) + ":" + str(timeM).zfill(2) + ":" + str(timeS).zfill(2)
        return timestamp
    def updateOutPutType(self):
        if self.ui.radioButton.isChecked():
            self.outPutType = 1
        elif self.ui.radioButton_2.isChecked():
            self.outPutType = 2
        elif self.ui.radioButton_3.isChecked():
            self.outPutType = 3
        elif self.ui.radioButton_4.isChecked():
            self.outPutType = 4
        elif self.ui.radioButton_5.isChecked():
            self.outPutType = 5
        else:
            self.outPutType = 0
        self.algo.updateOutPutType(self.outPutType)
            
    def changeVideoProgress(self):
        if self.cap:
            val = self.ui.slider_timestamp.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    def pushButtonPlayPause(self):
        if self.timer.isActive():
            self.ui.pushButton_playpause.setText("▶")
            self.timer.stop()
        else:
            if self.cap.isOpened():
                self.ui.pushButton_playpause.setText("‖")
                temp = self.cap.get(cv2.CAP_PROP_FPS)
                self.timer.start(np.ceil(1000.0 / temp))
        return

    def timeCurrentUpdate(self):
        timestr = self.current_time.currentDateTime().toString("yyyy/MM/dd hh:mm:ss")
        self.label_currentTime.setText(timestr)

    def SLOTabout(self):
        temp = "This application is just for testing.\n Author: Backlory 04/13/2022"
        QMessageBox.warning(None, "About", temp, QMessageBox.Ok)


    def analyseOffline(self):
        #打开视频

        dyn = "*.gif *.mp4 *.avi"
        path, _ = QFileDialog.getOpenFileName(self, "Load Video...", self.lastVideoPath, dyn)
        if path is "":
            return
        self.lastVideoPath = path
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)

        #选取视频保存路径
        dyn = "*.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Export Video...", self.lastVideoPath_save, dyn)
        if path is "":
            return
        self.lastVideoPath_save = path
        FPS = self.cap.get(cv2.CAP_PROP_FPS)
        outputVideo=cv2.VideoWriter(
            path, cv2.VideoWriter.fourcc('M','P','4','V'), 
            FPS, (512, 512), True
        )
        if not outputVideo.isOpened():
            outputVideo.release()
            QMessageBox.warning(None, "Alert", "Can not create target file. Save video Fail!", QMessageBox.Ok)
            return


        '''
        *(self.cap) >> *(self.frameNow)
        int frame_idx = 0
        int frame_total_d10 = (int)self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / 10
        std.string states_algo
        QTime startTime = QTime.currentTime()

        while (!self.frameNow.empty()) {
            frame_idx++
            states_algo.clear()
            *self.frameNowProcessed = self.algo.testMat(*self.frameNow, states_algo)

            assert(self.frameNowProcessed.channels() == 3)
            outputVideo << *self.frameNowProcessed
            *(self.cap) >> *(self.frameNow)
        '''
        outputVideo.release()
        QMessageBox.warning(None, "About", "File saved!", QMessageBox.Ok)
        
        return

class algorithm:
    def __init__(self) -> None:
        self.infer = Inference_all()
        self.his_info = None
        self.outPutType = 0

    def __call__(self, img_t0, img_t1):
        
        diffOrigin, moving_mask, out, img_t0_enhancement, img_t0_arrow, \
            effect, alg_type, temp_rate_1, self.his_info = self.infer.step(
            img_t0, img_t1, his_info=self.his_info
            )
        if self.outPutType == 1:
            output = img_t0
        elif self.outPutType == 2:
            output = diffOrigin
        elif self.outPutType == 3:
            output = moving_mask
        elif self.outPutType == 4:
            output = img_t0_enhancement
        elif self.outPutType == 5:
            output = img_t0_arrow
        else:
            output = img_t0
        logs = f"{alg_type}={effect:.4f}, dn={temp_rate_1:.4f}"
        return output, logs
    
    def updateOutPutType(self, outPutType):
        self.outPutType = outPutType

def run():
    app = QApplication(sys.argv)
    MainWindow = mywindow()
    MainWindow.show()
    sys.exit(app.exec_())
