# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'f:\资料\东北大学\学习信息\9大四下\211125毕设\code\motionDetection\utils\gui\MotionDetectionSystem.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MotionDetectionSystemClass(object):
    def setupUi(self, MotionDetectionSystemClass):
        MotionDetectionSystemClass.setObjectName("MotionDetectionSystemClass")
        MotionDetectionSystemClass.resize(1034, 570)
        self.centralWidget = QtWidgets.QWidget(MotionDetectionSystemClass)
        self.centralWidget.setObjectName("centralWidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_2.setGeometry(QtCore.QRect(630, 10, 391, 171))
        self.groupBox_2.setObjectName("groupBox_2")
        self.InputArea = QtWidgets.QLabel(self.groupBox_2)
        self.InputArea.setGeometry(QtCore.QRect(10, 20, 211, 141))
        self.InputArea.setText("")
        self.InputArea.setObjectName("InputArea")
        self.pushButton_loadvideo = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_loadvideo.setGeometry(QtCore.QRect(230, 10, 151, 31))
        self.pushButton_loadvideo.setObjectName("pushButton_loadvideo")
        self.pushButton_close = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_close.setGeometry(QtCore.QRect(230, 50, 151, 31))
        self.pushButton_close.setObjectName("pushButton_close")
        self.pushButton_analyzeOffline = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_analyzeOffline.setGeometry(QtCore.QRect(230, 130, 151, 31))
        self.pushButton_analyzeOffline.setObjectName("pushButton_analyzeOffline")
        self.groupBox = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox.setGeometry(QtCore.QRect(630, 360, 391, 151))
        self.groupBox.setObjectName("groupBox")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(20, 40, 121, 16))
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 60, 161, 16))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setGeometry(QtCore.QRect(20, 80, 131, 16))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_4.setGeometry(QtCore.QRect(20, 100, 161, 16))
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_5.setGeometry(QtCore.QRect(20, 120, 131, 16))
        self.radioButton_5.setObjectName("radioButton_5")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(20, 20, 54, 12))
        self.label_4.setObjectName("label_4")
        self.pushButton_Parameters = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_Parameters.setGeometry(QtCore.QRect(270, 110, 111, 31))
        self.pushButton_Parameters.setObjectName("pushButton_Parameters")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 611, 501))
        self.groupBox_3.setObjectName("groupBox_3")
        self.OutputArea = QtWidgets.QLabel(self.groupBox_3)
        self.OutputArea.setGeometry(QtCore.QRect(10, 20, 591, 441))
        self.OutputArea.setText("")
        self.OutputArea.setObjectName("OutputArea")
        self.slider_timestamp = QtWidgets.QSlider(self.groupBox_3)
        self.slider_timestamp.setGeometry(QtCore.QRect(60, 470, 431, 20))
        self.slider_timestamp.setOrientation(QtCore.Qt.Horizontal)
        self.slider_timestamp.setObjectName("slider_timestamp")
        self.label_timestamp = QtWidgets.QLabel(self.groupBox_3)
        self.label_timestamp.setGeometry(QtCore.QRect(500, 467, 111, 21))
        self.label_timestamp.setObjectName("label_timestamp")
        self.pushButton_playpause = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_playpause.setGeometry(QtCore.QRect(10, 463, 41, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(14)
        self.pushButton_playpause.setFont(font)
        self.pushButton_playpause.setObjectName("pushButton_playpause")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        self.textBrowser.setGeometry(QtCore.QRect(630, 190, 391, 151))
        self.textBrowser.setObjectName("textBrowser")
        MotionDetectionSystemClass.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MotionDetectionSystemClass)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1034, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu_Video = QtWidgets.QMenu(self.menuBar)
        self.menu_Video.setObjectName("menu_Video")
        self.menu_Algorithm = QtWidgets.QMenu(self.menuBar)
        self.menu_Algorithm.setObjectName("menu_Algorithm")
        self.menuFilter = QtWidgets.QMenu(self.menu_Algorithm)
        self.menuFilter.setObjectName("menuFilter")
        self.menuEqualization = QtWidgets.QMenu(self.menu_Algorithm)
        self.menuEqualization.setObjectName("menuEqualization")
        self.menuOutput = QtWidgets.QMenu(self.menuBar)
        self.menuOutput.setObjectName("menuOutput")
        self.menuFrame_Align = QtWidgets.QMenu(self.menuOutput)
        self.menuFrame_Align.setObjectName("menuFrame_Align")
        self.menu_about = QtWidgets.QMenu(self.menuBar)
        self.menu_about.setObjectName("menu_about")
        self.menu_About = QtWidgets.QMenu(self.menuBar)
        self.menu_About.setObjectName("menu_About")
        MotionDetectionSystemClass.setMenuBar(self.menuBar)
        self.statusBar = QtWidgets.QStatusBar(MotionDetectionSystemClass)
        self.statusBar.setObjectName("statusBar")
        MotionDetectionSystemClass.setStatusBar(self.statusBar)
        self.actionLoad_Video = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionLoad_Video.setObjectName("actionLoad_Video")
        self.actionClose_Video = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionClose_Video.setObjectName("actionClose_Video")
        self.actionNone = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionNone.setObjectName("actionNone")
        self.actionGrayscale = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionGrayscale.setObjectName("actionGrayscale")
        self.actionGrayscale_2 = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionGrayscale_2.setObjectName("actionGrayscale_2")
        self.actionGaussian_Filter = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionGaussian_Filter.setObjectName("actionGaussian_Filter")
        self.actionMean_Filter = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionMean_Filter.setObjectName("actionMean_Filter")
        self.actionMedian_fILTER = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionMedian_fILTER.setObjectName("actionMedian_fILTER")
        self.actionHistogram_Equalization = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionCLAHE = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionCLAHE.setObjectName("actionCLAHE")
        self.actionRANSAC_ORB = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionRANSAC_ORB.setObjectName("actionRANSAC_ORB")
        self.actionHomographynet = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionHomographynet.setObjectName("actionHomographynet")
        self.actionRSHomoNet = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionRSHomoNet.setObjectName("actionRSHomoNet")
        self.actionSwitcher = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionSwitcher.setObjectName("actionSwitcher")
        self.actionAbout_this_app = QtWidgets.QAction(MotionDetectionSystemClass)
        self.actionAbout_this_app.setObjectName("actionAbout_this_app")
        self.menu_Video.addAction(self.actionLoad_Video)
        self.menu_Video.addAction(self.actionClose_Video)
        self.menuFilter.addAction(self.actionGaussian_Filter)
        self.menuFilter.addAction(self.actionMean_Filter)
        self.menuFilter.addAction(self.actionMedian_fILTER)
        self.menuEqualization.addAction(self.actionHistogram_Equalization)
        self.menuEqualization.addAction(self.actionCLAHE)
        self.menu_Algorithm.addAction(self.actionNone)
        self.menu_Algorithm.addSeparator()
        self.menu_Algorithm.addAction(self.actionGrayscale_2)
        self.menu_Algorithm.addAction(self.menuFilter.menuAction())
        self.menu_Algorithm.addAction(self.menuEqualization.menuAction())
        self.menuFrame_Align.addAction(self.actionRANSAC_ORB)
        self.menuFrame_Align.addAction(self.actionHomographynet)
        self.menuFrame_Align.addAction(self.actionRSHomoNet)
        self.menuFrame_Align.addAction(self.actionSwitcher)
        self.menuOutput.addAction(self.menuFrame_Align.menuAction())
        self.menu_About.addAction(self.actionAbout_this_app)
        self.menuBar.addAction(self.menu_Video.menuAction())
        self.menuBar.addAction(self.menu_Algorithm.menuAction())
        self.menuBar.addAction(self.menuOutput.menuAction())
        self.menuBar.addAction(self.menu_about.menuAction())
        self.menuBar.addAction(self.menu_About.menuAction())

        self.retranslateUi(MotionDetectionSystemClass)
        self.pushButton_loadvideo.clicked.connect(MotionDetectionSystemClass.loadVideo)
        self.slider_timestamp.sliderMoved['int'].connect(MotionDetectionSystemClass.changeVideoProgress)
        self.pushButton_playpause.clicked.connect(MotionDetectionSystemClass.pushButtonPlayPause)
        self.pushButton_close.clicked.connect(MotionDetectionSystemClass.closeVideo)
        self.pushButton_analyzeOffline.clicked.connect(MotionDetectionSystemClass.analyseOffline)
        self.radioButton.clicked.connect(MotionDetectionSystemClass.updateOutPutType)
        self.radioButton_2.clicked.connect(MotionDetectionSystemClass.updateOutPutType)
        self.radioButton_3.clicked.connect(MotionDetectionSystemClass.updateOutPutType)
        self.radioButton_4.clicked.connect(MotionDetectionSystemClass.updateOutPutType)
        self.radioButton_5.clicked.connect(MotionDetectionSystemClass.updateOutPutType)
        QtCore.QMetaObject.connectSlotsByName(MotionDetectionSystemClass)

    def retranslateUi(self, MotionDetectionSystemClass):
        _translate = QtCore.QCoreApplication.translate
        MotionDetectionSystemClass.setWindowTitle(_translate("MotionDetectionSystemClass", "Motion Detection System"))
        self.groupBox_2.setTitle(_translate("MotionDetectionSystemClass", "Input Channel"))
        self.pushButton_loadvideo.setText(_translate("MotionDetectionSystemClass", "Load Video..."))
        self.pushButton_close.setText(_translate("MotionDetectionSystemClass", "Close Video"))
        self.pushButton_analyzeOffline.setText(_translate("MotionDetectionSystemClass", "Analyze offline"))
        self.groupBox.setTitle(_translate("MotionDetectionSystemClass", "Control Panal"))
        self.radioButton.setText(_translate("MotionDetectionSystemClass", "Original Video"))
        self.radioButton_2.setText(_translate("MotionDetectionSystemClass", "Frame Difference"))
        self.radioButton_3.setText(_translate("MotionDetectionSystemClass", "Moving Mask"))
        self.radioButton_4.setText(_translate("MotionDetectionSystemClass", "Motion Detection"))
        self.radioButton_5.setText(_translate("MotionDetectionSystemClass", "Motion Prediction"))
        self.label_4.setText(_translate("MotionDetectionSystemClass", "Mode："))
        self.pushButton_Parameters.setText(_translate("MotionDetectionSystemClass", "Parameters..."))
        self.groupBox_3.setTitle(_translate("MotionDetectionSystemClass", "Output Channel"))
        self.label_timestamp.setText(_translate("MotionDetectionSystemClass", "00:00:00/00:00:00"))
        self.pushButton_playpause.setText(_translate("MotionDetectionSystemClass", "▶"))
        self.menu_Video.setTitle(_translate("MotionDetectionSystemClass", "&Video"))
        self.menu_Algorithm.setTitle(_translate("MotionDetectionSystemClass", "&Preprocess"))
        self.menuFilter.setTitle(_translate("MotionDetectionSystemClass", "Filter"))
        self.menuEqualization.setTitle(_translate("MotionDetectionSystemClass", "Equalization"))
        self.menuOutput.setTitle(_translate("MotionDetectionSystemClass", "&Detection"))
        self.menuFrame_Align.setTitle(_translate("MotionDetectionSystemClass", "Frame Align"))
        self.menu_about.setTitle(_translate("MotionDetectionSystemClass", "&Export"))
        self.menu_About.setTitle(_translate("MotionDetectionSystemClass", "&About"))
        self.actionLoad_Video.setText(_translate("MotionDetectionSystemClass", "Load Video..."))
        self.actionClose_Video.setText(_translate("MotionDetectionSystemClass", "Close Video"))
        self.actionNone.setText(_translate("MotionDetectionSystemClass", "Clear All"))
        self.actionGrayscale.setText(_translate("MotionDetectionSystemClass", "Grayscale"))
        self.actionGrayscale_2.setText(_translate("MotionDetectionSystemClass", "Grayscale"))
        self.actionGaussian_Filter.setText(_translate("MotionDetectionSystemClass", "Gaussian Filter..."))
        self.actionMean_Filter.setText(_translate("MotionDetectionSystemClass", "Mean Filter..."))
        self.actionMedian_fILTER.setText(_translate("MotionDetectionSystemClass", "Median fILTER..."))
        self.actionHistogram_Equalization.setText(_translate("MotionDetectionSystemClass", "Histogram Equalization"))
        self.actionCLAHE.setText(_translate("MotionDetectionSystemClass", "CLAHE"))
        self.actionRANSAC_ORB.setText(_translate("MotionDetectionSystemClass", "RANSAC+ORB"))
        self.actionHomographynet.setText(_translate("MotionDetectionSystemClass", "Homographynet"))
        self.actionRSHomoNet.setText(_translate("MotionDetectionSystemClass", "RSHomoNet"))
        self.actionSwitcher.setText(_translate("MotionDetectionSystemClass", "Switcher"))
        self.actionAbout_this_app.setText(_translate("MotionDetectionSystemClass", "About this app..."))

