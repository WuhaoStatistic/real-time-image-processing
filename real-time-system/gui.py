# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1649, 950)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setGeometry(QtCore.QRect(180, 840, 131, 51))
        self.open.setObjectName("open")
        self.end = QtWidgets.QPushButton(self.centralwidget)
        self.end.setGeometry(QtCore.QRect(580, 840, 131, 51))
        self.end.setObjectName("end")
        self.original = QtWidgets.QLabel(self.centralwidget)
        self.original.setGeometry(QtCore.QRect(20, 30, 769, 769))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.original.setFont(font)
        self.original.setObjectName("original")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(410, 840, 111, 51))
        self.start.setObjectName("start")
        self.process = QtWidgets.QLabel(self.centralwidget)
        self.process.setGeometry(QtCore.QRect(820, 30, 769, 769))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.process.setFont(font)
        self.process.setObjectName("process")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1649, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.open.setText(_translate("MainWindow", "Open Carema"))
        self.end.setText(_translate("MainWindow", "End"))
        self.original.setText(_translate("MainWindow", "carema"))
        self.start.setText(_translate("MainWindow", "Start"))
        self.process.setText(_translate("MainWindow", "generalized"))
