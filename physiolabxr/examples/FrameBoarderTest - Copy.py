import sys
import os

from PyQt6.QtCore import QFile, QTextStream
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame, QLabel
from PyQt6 import QtWidgets, QtCore, uic


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # frame = QFrame(self)
        # frame.setGeometry(50, 50, 200, 200)
        # frame.setFrameStyle(QFrame.Panel | QFrame.Plain)
        # frame.setLineWidth(1)
        # vbox = QVBoxLayout()
        # vbox.addWidget(frame)
        #
        stylesheet = QFile(r'D:\PycharmProjects\RenaLabApp\rena\ui\stylesheet\dark_frame_test.qss')
        stylesheet.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(stylesheet)
        QtWidgets.qApp.setStyleSheet(stream.readAll())
        #
        # label = QLabel('Test')
        # vbox = QVBoxLayout()
        # vbox.addWidget(label)
        # frame.setLayout(vbox)
        #
        # self.setLayout(vbox)
        #
        #
        # self.setGeometry(300, 300, 300, 300)

        self.ui = uic.loadUi(r"D:\PycharmProjects\RenaLabApp\rena\_ui\StreamContainer._ui", self)

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())