from PyQt5 import QtGui
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QColor


#
# class VideoThread(QThread):
#
#     def __init__(self, camID=0):
#         super().__init__()
#         self.camID = camID
#         self.cap = None
#
#
#     change_pixmap_signal = pyqtSignal(np.ndarray)
#
#     def run(self):
#         # capture from web cam
#         self.cap = cv2.VideoCapture(self.camID)
#         while True:
#             ret, cv_img = self.cap.read()
#             if ret:
#                 self.change_pixmap_signal.emit(cv_img)
#
#     def quit(self):
#         self.cap.release()
#
#
# class WebcamInterface:
#     def __init__(self, camera_display = QGraphicsPixmapItem()):
#         self.camID = None
#         self.thread = None
#         self.camera_display = QGraphicsPixmapItem()

        #
        # self.setWindowTitle("Qt live label demo")
        # self.disply_width = 640
        # self.display_height = 480
        # # create the label that holds the image
        # self.image_label = QLabel(self)
        # self.image_label.resize(self.disply_width, self.display_height)
        # # create a text label
        # self.textLabel = QLabel('Webcam')
        #
        # # create a vertical box layout and add the two labels
        # vbox = QVBoxLayout()
        # vbox.addWidget(self.image_label)
        # vbox.addWidget(self.textLabel)
        # # set the vbox layout as the widgets layout
        # self.setLayout(vbox)
        #
        # # create the video capture thread
        # self.thread = VideoThread()
        # # connect its signal to the update_image slot
        # self.thread.change_pixmap_signal.connect(self.update_image)
        # # start the thread
        # self.thread.start()




    # def connect_sensor(self):
    #     self.thread = VideoThread(self.camID)
    #     self.thread.change_pixmap_signal.connect(self.update_image)
    #     self.thread.start()
    #
    # def stop_sensor(self):
    #     self.thread.quit()
    #
    #
    # @pyqtSlot(np.ndarray)
    # def update_image(self, cv_img):
    #     """Updates the image_label with a new opencv image"""
    #     qt_img = self.convert_cv_qt(cv_img)
    #     self.camera_display.setPixmap(qt_img)
    #
    # def convert_cv_qt(self, cv_img):
    #     """Convert from an opencv image to QPixmap"""
    #     rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb_image.shape
    #     bytes_per_line = ch * w
    #     convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #     p = convert_to_Qt_format.scaled(100, 100, Qt.KeepAspectRatio)
    #     return QPixmap.fromImage(p)
















# class App(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Qt static label demo")
#         width = 640
#         height = 480
#         # create the label that holds the image
#         self.image_label = QLabel(self)
#         # create a text label
#         self.textLabel = QLabel('Demo')
#
#         # create a vertical box layout and add the two labels
#         vbox = QVBoxLayout()
#         vbox.addWidget(self.image_label)
#         vbox.addWidget(self.textLabel)
#         # set the vbox layout as the widgets layout
#         self.setLayout(vbox)
#         # create a grey pixmap
#         grey = QPixmap(width, height)
#         grey.fill(QColor('darkGray'))
#         # set the image image to the grey pixmap
#         self.image_label.setPixmap(grey)


    # def convert_cv_qt(self, cv_img):
    #     """Convert from an opencv image to QPixmap"""
    #     rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #     h, w, ch = rgb_image.shape
    #     bytes_per_line = ch * w
    #     convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    #     p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
    #     return QPixmap.fromImage(p)

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(300, 300, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)



