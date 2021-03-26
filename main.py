import sys

from PyQt5 import QtWidgets

# Press the green button in the gutter to run the script.
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from MainWindow import MainWindow
from interfaces.InferenceInterface import InferenceInterface

from PyQt5.QtCore import Qt, QFile, QTextStream

app = None

if __name__ == '__main__':
    # Define the sensor interfaces

    # load the qt application
    app = QtWidgets.QApplication(sys.argv)

    # splash screen
    splash = QLabel()
    pixmap = QPixmap('media/logo/RN.png')
    # pixmap = pixmap.scaled(640, 640)
    splash.setPixmap(pixmap)
    splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    splash.show()

    # main window init

    inference_interface = InferenceInterface()
    window = MainWindow(inference_interface=inference_interface)

    # stylesheet init

    stylesheet = QFile('ui/stylesheet/dark.qss')
    stylesheet.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(stylesheet)
    app.setStyleSheet(stream.readAll())

    # splash screen destroy
    splash.destroy()

    window.show()
    app.exec_()
    print('Resuming Console Interaction.')
