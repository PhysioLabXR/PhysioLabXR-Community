import os
import subprocess
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QLabel, QMenu

from MainWindow import MainWindow

app = None

# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    # load the qt application
    app = QtWidgets.QApplication(sys.argv)

    # splash screen
    splash = QLabel()
    pixmap = QPixmap('../media/logo/RN.png')
    # pixmap = pixmap.scaled(640, 640)
    splash.setPixmap(pixmap)
    splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    splash.show()

    # start server
    env = os.environ.copy()
    p = subprocess.Popen(['python', 'MainServer.py'], env=env)

    # main window init
    visualizer_window = MainWindow(app=app)
    visualizer_window.setWindowIcon(QIcon('../media/logo/RN.png'))

    # make tray menu
    menu = QMenu()
    exit_action = menu.addAction('Exit')
    exit_action.triggered.connect(visualizer_window.close)

    # stylesheet init

    stylesheet = QFile('ui/stylesheet/dark.qss')
    stylesheet.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(stylesheet)
    app.setStyleSheet(stream.readAll())
    # splash screen destroy
    splash.destroy()

    visualizer_window.show()
    app.exec_()
    print('Resuming Console Interaction.')

