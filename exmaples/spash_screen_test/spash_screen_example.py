import sys
from PyQt5 import QtCore, QtGui, QtWidgets     # + QtWidgets
import time

import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore    import QTimer, Qt

if __name__ == '__main__':
    app = QApplication(sys.argv)

    label = QLabel("""
            <font color=red size=128>
               <b>Hello PyQt， The window will disappear after 5 seconds！</b>
            </font>""")

    # SplashScreen - Indicates that the window is a splash screen. This is the default type for .QSplashScreen
    # FramelessWindowHint - Creates a borderless window. The user cannot move or resize the borderless window through the window system.
    label.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    label.show()

    # Automatically exit after  5 seconds
    time.sleep(5)
    print("John")
    label.close()

    app.exec_()