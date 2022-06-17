import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QLabel, QSystemTrayIcon, QMenu


class MainServer(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.app = app

if __name__ == '__main__':
    server_app = QtWidgets.QApplication(sys.argv)

    tray_icon = QSystemTrayIcon(QIcon('icon.PNG'), parent=server_app)
    tray_icon.setToolTip('ReNaServer')
    tray_icon.show()

    server = MainServer(server_app)

    server_app.exec_()
