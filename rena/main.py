import sys

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtWidgets import QLabel, QSystemTrayIcon, QMenu

from rena.config import app_logo_path
from rena.configs.configs import AppConfigs

AppConfigs(_reset=False)  # create the singleton app configs object
from MainWindow import MainWindow
from rena.startup import load_settings

app = None

if __name__ == '__main__':

    # load the qt application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    tray_icon = QSystemTrayIcon(QIcon(app_logo_path), parent=app)
    tray_icon.setToolTip('RenaLabApp')
    tray_icon.show()

    # splash screen
    splash = QLabel()
    pixmap = QPixmap('../media/logo/RenaLabAppDeprecated.png')
    splash.setPixmap(pixmap)
    splash.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
    splash.show()

    # load default settings
    load_settings(revert_to_default=False, reload_presets=True)

    # main window init
    window = MainWindow(app=app)

    window.setWindowIcon(QIcon(app_logo_path))
    # make tray menu
    menu = QMenu()
    exit_action = menu.addAction('Exit')
    exit_action.triggered.connect(window.close)

    # splash screen destroy
    splash.destroy()
    window.show()

    try:
        app.exec()
        print('App closed by user')
        sys.exit()
    except KeyboardInterrupt:
        print('App terminate by KeyboardInterrupt')
        sys.exit()
