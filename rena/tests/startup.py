import os
import sys

from PyQt5 import QtCore, QtWidgets

from rena.MainWindow import MainWindow
from rena.startup import load_settings


def app(qtbot):
    print('Initializing test fixature for ' + 'Visualization Features')
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)

    # app initialization
    load_settings()
    test_renalab_app = MainWindow(app=app)

    return test_renalab_app


def test_starts_and_closes(app, qtbot):
    qtbot.mouseClick(app.add_lslStream_btn, QtCore.Qt.LeftButton)
