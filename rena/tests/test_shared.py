import threading

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox


def click_message_box_ok(qtbot, delay=1000):
    def handle_dialog():
        w = QtWidgets.QApplication.activeWindow()
        if isinstance(w, QMessageBox):
            yes_button = w.button(QtWidgets.QMessageBox.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=delay)  # delay 1 second for the data to come in

    threading.Timer(1, handle_dialog).start()