from datetime import datetime

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QLabel


class ScriptConsoleLog(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptConsoleLog.ui", self)

        # start the play pause button

    def print_msg(self, msg):
        msg_label = QLabel(msg)
        now = datetime.now()  # current date and time
        timestamp_label = QLabel(now.strftime("%m/%d/%Y, %H:%M:%S"))

        msg_label.adjustSize()
        timestamp_label.adjustSize()

        self.LogContentLayout.addRow(timestamp_label, msg_label)
