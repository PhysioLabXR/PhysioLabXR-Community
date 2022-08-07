from datetime import datetime

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QLabel, QHBoxLayout

from rena.config import CONSOLE_LOG_MAX_NUM_ROWS


class ScriptConsoleLog(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptConsoleLog.ui", self)
        self.log_history = ''
        self.msg_labels = []

        # start the play pause button

    def print_msg(self, msg):
        msg_label = QLabel(msg)
        now = datetime.now()  # current date and time
        timestamp_string = now.strftime("%m/%d/%Y, %H:%M:%S")
        timestamp_label = QLabel(timestamp_string)

        msg_label.adjustSize()
        timestamp_label.adjustSize()

        self.log_history += timestamp_string + msg + '\n'

        self.LogContentLayout.addRow(timestamp_label, msg_label)
        self.msg_labels.append(msg_label)

        if len(self.msg_labels) > CONSOLE_LOG_MAX_NUM_ROWS:
            self.LogContentLayout.removeRow(self.msg_labels[0])
            self.msg_labels = self.msg_labels[1:]
