from PyQt6 import QtWidgets, QtCore, uic
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs


class NotificationPaneItem(QWidget):
    """
    other UI controllers may use signals to show notifications
    """
    def __init__(self, parent, message_dict):
        super(NotificationPaneItem, self).__init__(parent)
        self.ui = uic.loadUi(AppConfigs()._ui_NotificationPaneItem, self)

        self.title_label.setText(message_dict['title'])
        self.body_label.setText(message_dict['body'])

        self.close_button.clicked.connect(self.close)
