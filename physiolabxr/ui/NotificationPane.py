from PyQt6 import QtWidgets, QtCore, uic
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs import config
from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.ui.NotificationPaneItem import NotificationPaneItem


class NotificationPane(QWidget):
    """
    other UI controllers may emit to GlobalSignals.show_notification_signal to show notifications
    """
    def __init__(self, parent=None):
        super(NotificationPane, self).__init__(parent)

        self.ui = uic.loadUi(AppConfigs()._ui_NotificationPane, self)
        self.move(self.width() + 10, 10)
        self.show()

        self.notification_items = []
        GlobalSignals().show_notification_signal.connect(self.show_notification)

        self.hide()

    def show_notification(self, message):
        new_item = NotificationPaneItem(self, message)
        new_item.close_button.clicked.connect(lambda: self.item_closed(new_item))
        self.notification_items.append(new_item)
        self.notification_scroll_vlayout.addWidget(new_item)
        self.notification_scroll_vlayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)
        self.adjust_size()

        self.show()

    def item_closed(self, item):
        self.notification_items.remove(item)
        self.adjust_size()
        if len(self.notification_items) == 0:
            self.hide()

    def adjust_size(self):
        new_height = sum([x.height() for x in self.notification_items]) + 10
        new_height = min(new_height, int(self.parent().height()/2))
        self.resize(self.width(), new_height)
        self.parent().adjust_notification_panel_location()
