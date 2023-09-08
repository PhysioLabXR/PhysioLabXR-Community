from PyQt6 import QtWidgets
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QLayout, QPushButton

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.utils.ui_utils import AnotherWindow


class Poppable(QtWidgets.QWidget):
    def __init__(self, window_title, parent_widget: QtWidgets, parent_layout: QLayout, close_function: callable):
        super().__init__()
        self.pop_button = None
        self.parent_layout = parent_layout
        self.parent_widget = parent_widget
        self.window_title = window_title
        self.close_function = close_function

        self.is_popped = False
        self.window_icon = QIcon(AppConfigs()._app_logo)

    def set_pop_button(self, pop_button: QPushButton):
        self.pop_button = pop_button
        self.pop_button.clicked.connect(self.pop_window)
        self.set_pop_icons()

    def dock_window(self):
        assert self.pop_button is not None, "PoppableWidget must have a pop_button set before calling dock_window"
        self.parent_layout.insertWidget(self.parent_layout.count() - 1, self)
        self.another_window.hide()
        self.another_window.deleteLater()

        self.parent_widget.activateWindow()

        self.pop_button.clicked.disconnect()
        self.pop_button.clicked.connect(self.pop_window)

        self.is_popped = False
        self.set_pop_icons()

    def pop_window(self):
        assert self.pop_button is not None, "PoppableWidget must have a pop_button set before calling pop_window"
        self.another_window = AnotherWindow(self, self.close_function)
        self.another_window.setWindowIcon(self.window_icon)
        self.another_window.setWindowTitle(self.window_title)
        self.another_window.show()
        self.another_window.activateWindow()

        self.pop_button.clicked.disconnect()
        self.pop_button.clicked.connect(self.dock_window)

        self.is_popped = True
        self.set_pop_icons()

    def set_pop_icons(self):
        assert self.pop_button is not None, "PoppableWidget must have a pop_button set before calling pop_window"
        if not self.is_popped:
            self.pop_button.setIcon(AppConfigs()._icon_pop_window)
        else:
            self.pop_button.setIcon(AppConfigs()._icon_dock_window)

    def delete_window(self):
        self.another_window.deleteLater()
