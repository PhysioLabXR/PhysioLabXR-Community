from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLayout, QPushButton

from rena.ui_shared import pop_window_icon, dock_window_icon
from rena.utils.ui_utils import AnotherWindow


class Poppable(QtWidgets.QWidget):
    def __init__(self, window_title, parent_widget: QtWidgets, parent_layout: QLayout, close_function: callable):
        super().__init__()
        self.pop_button = None
        self.parent_layout = parent_layout
        self.parent_widget = parent_widget
        self.window_title = window_title
        self.close_function = close_function

        self.is_popped = False

    def set_pop_button(self, pop_button: QPushButton):
        self.pop_button = pop_button

        self.pop_button.clicked.connect(self.pop_window)
        self.set_button_icons()

    def dock_window(self):
        assert self.pop_button is not None, "PoppableWidget must have a pop_button set before calling dock_window"
        self.parent_layout.insertWidget(self.parent_layout.count() - 1, self)
        self.another_window.hide()
        self.another_window.deleteLater()

        self.parent_widget.activateWindow()

        self.pop_button.clicked.disconnect()
        self.pop_button.clicked.connect(self.pop_window)

        self.is_popped = False
        self.set_button_icons()

    def pop_window(self):
        assert self.pop_button is not None, "PoppableWidget must have a pop_button set before calling pop_window"
        self.another_window = AnotherWindow(self, self.close_function)
        self.another_window.setWindowTitle(self.window_title)
        self.another_window.show()
        self.another_window.activateWindow()

        self.pop_button.clicked.disconnect()
        self.pop_button.clicked.connect(self.dock_window)

        self.is_popped = True
        self.set_button_icons()

    def set_button_icons(self):
        if not self.is_popped:
            self.pop_button.setIcon(pop_window_icon)
        else:
            self.pop_button.setIcon(dock_window_icon)

    def delete_window(self):
        self.another_window.deleteLater()
