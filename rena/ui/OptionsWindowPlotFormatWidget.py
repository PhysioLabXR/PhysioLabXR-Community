# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QDialog, QTreeWidget, QLabel, QTreeWidgetItem

from rena import config_signal, config
from rena.config_ui import *
from rena.ui.StreamGroupView import StreamGroupView
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup, init_label, init_button, init_scroll_label
from PyQt5 import QtCore, QtGui, QtWidgets


class OptionsWindowPlotFormatWidget(QtWidgets.QWidget):

    plot_format_on_change = QtCore.pyqtSignal(str)

    def __init__(self, stream_name):
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        # self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindowPlotFormatWidget.ui", self)

    #[time_series, image_rgb_format, bar_plot, ]

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())


    def set_plot_format(self):
        # get plot format
        # checkbox
        # actions after plot format changed
        pass






