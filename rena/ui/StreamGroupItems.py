## Reference:
## https://stackoverflow.com/questions/13662020/how-to-implement-itemchecked-and-itemunchecked-signals-for-qtreewidget-in-pyqt4
import sys
from collections import deque

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from rena import config
from rena.config_ui import *
from rena.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from rena.ui_shared import CHANNEL_ITEM_IS_DISPLAY_CHANGED
from rena.utils.settings_utils import get_stream_preset_info, is_group_shown
from rena.utils.ui_utils import dialog_popup
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

class GroupItem(QTreeWidgetItem):
    item_type = 'group'

    def __init__(self, parent, is_shown, plot_format, stream_name, group_name):
        super().__init__(parent)
        self.is_shown = is_shown  # show the channel plot or not
        self.plot_format = plot_format
        self.stream_name = stream_name
        self.group_name = group_name
        self.setText(0, group_name)

        # self.OptionsWindowPlotFormatWidget = OptionsWindowPlotFormatWidget(self.stream_name, self.group_name)



    def setData(self, column, role, value):
        check_state_before = self.checkState(column)
        super(GroupItem, self).setData(column, role, value)
        check_state_after = self.checkState(column)

        if check_state_before != check_state_after:
            if check_state_after == Qt.Checked or check_state_after == Qt.PartiallyChecked:
                self.display=True
                self.setForeground(0, QBrush(QColor(color_green)))
            else:
                self.display=False
                self.setForeground(0, QBrush(QColor(color_white)))


class ChannelItem(QTreeWidgetItem):
    def __init__(self, parent, is_shown, lsl_index, channel_name):
        super().__init__(parent)
        self.is_shown = is_shown  # show the channel plot or not
        self.lsl_index = lsl_index
        self.most_recent_change = None
        self.channel_name = channel_name
        self.setText(0, channel_name)
        self.setText(1, '['+str(lsl_index)+']')


    def setData(self, column, role, value):
        parent_check_state_before = self.parent().checkState(column)
        item_check_state_before = self.checkState(column)

        super(ChannelItem, self).setData(column, role, value)
        item_check_state_after = self.checkState(column)
        parent_check_state_after = self.parent().checkState(column)


        if role == Qt.EditRole:
            pass
            # editing the name


        if role == Qt.CheckStateRole and item_check_state_before != item_check_state_after:
            # set text to green
            if item_check_state_after == Qt.Checked or item_check_state_after == Qt.PartiallyChecked:
                self.display = True
                self.setForeground(0, QBrush(QColor(color_green)))
            else:
                self.display = False
                self.setForeground(0, QBrush(QColor(color_white)))

            if parent_check_state_after != parent_check_state_before:
                if parent_check_state_after == Qt.Checked or parent_check_state_after == Qt.PartiallyChecked:
                    self.parent().display = True
                    self.parent().setForeground(0, QBrush(QColor(color_green)))
                else:
                    self.parent().display = False
                    self.parent().setForeground(0, QBrush(QColor(color_white)))
