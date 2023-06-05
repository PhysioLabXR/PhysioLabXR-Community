from PyQt5.QtWidgets import QWidget
# This Python file uses the following encoding: utf-8
import time
from collections import deque

import numpy as np
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import QTimer, QThread, QMutex, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialogButtonBox, QSplitter

from exceptions.exceptions import ChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from rena import config, config_ui
from rena.configs.configs import AppConfigs, LinechartVizMode
from rena.presets.load_user_preset import create_default_group_entry
from rena.presets.presets_utils import get_stream_preset_info, set_stream_preset_info, get_stream_group_info, \
    get_is_group_shown, pop_group_from_stream_preset, add_group_entry_to_stream, change_stream_group_order, \
    change_stream_group_name, pop_stream_preset_from_settings, change_group_channels
from rena.sub_process.TCPInterface import RenaTCPAddDSPWorkerRequestObject, RenaTCPInterface
from rena.threadings import workers
from rena.ui.GroupPlotWidget import GroupPlotWidget
from rena.ui.PoppableWidget import Poppable
from rena.ui.StreamOptionsWindow import StreamOptionsWindow
from rena.ui.VizComponents import VizComponents
from rena.ui.device_ui.CustomePropertyWidget import CustomPropertyWidget
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, \
    options_icon
from rena.utils.buffers import DataBufferSingleStream
from rena.utils.performance_utils import timeit
from rena.utils.ui_utils import dialog_popup, clear_widget
from rena.presets.presets_utils import get_stream_device_preset, dataclass_to_dict
from dataclasses import fields


class DeviceOptionsWindow(QWidget):
    def __init__(self, stream_name, parent_widget):
        super().__init__()
        self.ui = uic.loadUi("ui/device_ui/DeviceOptionsWindow.ui", self)
        self.stream_name = stream_name
        self.parent_widget = parent_widget
        self.device_options_entry = get_stream_device_preset(stream_name=stream_name)
        self.device_options_field = {}
        # self.device_options_valid = False
        self.add_custom_device_property_uis()

    def add_custom_device_property_uis(self):
        device_preset = get_stream_device_preset(stream_name=self.stream_name)
        device_preset_dict = dataclass_to_dict(device_preset)

        # fixed attribute
        for property_name in device_preset_dict:
            self.device_options_field[property_name] = CustomPropertyWidget(parent=self,
                                                                             stream_name=self.stream_name,
                                                                             property_entry=self.device_options_entry,
                                                                             property_name=property_name,
                                                                             property_value=device_preset_dict[
                                                                                 property_name])

            self.DeviceOptionsWidgetVerticalLayout.insertWidget(2, self.device_options_field[property_name])

        # dynamic attribute

        # attributes = fields(device_preset)

        # attribute_dict = {attr.name: getattr(device_preset, attr.name) for attr in device_preset}

        # attribute_dict = {attr.name: getattr(a, attr.name) for attr in attributes}

        # attribute_dict = {attr.name: getattr(person, attr.name) for attr in attributes}

        # loop private properties
