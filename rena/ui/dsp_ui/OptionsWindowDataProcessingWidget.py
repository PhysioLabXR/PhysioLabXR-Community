import sys
# This Python file uses the following encoding: utf-8

from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from rena.presets.Cmap import Cmap
from rena.presets.PlotConfig import ImageFormat, ChannelFormat
from rena.presets.presets_utils import get_stream_group_info, get_stream_a_group_info, \
    set_stream_a_group_selected_plot_format, set_stream_a_group_selected_img_config, \
    set_bar_chart_max_min_range, set_group_image_format, set_group_image_channel_format, \
    get_group_image_config, set_spectrogram_time_per_segment, set_spectrogram_time_overlap, \
    get_spectrogram_time_per_segment, get_spectrogram_time_overlap, set_spectrogram_cmap, \
    set_spectrogram_percentile_level_min, set_spectrogram_percentile_level_max, get_group_data_processors
from rena.ui.SliderWithValueLabel import SliderWithValueLabel

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QLabel, QSystemTrayIcon, QMenu

from rena.config import app_logo_path
from rena.configs.configs import AppConfigs
from rena.ui.dsp_ui.RealtimeButterworthBandPassWidget import RealtimeButterworthBandPassWidget
from rena.utils.realtime_DSP import DataProcessorType
from enum import Enum


class data_processor_widget_type:
    def __init__(self, stream_name, group_name):
        self.RealtimeRealtimeButterworthBandPass = RealtimeButterworthBandPassWidget()

class OptionsWindowDataProcessingWidget(QtWidgets.QWidget):

    def __init__(self, parent, stream_widget, stream_name):
        super().__init__()
        self.ui = uic.loadUi("ui/dsp_ui/OptionsWindowDataProcessingWidget.ui", self)
        self.parent = parent
        self.stream_widget = stream_widget
        self.stream_name = stream_name

        self.AddDataProcessorBtn.clicked.connect(self.add_processor_btn_clicked)

    def init_data_processor_widget(self, selected_group_name):
        data_processors = get_group_data_processors(self.stream_name, group_name=selected_group_name)
        for data_processor in data_processors:
            # data_processor =
            if data_processor.data_processor_type == DataProcessorType.RealtimeButterBandpass:
                data_processor_widget = RealtimeButterworthBandPassWidget(data_processor)
            # if data_processor.data_processor_type == DataProcessorType.RealtimeNotch:
            #     # data_processor_widget = RealtimeButterworthBandPassWidget(data_processor)
                pass
            self.DataProcessorScrollArea.addWidget(data_processor_widget)


    def add_processor_btn_clicked(self):
        pass
        # data_processor_selector = data_processor_widget_type()
        # selected_data_processor = self.DataProcessorComboBox.currentText()
        # data_processor_widget = getattr(data_processor_selector, selected_data_processor)
        # self.DataProcessorScrollAreaVerticalLayout.addWidget(data_processor_widget)



