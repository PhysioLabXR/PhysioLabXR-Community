# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets
from PyQt6 import uic

from rena.configs.configs import AppConfigs
from rena.presets.presets_utils import get_group_data_processors
from rena.ui.dsp_ui.DataProcessorWidget import DataProcessorWidgetType
from rena.utils.dsp_utils.dsp_modules import *
# class data_processor_widget_type:
#     def __init__(self, stream_name):
#         self.RealtimeButterWorthBandPass = RealtimeButterworthBandPassWidget()
from rena.utils.ui_utils import clear_layout


class OptionsWindowDataProcessingWidget(QtWidgets.QWidget):

    def __init__(self, parent, stream_widget, stream_name):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_OptionsWindowDataProcessingWidget, self)
        self.parent = parent
        self.stream_widget = stream_widget
        self.stream_name = stream_name
        self.group_name = None

        self.AddDataProcessorBtn.clicked.connect(self.add_processor_btn_clicked)
        # add add button icon
        self.AddDataProcessorBtn.setIcon(AppConfigs()._icon_add)

        self.init_data_processor_combobox()

    def init_data_processor_combobox(self):
        for data_processor_type in DataProcessorType:
            self.DataProcessorComboBox.addItem(data_processor_type.value)

    def set_data_processing_widget_info(self, group_name):
        # set group name
        self.group_name = group_name

        clear_layout(self.DataProcessorScrollAreaVerticalLayout)
        data_processors = get_group_data_processors(self.stream_name, group_name=group_name)
        for data_processor in data_processors:
            data_processor_widget = getattr(DataProcessorWidgetType, data_processor.data_processor_type.value).value(self, data_processor)
            self.DataProcessorScrollAreaVerticalLayout.addWidget(data_processor_widget)

    def add_processor_btn_clicked(self):
        print('add_processor_btn_clicked')
        selected_data_processor = self.DataProcessorComboBox.currentText()
        data_processor_widget = getattr(DataProcessorWidgetType, selected_data_processor).value(self, adding_data_processor=True)
        self.DataProcessorScrollAreaVerticalLayout.addWidget(data_processor_widget)

    def remove_data_processor_widget(self, target):
        self.DataProcessorScrollAreaVerticalLayout.removeWidget(target)
        target.deleteLater()


    def change_group_name(self, new_name):
        self.group_name = new_name







