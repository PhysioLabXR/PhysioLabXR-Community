# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from rena.presets.presets_utils import add_data_processor_to_group_entry, remove_data_processor_to_group_entry
from rena.utils.dsp_utils.dsp_modules import *


class DataProcessorWidget(QtWidgets.QWidget):

    def __init__(self, parent, data_processor: DataProcessor, adding_data_processor=False):
        super().__init__()
        self.parent = parent
        self.data_processor: DataProcessor = data_processor


        if adding_data_processor:
            self.add_data_processor_to_group_entry()

    def add_data_processor_to_group_entry(self):
        # add data processor to group
        add_data_processor_to_group_entry(self.parent.stream_name,
                                          self.parent.group_name,
                                          data_processor=self.data_processor)

    def remove_data_processor_btn_clicked(self):
        # remove data processor from the group
        remove_data_processor_to_group_entry(self.parent.stream_name,
                                             self.parent.group_name,
                                             data_processor=self.data_processor)

        # remove the widget
        self.parent.remove_data_processor_widget(self)

    def init_input_field_constrain(self):
        pass

    def data_processor_settings_on_changed(self):
        pass





class RealtimeButterworthBandPassWidget(DataProcessorWidget):

    def __init__(self, parent, data_processor: RealtimeButterworthBandpass = RealtimeButterworthBandpass(), adding_data_processor=False):
        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi("ui/dsp_ui/RealtimeButterworthBandPassWidget.ui", self)
        self.data_processor = data_processor
        self.init_input_field_constrain()
        self.removeDataProcessorBtn.clicked.connect(self.remove_data_processor_btn_clicked)

    def init_input_field_constrain(self):
        self.lowCutLineEdit.setValidator(QDoubleValidator())
        self.highCutLineEdit.setValidator(QDoubleValidator())
        self.samplingRateLineEdit.setValidator(QDoubleValidator())
        self.orderLineEdit.setValidator(QIntValidator())

        self.lowCutLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.highCutLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.samplingRateLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.orderLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def data_processor_settings_on_changed(self):
        lowcut = self.get_lowcut()
        get_highcut = self.get_highcut()
        sampling_rate = self.get_sampling_rate()
        order = self.get_order()

        self.data_processor.evoke()

    def get_lowcut(self):
        try:
            lowcut = abs(float(self.lowCutLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return lowcut

    def get_highcut(self):
        try:
            highcut = abs(float(self.highCutLineEdit.text()))
        except ValueError:
            return 0
        return highcut

    def get_sampling_rate(self):
        try:
            sampling_rate = abs(float(self.samplingRateLineEdit.text()))
        except ValueError:
            return 0
        return sampling_rate

    def get_order(self):
        try:
            order = abs(int(self.orderLineEdit.text()))
        except ValueError:
            return 0
        return order





class DataProcessorWidgetType(Enum):
    RealtimeButterworthBandpass = RealtimeButterworthBandPassWidget

# if __name__ == '__main__':
#     print("test")
#     a = RealtimeButterBandpass()
#     b = getattr(DataProcessorWidgetType, type(a).__name__).value()
#     print("John")
