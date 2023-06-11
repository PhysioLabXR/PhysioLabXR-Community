# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtGui import QPixmap
from rena.presets.presets_utils import add_data_processor_to_group_entry, remove_data_processor_to_group_entry, \
    get_group_channel_num
from rena.ui_shared import minus_icon
from rena.utils.dsp_utils.dsp_modules import *


class DataProcessorWidget(QtWidgets.QWidget):

    def __init__(self, parent, data_processor: DataProcessor, adding_data_processor=False):
        super().__init__()
        self.parent = parent
        self.data_processor_invalid_pixmap = QPixmap('../media/icons/streamwidget_stream_unavailable.png')
        self.data_processor_valid_pixmap = QPixmap('../media/icons/streamwidget_stream_available.png')
        self.data_processor_activated_pixmap = QPixmap('../media/icons/streamwidget_stream_viz_active.png')
        self.data_processor = data_processor


        if adding_data_processor:
            self.add_data_processor_to_group_entry()

    def __post_init__(self):
        self.removeDataProcessorBtn.clicked.connect(self.remove_data_processor_btn_clicked)
        self.removeDataProcessorBtn.setIcon(minus_icon)

        self.set_data_processor_input_field_value()
        self.set_data_processor_input_field_constrain()
        self.connect_data_processor_input_field_signal()
        self.set_data_processor_state_label()

        self.data_processor.data_processor_valid_signal.connect(self.set_data_processor_state_label)
        self.data_processor.data_processor_activated_signal.connect(self.set_data_processor_state_label)
        self.ActivateDataProcessorCheckbox.setStyleSheet("QCheckBox { }")
        # self.ActivateDataProcessorCheckbox.setFixedSize(20, 20)

    def add_data_processor_to_group_entry(self):
        # add data processor to group
        add_data_processor_to_group_entry(self.parent.stream_name,
                                          self.parent.group_name,
                                          data_processor=self.data_processor)
        self.data_processor.set_channel_num(channel_num=get_group_channel_num(self.parent.stream_name,
                                                                              self.parent.group_name))

    def remove_data_processor_btn_clicked(self):
        # remove data processor from the group
        remove_data_processor_to_group_entry(self.parent.stream_name,
                                             self.parent.group_name,
                                             data_processor=self.data_processor)

        # remove the widget
        self.parent.remove_data_processor_widget(self)

    def set_data_processor_input_field_value(self):
        self.ActivateDataProcessorCheckbox.setCheckState(self.data_processor.data_processor_activated)

    def set_data_processor_input_field_constrain(self):
        pass

    def connect_data_processor_input_field_signal(self):
        # connect activate signal
        self.ActivateDataProcessorCheckbox.stateChanged.connect(self.activate_data_processor_checkbox_on_changed)

    def activate_data_processor_checkbox_on_changed(self):
        if self.ActivateDataProcessorCheckbox.isChecked():
            self.data_processor.set_data_processor_activated(True)
        else:
            self.data_processor.set_data_processor_activated(False)

    def data_processor_settings_on_changed(self):
        pass

    def set_data_processor_state_label(self):
        print(self.data_processor.data_processor_valid, self.data_processor.data_processor_activated)
        if not self.data_processor.data_processor_valid:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_invalid_pixmap)
        elif self.data_processor.data_processor_valid and self.data_processor.data_processor_activated:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_activated_pixmap)
        elif self.data_processor.data_processor_valid:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_valid_pixmap)
        else:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_invalid_pixmap)


class ButterworthBandPassFilterWidget(DataProcessorWidget):

    def __init__(self, parent, data_processor: ButterworthBandpassFilter = ButterworthBandpassFilter(),
                 adding_data_processor=False):
        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi("ui/dsp_ui/ButterworthBandPassFilterWidget.ui", self)
        # self.data_processor = data_processor

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(ButterworthBandPassFilterWidget, self).set_data_processor_input_field_value()
        self.lowCutLineEdit.setText(str(self.data_processor.lowcut))
        self.highCutLineEdit.setText(str(self.data_processor.highcut))
        self.fsLineEdit.setText(str(self.data_processor.fs))
        self.orderLineEdit.setText(str(self.data_processor.order))

    def set_data_processor_input_field_constrain(self):
        self.lowCutLineEdit.setValidator(QDoubleValidator())
        self.highCutLineEdit.setValidator(QDoubleValidator())
        self.fsLineEdit.setValidator(QDoubleValidator())
        self.orderLineEdit.setValidator(QIntValidator())

    def connect_data_processor_input_field_signal(self):
        super(ButterworthBandPassFilterWidget, self).connect_data_processor_input_field_signal()
        self.lowCutLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.highCutLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.fsLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.orderLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def data_processor_settings_on_changed(self):
        lowcut = self.get_lowcut()
        highcut = self.get_highcut()
        fs = self.get_fs()
        order = self.get_order()
        # try evoke data processor
        self.data_processor.set_data_processor_params(lowcut=lowcut, highcut=highcut, fs=fs, order=order)
        # self.set_data_processor_state_label()

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

    def get_fs(self):
        try:
            sampling_rate = abs(float(self.fsLineEdit.text()))
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
    ButterworthBandpassFilter = ButterworthBandPassFilterWidget
    # NotchFilter = NotchFilterWidgetWidget
