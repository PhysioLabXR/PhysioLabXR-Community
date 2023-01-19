# This Python file uses the following encoding: utf-8

# This Python file uses the following encoding: utf-8

from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from rena.utils.rena_dsp_utils import RealtimeButterBandpass, RenaFilter, RealtimeButterHighpass


# This Python file uses the following encoding: utf-8


class FilterComponentButterworthHighPass(QtWidgets.QWidget):
    filter_on_change_signal = QtCore.pyqtSignal(RenaFilter)

    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/FilterComponentButterworthHighPass.ui", self)
        self.rena_filter = RealtimeButterHighpass()
        self.filter_activated = False
        self.filter_valid = False
        # low cut

        self.low_cut_frequency = 0
        self.order = 0

        self.lowCutFrequencyLineEdit.setValidator(QDoubleValidator())
        # high cut
        #
        self.filterOrderLineEdit.setValidator(QIntValidator())
        # self.high_cut_frequency = 0
        # self.low_cut_frequency = 0
        # self.order = 0
        self.lowCutFrequencyLineEdit.textChanged.connect(self.filter_config_on_change)
        self.filterOrderLineEdit.textChanged.connect(self.filter_config_on_change)

        self.removeFilterBtn.clicked.connect(self.remove_filter_button)

    def get_group_info(self):
        pass

    def filter_process_buffer(self, input_buffer):
        output = self.rena_filter.process_buffer(input_buffer)
        return output

    def remove_filter_button(self):
        # deactivate filter

        self.deleteLater()

    def filter_config_on_change(self):
        self.low_cut_frequency = self.get_filter_low_cut_frequency()
        self.order = self.get_filter_order()

        self.filter_config_valid()
        # if filter valid, send to the worker
        if self.filter_valid:
            self.filter_on_change_signal.emit(self.rena_filter)




    def get_filter_low_cut_frequency(self):
        try:
            low_cut_frequency = abs(float(self.lowCutFrequencyLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return low_cut_frequency

    def get_filter_order(self):
        try:
            order = abs(float(self.filterOrderLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return order

    def filter_config_valid(self):

        try:
            self.rena_filter = \
                RealtimeButterHighpass(lowcut=self.low_cut_frequency, order=self.order)
            self.filterInfoLabel.setText("filter valid")
            self.filter_valid = True
        except:
            self.filter_valid = False
            self.filterInfoLabel.setText("Invalid filter design")
            print("invalid filter design")

        if self.filter_valid:
            self.filterInfoLabel.setStyleSheet('color: green')
        else:
            self.filterInfoLabel.setStyleSheet('color: red')

