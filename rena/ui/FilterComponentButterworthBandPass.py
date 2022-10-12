# This Python file uses the following encoding: utf-8

# This Python file uses the following encoding: utf-8
from PyQt5 import QtWidgets, uic
from rena.utils.rena_dsp_utils import RealtimeButterHighpass


class FilterComponentButterworthBandPass(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("ui/FilterComponentButterworthBandPass.ui", self)
        self.rena_filter = RealtimeButterHighpass()

    def filter_process_buffer(self, input_buffer):
        output = self.rena_filter.process_buffer(input_buffer)
        return output


