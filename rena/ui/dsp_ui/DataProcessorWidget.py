# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets
from PyQt5 import uic
from rena.utils.realtime_DSP import *


class DataProcessorWidgetType(Enum):
    RealtimeButterworthBandPassWidget = RealtimeButterworthBandPassWidget


class DataProcessorWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

    def evoke_data_processor(self):
        pass



class RealtimeButterworthBandPassWidget(DataProcessorWidget):

    def __init__(self, data_processor=RealtimeButterBandpass()):
        super().__init__()
        self.ui = uic.loadUi("ui/dsp_ui/RealtimeButterworthBandPassWidget.ui", self)

