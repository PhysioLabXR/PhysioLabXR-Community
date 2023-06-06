# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets
from PyQt5 import uic

from rena.ui.dsp_ui.DataProcessorWidget import DataProcessorWidget
from rena.utils.realtime_DSP import *


class RealtimeButterworthBandPassWidget(DataProcessorWidget):

    def __init__(self, data_processor=RealtimeButterBandpass()):
        super().__init__()
        self.ui = uic.loadUi("ui/dsp_ui/RealtimeButterworthBandPassWidget.ui", self)


