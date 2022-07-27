# This Python file uses the following encoding: utf-8
import os
import pickle
import time

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtWidgets, uic, sip

import numpy as np
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog

import rena.config
from rena.utils.data_utils import RNStream
from rena.utils.ui_utils import dialog_popup
import pylsl
from rena.threadings.workers import PlaybackWorker
from PyQt5.QtCore import QObject, QThread, pyqtSignal

class ReplaySeekBar(QtWidgets.QWidget):
    def __init__(self, parent, endTime):
        super().__init__()
        self.ui = uic.loadUi("ui/ReplaySeekBar.ui", self)
        self.parent = parent
        self.startTime = '0:00'
        self.endTime = self.getTimeStr(endTime)
        self.PlayPauseButton.clicked.connect(self.ex)
        self.StopButton.clicked.connect(self.ex)
        self.SkipBackButton.clicked.connect(self.ex)
        self.SkipForwardButton.clicked.connect(self.ex)
        self.SeekBar.valueChanged.connect(self.ex)

    def getTimeStr(self, endTime):
        hours = endTime // 60
        mins = endTime % 60
        return '{0:d}:{1:02d}'.format(hours,mins)

    def ex(self):
        print('aa')

