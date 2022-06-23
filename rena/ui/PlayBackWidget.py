import time

from PyQt5 import QtCore
from PyQt5 import QtWidgets, uic, sip
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import pyqtSignal

import numpy as np
from datetime import datetime

from rena import config

class PlayBackWidget(QtWidgets.QWidget):
    playback_signal = pyqtSignal(int)
    play_pause_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/PlayBackWidget.ui", self)
        self.parent = parent

        # playback status
        self.horizontalSlider.valueChanged.connect(self.emit_playback_position)
        self.playPauseButton.clicked.connect(self.emit_playback_stop)
        self.stopButton.clicked.connect(self.emit_playback_stop)

    def emit_playback_stop(self):
        self.play_pause_signal.emit()
        # self.playing = not self.playing
        # if self.playing:
        #     self.parent.start_replay()
        # else:
        #     self.parent.pause_replay()

    def emit_playback_stop(self):
        # self.playing = False
        # self.parent.stop_replay_btn_pressed()
        self.stop_signal.emit()

    def emit_playback_position(self, event):
        # use signal
        self.playback_signal.emit(event)