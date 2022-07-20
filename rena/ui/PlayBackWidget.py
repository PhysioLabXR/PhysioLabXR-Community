import time

from PyQt5 import QtCore
from PyQt5 import QtWidgets, uic, sip
from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap

import numpy as np
from datetime import datetime

from rena import config
from rena.ui_shared import start_stream_icon, stop_stream_icon

class PlayBackWidget(QtWidgets.QWidget):
    playback_signal = pyqtSignal(int)
    play_pause_signal = pyqtSignal(bool)
    stop_signal = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/PlayBackWidget.ui", self)
        self.parent = parent
        self.is_playing = True # default : True (widget is created when the replay begins)

        # playback status
        self.horizontalSlider.valueChanged.connect(self.emit_playback_position)
        self.playPauseButton.clicked.connect(self.emit_play_pause_button_clicked)
        self.stopButton.clicked.connect(self.emit_playback_stop)

    def emit_play_pause_button_clicked(self):
        print("Its clicked in playbackwidget")
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.playPauseButton.setIcon(stop_stream_icon)
            # self.playPauseButton.setIconSize(QtCore.QSize(100, 100))
        else:
            self.playPauseButton.setIcon(start_stream_icon)
            # self.playPauseButton.setIconSize(QtCore.QSize(100, 100))
        self.play_pause_signal.emit(self.is_playing)

    def emit_playback_stop(self):
        # self.playing = False
        # self.parent.stop_replay_btn_pressed()
        self.stop_signal.emit()

    def emit_playback_position(self, event):
        # use signal
        self.playback_signal.emit(event)

    def on_replay_tick(self, replay_progress):
        print("slider value is being updated!!", replay_progress)
        self.horizontalSlider.setValue(replay_progress)
