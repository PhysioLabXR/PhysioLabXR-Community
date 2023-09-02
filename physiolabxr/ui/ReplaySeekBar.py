# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets, uic

from physiolabxr.configs.configs import AppConfigs


class ReplaySeekBar(QtWidgets.QWidget):
    def __init__(self, parent, endTime):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_ReplaySeekBar, self)
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

