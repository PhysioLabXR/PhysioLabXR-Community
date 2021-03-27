# This Python file uses the following encoding: utf-8
import os
import pickle
import sys
import time

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtWidgets, uic, sip

import numpy as np
from datetime import datetime

from PyQt5.QtCore import QTimer, QFile, QTextStream
from PyQt5.QtWidgets import QFileDialog

import config_ui

from utils.ui_utils import stream_stylesheet
from utils.data_utils import RNStream
from utils.ui_utils import dialog_popup


class SettingsTab(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/SettingsTab.ui", self)

        self.theme = config_ui.default_theme

        self.ToggleThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)

    def toggle_theme_btn_pressed(self):
        print("toggle theme")

        if self.theme == 'dark':
            url = 'ui/stylesheet/light.qss'
            stream_stylesheet(url)
            self.theme = 'light'
        else:
            url = 'ui/stylesheet/dark.qss'
            stream_stylesheet(url)
            self.theme = 'dark'
