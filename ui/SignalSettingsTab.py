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
from PyQt5.QtWidgets import QFileDialog, QDialog

import config
from utils.data_utils import RNStream
from utils.ui_utils import dialog_popup


class SignalSettingsTab(QDialog):
    def __init__(self, parent=None):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__(parent=parent)
        self.ui = uic.loadUi("ui/SignalSettingsTab.ui", self)


        self.parent = parent


