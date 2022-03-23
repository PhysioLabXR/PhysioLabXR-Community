# This Python file uses the following encoding: utf-8
import os
import time

from PyQt5 import QtWidgets, uic

import numpy as np
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog

from rena import config
from rena.utils.data_utils import RNStream
from rena.utils.ui_utils import dialog_popup


class InferenceTab(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/InferenceTab.ui", self)
        self.SelectModelFileBtn.clicked.connect(self.select_model_file_btn_pressed)

    def select_model_file_btn_pressed(self):
        selected_model_file = QFileDialog.getOpenFileName(self.widget_3, "Select File")[0]
        if selected_model_file != '':
            self.model_file_path = selected_model_file

    def start_inference_btn_pressed(self):
        pass