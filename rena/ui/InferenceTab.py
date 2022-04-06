# This Python file uses the following encoding: utf-8
import os
import time

from PyQt5 import QtWidgets, uic

import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog

from rena import config
from rena.utils.data_utils import RNStream
from rena.utils.ui_utils import dialog_popup


class RealTimeModel(ABC):
    """
    An abstract class for implementing inference models.
    """
    def preprocess(self, x, **kwargs):
        pass

    def predict(self, x, **kwargs):
        pass

    def prepare_model(self, model_path, preprocess_params_path):
        pass


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

    def process_on_tick(self, window_size, time_step):
        """
        Receive data from buffer.
        Slice the buffer depending on the window size and time step.
        Preprocess the buffer & model.
        Predict the result.
        Emit the data to the LSL outlet.
        """
        pass

    def update_buffers(self, data_dict: dict):
        lsl_data_type = data_dict['lsl_data_type']  # get the type of the newly-come data
        if lsl_data_type not in self.recording_buffer.keys():
            self.recording_buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                                    np.empty(shape=(0,))]  # data first, timestamps second

        buffered_data = self.recording_buffer[data_dict['lsl_data_type']][0]
        buffered_timestamps = self.recording_buffer[data_dict['lsl_data_type']][1]

        self.recording_buffer[lsl_data_type][0] = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
        self.recording_buffer[lsl_data_type][1] = np.concatenate([buffered_timestamps, data_dict['timestamps']])
        pass