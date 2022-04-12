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

        # set file path for real time model.
        self.model_file_path = ''
        self.SelectModelFileBtn.clicked.connect(self.select_model_file_btn_pressed)

        # keep track of inference status.
        self.is_inferencing = False

        # a ring-buffer that will hold data to be inferenced.
        self.inference_buffer = {}

        # common initializations for tabs.
        self.parent = parent

        self.timer = QTimer()
        # TODO: user should be able to define inference interval; for now, use 500ms
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.emit_inference_data)

        self.startInferenceBtn.clicked.connect(self.start_inference_btn_pressed)
        self.stopInferenceBtn.clicked.connect(self.stop_inference_btn_pressed)
        self.stopInferenceBtn.setEnabled(False)


    def select_model_file_btn_pressed(self):
        selected_model_file = QFileDialog.getOpenFileName(self.widget_3, "Select File")[0]
        if selected_model_file != '':
            self.model_file_path = selected_model_file

    def start_inference_btn_pressed(self):
        if not (len(self.parent.LSL_data_buffer_dicts.keys()) >= 1):
            dialog_popup('You need at least one LSL Stream opened to start inference!')
            return
        self.inference_buffer = {} # clear buffer
        self.is_inferencing = True
        self.startInferenceBtn.setEnabled(False)
        self.stopInferenceBtn.setenabled(True)
        self.timer.start()

    def stop_inference_btn_pressed(self):
        self.is_inferencing = False
        self.stopInferenceBtn.setEnabled(False)
        self.startInferenceBtn.setEnabled(True)

        self.timer.stop()

    def emit_inference_data(self):
        pass

    def process_on_tick(self, window_size, time_step):
        """
        Buffer is loaded (from update_buffers).
        Slice the buffer depending on the window size and time step.
        Preprocess the buffer & model.
        Predict the result.
        Emit the data to the LSL outlet (call emit_inference_data)
        """

        # Buffer processing

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