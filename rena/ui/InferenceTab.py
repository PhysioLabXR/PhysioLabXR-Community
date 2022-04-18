# This Python file uses the following encoding: utf-8
import os
import time

from datetime import datetime
from abc import ABC, abstractmethod

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog

import numpy as np
import tensorflow as tf

# lsl related imports
from pylsl import StreamInfo, StreamOutlet, local_clock

from rena import config
from rena.utils.data_utils import RNStream
from rena.utils.ui_utils import dialog_popup


class RealTimeModel(ABC):
    """
    An abstract class for implementing inference models.
    """
    expected_input_size = None
    expected_preprocessed_input_size = None

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, new_model):
        self.__model = new_model

    @property
    def data_min(self):
        return self.__data_min

    @data_min.setter
    def data_min(self, value):
        self.__data_min = value

    @property
    def data_max(self):
        return self.__data_max

    @data_max.setter
    def data_max(self, value):
        self.__data_max = value

    @abstractmethod
    def resample(self, input, freq):
        """
        Implement for model-specific resampling method.
        This method will be used in preprocess().
        """
        pass

    def preprocess(self, input, window_size, time_step):
        if input.shape != self.expected_input_size:
            raise ValueError("Unexpected Input Size Provided.")

        # resample
        preprocessed = self.resample(input, __)

        # min max normalization
        preprocessed = (preprocessed - self.data_min) / (self.data_max - self.data_min)

        # slice into samples

        return preprocessed

    def predict(self, input, **kwargs):
        """
        Have ML Model predict the probability of the data being a target,
        a distractor, or a novelty.

        Returns 3 probability values.
        """
        if input.shape != self.expected_preprocessed_input_size:
            raise ValueError("Unexpected Expected Input Size Provided.")

        y_hypo = self.model.predict(input)
        return y_hypo

    def prepare_model(self, model_path, preprocess_params_path):
        self.model = tf.keras.models.load_model(model_path)
        # self.data_min, self.data_max = load_params(preprocess_params_path)

class EEGModel(RealTimeModel):
    """
    A concrete implementation of RealTimeModel for EEG data.
    """
    state = 2048
    window = 500 # ms
    num_channel = 64
    # expected input size for EEG = [1024, 64]
    # expected input size should be calculated
    # TODO: override the methods for custom behaviors.

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
        self.buffer_size = 5000 # arbitrarily set to 5000 for now.

        # common initializations for tabs.
        self.parent = parent

        self.timer = QTimer()
        # TODO: user should be able to define inference interval; for now, use 500ms
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.emit_inference_data)

        self.StartInferenceBtn.clicked.connect(self.start_inference_btn_pressed)
        self.StopInferenceBtn.clicked.connect(self.stop_inference_btn_pressed)
        self.StopInferenceBtn.setEnabled(False)


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
        self.StartInferenceBtn.setEnabled(False)
        self.StopInferenceBtn.setenabled(True)
        self.timer.start()

    def stop_inference_btn_pressed(self):
        self.is_inferencing = False
        self.StopInferenceBtn.setEnabled(False)
        self.StartInferenceBtn.setEnabled(True)

        self.timer.stop()

    def emit_inference_data(self):
        """
        Emits inferenced data to a LSL stream outlet.
        """
        # create a new stream info
        # info = StreamInfo(name, type_, n_channels, srate, 'float32', uuid_)

        # make an outlet
        outlet = StreamOutlet(info)

        # while True:
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
        if lsl_data_type not in self.inference_buffer.keys():
            self.inference_buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                                    np.empty(shape=(0,))]  # data first, timestamps second

        buffered_data = self.inference_buffer[data_dict['lsl_data_type']][0]
        buffered_timestamps = self.inference_buffer[data_dict['lsl_data_type']][1]

        # TODO: ring buffer implementation
        temp_data = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
        temp_timestamps = np.concatenate([buffered_timestamps, data_dict['timestamps']])
        # last <buffer_size> elements taken from the temp data buffer
        size_adjusted_data = temp_data[-self.buffer_size:]
        # last <buffer_size> elements taken from the temp timestamps buffer
        size_adjusted_timestamps = temp_timestamps[-self.buffer_size:]

        self.inference_buffer[lsl_data_type][0] = size_adjusted_data
        self.inference_buffer[lsl_data_type][1] = size_adjusted_timestamps