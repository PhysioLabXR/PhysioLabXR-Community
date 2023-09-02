# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFileDialog

import numpy as np

# lsl related imports
from pylsl import StreamInfo, StreamOutlet

from physiolabxr.scripting.PupilTensorflowModel import PupilTensorflowModel

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.utils.ui_utils import dialog_popup


# class EEGModel(RealTimeModel):
#     """
#     A concrete implementation of RealTimeModel for EEG data.
#     """
#     state = 2048
#     window = 500 # ms
#     num_channel = 64
#     # expected input size for EEG = [1024, 64]
#     # expected input size should be calculated
#     # TODO: override the methods for custom behaviors.

class InferenceTab(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_InferenceTab, self)

        # set file path for real time model.
        self.model_file_path = ''
        self.SelectModelFileBtn.clicked.connect(self.select_model_file_btn_pressed)

        # keep track of scripting status.
        self.is_inferencing = False

        # a ring-buffer that will hold data to be inferenced.
        self.inference_buffer = {}
        self.buffer_size = 1242 # arbitrarily set to 5000 for now.

        # common initializations for tabs.
        self.parent = parent

        self.timer = QTimer()
        # TODO: user should be able to define scripting interval; for now, use 500ms
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.emit_inference_data)

        self.StartInferenceBtn.clicked.connect(self.start_inference_btn_pressed)
        self.StopInferenceBtn.clicked.connect(self.stop_inference_btn_pressed)
        self.StopInferenceBtn.setEnabled(False)

        info = StreamInfo('PredictionResult', 'PredictionResult', 3, 8, 'float32', 'someuuid')
        self.prediction_stream = StreamOutlet(info)


    def select_model_file_btn_pressed(self):
        selected_model_dir = QFileDialog.getExistingDirectory(self.widget_3, "Select Model Directory")
        if selected_model_dir != '':
            self.ModelFileText.setPlainText(selected_model_dir)

    def start_inference_btn_pressed(self):
        if not (len(self.parent.LSL_data_buffer_dicts.keys()) >= 1):
            dialog_popup('You need at least one LSL Stream opened to start scripting!')
            return
        self.inference_buffer = {} # clear buffer

        # prepare the model
        self.model = PupilTensorflowModel('D:\PycharmProjects\ReNaAnalysis\Learning\Model\Pupil_ANN')  # TODO move this to the UI

        self.is_inferencing = True
        self.StartInferenceBtn.setEnabled(False)
        self.StopInferenceBtn.setEnabled(True)
        self.timer.start()

    def stop_inference_btn_pressed(self):
        self.is_inferencing = False
        self.StopInferenceBtn.setEnabled(False)
        self.StartInferenceBtn.setEnabled(True)

        self.timer.stop()

    def emit_inference_data(self):
        """
        Emits scripting results to a LSL stream outlet.
        """
        preprocessed_data = self.model.preprocess(self.inference_buffer[self.sourceStreamTextEdit.toPlainText()])
        if preprocessed_data is not None:
            y_pred = self.model.predict(preprocessed_data)
            self.prediction_stream.push_chunk(y_pred)
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
        lsl_data_type = data_dict['stream_name']  # get the type of the newly-come data
        if lsl_data_type not in self.inference_buffer.keys():
            self.inference_buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                                    np.empty(shape=(0,))]  # data first, timestamps second

        buffered_data = self.inference_buffer[data_dict['stream_name']][0]
        buffered_timestamps = self.inference_buffer[data_dict['stream_name']][1]

        # TODO: ring buffer implementation
        temp_data = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
        temp_timestamps = np.concatenate([buffered_timestamps, data_dict['timestamps']])
        # last <buffer_size> elements taken from the temp data buffer
        size_adjusted_data = temp_data[-self.buffer_size:]
        # last <buffer_size> elements taken from the temp timestamps buffer
        size_adjusted_timestamps = temp_timestamps[-self.buffer_size:]

        self.inference_buffer[lsl_data_type][0] = size_adjusted_data
        self.inference_buffer[lsl_data_type][1] = size_adjusted_timestamps