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
from PyQt5.QtWidgets import QFileDialog

import config
from utils.data_utils import RNStream
from utils.ui_utils import dialog_popup


class RecordingsTab(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/RecordingsTab.ui", self)

        self.recording_buffer = {}

        self.is_recording = False

        self.StartRecordingBtn.clicked.connect(self.start_recording_btn_ressed)
        self.StopRecordingBtn.clicked.connect(self.stop_recording_btn_pressed)
        self.SelectDataDirBtn.clicked.connect(self.select_data_dir_btn_pressed)

        self.StopRecordingBtn.setEnabled(False)
        self.parent = parent

        self.data_dir = config.DEFAULT_DATA_DIR
        self.save_path = ''
        self.save_stream = None

        self.saveRootTextEdit.setText(self.data_dir + '/')

        self.timer = QTimer()
        self.timer.setInterval(config.EVICTION_INTERVAL)
        self.timer.timeout.connect(self.evict_buffer)

        self.recording_byte_count = 0

    def select_data_dir_btn_pressed(self):

        selected_data_dir = str(QFileDialog.getExistingDirectory(self.widget_3, "Select Directory"))

        if selected_data_dir != '':
            self.data_dir = selected_data_dir

        print("Selected data dir: ", self.data_dir)
        self.saveRootTextEdit.setText(self.data_dir + '/')

    def start_recording_btn_ressed(self):
        if not (len(self.parent.LSL_data_buffer_dicts.keys()) >= 1 or len(self.parent.cam_workers) >= 1):
            dialog_popup('You need at least one LSL Stream or Capture opened to start recording!')
            return
        self.save_path = self.generate_save_path()  # get a new save path
        self.save_stream = RNStream(self.save_path)
        self.recording_buffer = {}  # clear buffer
        self.is_recording = True
        self.StartRecordingBtn.setEnabled(False)
        self.StopRecordingBtn.setEnabled(True)
        self.recording_byte_count = 0

        self.timer.start()

    def stop_recording_btn_pressed(self):
        self.is_recording = False
        self.StopRecordingBtn.setEnabled(False)
        self.StartRecordingBtn.setEnabled(True)

        self.evict_buffer()
        self.timer.stop()

        self.recording_byte_count = 0
        self.update_file_size_label()
        dialog_popup('Saved to {0}'.format(self.save_path), title='Info')

    def update_buffers(self, data_dict: dict):
        if self.is_recording:
            lsl_data_type = data_dict['lsl_data_type']  # get the type of the newly-come data
            if lsl_data_type not in self.recording_buffer.keys():
                self.recording_buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                                        np.empty(shape=(0,))]  # data first, timestamps second

            buffered_data = self.recording_buffer[data_dict['lsl_data_type']][0]
            buffered_timestamps = self.recording_buffer[data_dict['lsl_data_type']][1]

            self.recording_buffer[lsl_data_type][0] = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
            self.recording_buffer[lsl_data_type][1] = np.concatenate([buffered_timestamps, data_dict['timestamps']])
            pass

    def update_camera_screen_buffer(self, cam_id, new_frame, timestamp):
        if self.is_recording:
            if cam_id not in self.recording_buffer.keys():  # note array data type is uint8 0~255
                self.recording_buffer[cam_id] = [np.empty(shape=new_frame.shape + (0,), dtype=np.uint8),
                                                 np.empty(shape=(0,)), np.empty(shape=(0,))]

            _new_frame = np.expand_dims(new_frame, axis=-1)
            buffered_data = self.recording_buffer[cam_id][0]
            buffered_timestamps = self.recording_buffer[cam_id][1]

            self.recording_buffer[cam_id][0] = np.concatenate([buffered_data, _new_frame.astype(np.uint8)], axis=-1)
            self.recording_buffer[cam_id][1] = np.concatenate([buffered_timestamps, [timestamp]])
            self.recording_buffer[cam_id][2] = np.concatenate([self.recording_buffer[cam_id][2], [time.time()]])

            pass

    def generate_save_path(self):
        os.makedirs(self.saveRootTextEdit.toPlainText(), exist_ok=True)
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
        return os.path.join(self.saveRootTextEdit.toPlainText(),
                            '{0}-Exp_{1}-Sbj_{2}-Ssn_{3}.dats'.format(dt_string,
                                                                      self.experimentNameTextEdit.toPlainText(),
                                                                      self.subjectTagTextEdit.toPlainText(),
                                                                      self.sessionTagTextEdit.toPlainText()))

    def evict_buffer(self):
        self.recording_byte_count += self.save_stream.stream_out(self.recording_buffer)
        self.recording_buffer = {}
        self.update_file_size_label()

    def update_file_size_label(self):
        self.parent.recordingFileSizeLabel. \
            setText('    Recording file size: {0} Mb'.format(str(round(self.recording_byte_count / 10 ** 6, 2))))
