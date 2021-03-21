# This Python file uses the following encoding: utf-8
import os
import pickle
import time

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtWidgets, uic, sip

import numpy as np
from datetime import datetime

from utils.ui_utils import dialog_popup


class RecordingsTab(QtWidgets.QWidget):
    def __init__(self, parent, lsl_data_buffer: dict):
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

        self.StopRecordingBtn.setEnabled(False)
        self.parent = parent

    def start_recording_btn_ressed(self):
        if not (len(self.parent.LSL_data_buffer_dicts.keys()) >= 1 or len(self.parent.cam_workers) >= 1):
            dialog_popup('You need at least one LSL Stream or Capture opened to start recording!')
            return
        self.recording_buffer = {}  # clear buffer
        self.is_recording = True

        self.StartRecordingBtn.setEnabled(False)
        self.StopRecordingBtn.setEnabled(True)

    def stop_recording_btn_pressed(self):
        self.is_recording = False
        os.makedirs(self.saveRootTextEdit.toPlainText(), exist_ok=True)

        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")

        save_path = os.path.join(self.saveRootTextEdit.toPlainText(), '{0}-Exp_{1}-Sbj_{2}-Ssn_{3}.p'.format(dt_string,
                                                                                                             self.experimentNameTextEdit.toPlainText(),
                                                                                                             self.subjectTagTextEdit.toPlainText(),
                                                                                                             self.sessionTagTextEdit.toPlainText()))
        pickle.dump(self.recording_buffer, open(save_path, 'wb'))
        print('Saved to {0}'.format(save_path))
        dialog_popup('Saved to {0}'.format(save_path))

        self.StopRecordingBtn.setEnabled(False)
        self.StartRecordingBtn.setEnabled(True)

    def update_buffers(self, data_dict: dict):
        if self.is_recording:
            lsl_data_type = data_dict['lsl_data_type']  # what is the type of the newly-come data
            if lsl_data_type not in self.recording_buffer.keys():
                a = np.empty(shape=(data_dict['frames'].shape[0], 0))
                self.recording_buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                                        np.empty(shape=(0,))]  # data first, timestamps second

            buffered_data = self.recording_buffer[data_dict['lsl_data_type']][0]
            buffered_timestamps = self.recording_buffer[data_dict['lsl_data_type']][1]

            self.recording_buffer[lsl_data_type][0] = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
            self.recording_buffer[lsl_data_type][1] = np.concatenate([buffered_timestamps, data_dict['timestamps']])
            pass

    def update_camera_screen_buffer(self, cam_id, new_frame):
        if self.is_recording:
            if cam_id not in self.recording_buffer.keys():
                self.recording_buffer[cam_id] = [np.empty(shape=new_frame.shape + (0,)),
                                                 np.empty(shape=(0,))]

            _new_frame = np.expand_dims(new_frame, axis=-1)
            buffered_data = self.recording_buffer[cam_id][0]
            buffered_timestamps = self.recording_buffer[cam_id][1]

            self.recording_buffer[cam_id][0] = np.concatenate([buffered_data, _new_frame], axis=-1)
            self.recording_buffer[cam_id][1] = np.concatenate([buffered_timestamps, [time.time()]])
            pass
