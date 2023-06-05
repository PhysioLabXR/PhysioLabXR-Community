# This Python file uses the following encoding: utf-8
import os
import sys
import time

from PyQt5 import QtWidgets, uic
import pyqtgraph as pg

import numpy as np
from datetime import datetime

from PyQt5.QtCore import QTimer, QSettings, QObject, pyqtSignal
from PyQt5.QtWidgets import QDialogButtonBox

from rena import config, ui_shared
from rena.configs.configs import AppConfigs, RecordingFileFormat
from rena.ui.RecordingConversionDialog import RecordingConversionDialog
from rena.ui_shared import start_stream_icon, stop_stream_icon
from rena.utils.data_utils import RNStream
from rena.utils.ui_utils import dialog_popup
import subprocess

class RecordingsTab(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/RecordingsTab.ui", self)
        self.settings = QSettings('TeamRena', 'RenaLabApp')  # load the user settings

        self.recording_buffer = {}

        self.is_recording = False

        self.StartStopRecordingBtn.clicked.connect(self.start_stop_recording_pressed)
        self.RecordingOptionBtn.clicked.connect(self.on_option_button_clicked)

        self.parent = parent

        self.save_stream = None

        self.save_path = ''

        self.timer = QTimer()
        self.timer.setInterval(config.EVICTION_INTERVAL)
        self.timer.timeout.connect(self.evict_buffer)

        self.recording_byte_count = 0

        self.experimentNameTextEdit.textChanged.connect(self.update_ui_save_file)
        self.subjectTagTextEdit.textChanged.connect(self.update_ui_save_file)
        self.sessionTagTextEdit.textChanged.connect(self.update_ui_save_file)

        self.update_ui_save_file()

    def start_stop_recording_pressed(self):
        if self.is_recording:
            self.stop_recording_btn_pressed()
        else:
            self.start_recording_btn_pressed()

    def start_recording_btn_pressed(self):
        if not self.parent.is_any_streaming():
            self.parent.current_dialog = dialog_popup('You need at least one stream opened to start recording.',
                                                      title='Warning', main_parent=self.parent, buttons=QDialogButtonBox.Ok)
            return
        self.save_path = self.generate_save_path()  # get a new save path
        self.save_stream = RNStream(self.save_path)
        self.recording_buffer = {}  # clear buffer
        self.is_recording = True
        self.recording_byte_count = 0
        self.StartStopRecordingBtn.setText(ui_shared.stop_recording_text)
        self.StartStopRecordingBtn.setIcon(stop_stream_icon)

        # disable the text edit fields
        self.experimentNameTextEdit.setEnabled(False)
        self.subjectTagTextEdit.setEnabled(False)
        self.sessionTagTextEdit.setEnabled(False)

        self.timer.start()

    def stop_recording_btn_pressed(self):
        self.is_recording = False

        self.evict_buffer()
        self.timer.stop()

        self.recording_byte_count = 0
        self.update_file_size_label()

        # convert file format
        if AppConfigs().recording_file_format != RecordingFileFormat.dats:
            self.convert_file_format(self.save_path, AppConfigs().recording_file_format )
        else:
            self.parent.current_dialog = dialog_popup('Saved to {0}'.format(self.save_path), title='Info', mode='modeless', buttons=QDialogButtonBox.Ok)

        self.StartStopRecordingBtn.setText(ui_shared.start_recording_text)
        self.StartStopRecordingBtn.setIcon(start_stream_icon)

        # reenable the text edit fields
        self.experimentNameTextEdit.setEnabled(True)
        self.subjectTagTextEdit.setEnabled(True)
        self.sessionTagTextEdit.setEnabled(True)

    def update_recording_buffer(self, data_dict: dict):
        if self.is_recording:
            lsl_data_type = data_dict['stream_name']  # get the type of the newly-come data

            if lsl_data_type not in self.recording_buffer.keys():
                self.recording_buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                                        np.empty(shape=(0,))]  # data first, timestamps second

            buffered_data = self.recording_buffer[data_dict['stream_name']][0]
            buffered_timestamps = self.recording_buffer[data_dict['stream_name']][1]

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

    def update_ui_save_file(self):
        self.FileSaveLabel.setText(ui_shared.recording_tab_file_save_label_prefix + self.generate_save_path())

    def on_option_button_clicked(self):
        self.parent.open_settings_tab('recording')

    def generate_save_path(self):
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
        return os.path.join(config.settings.value('recording_file_location'),
                            '{0}-Exp_{1}-Sbj_{2}-Ssn_{3}.dats'.format(dt_string,
                                                                      self.experimentNameTextEdit.toPlainText(),
                                                                      self.subjectTagTextEdit.toPlainText(),
                                                                      self.sessionTagTextEdit.toPlainText()))

    def evict_buffer(self):
        # print(self.recording_buffer)
        self.recording_byte_count += self.save_stream.stream_out(self.recording_buffer)
        self.recording_buffer = {}
        self.update_file_size_label()

    def update_file_size_label(self):
        self.parent.recordingFileSizeLabel. \
            setText('    Recording file size: {0} Mb'.format(str(round(self.recording_byte_count / 10 ** 6, 2))))

    def open_recording_directory(self):
        try:
            if sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
                os.startfile(config.settings.value('recording_file_location'))
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, "-R", config.settings.value('recording_file_location')])
        except FileNotFoundError:
            self.parent.current_dialog = dialog_popup(msg="Recording directory does not exist. "
                             "Please use a valid directory in the Recording Tab.", title="Error")

    def convert_file_format(self, file_path, file_format: RecordingFileFormat):
        #first load the .dats back
        recordingConversionDialog = RecordingConversionDialog(file_path, file_format)
        recordingConversionDialog.show()



