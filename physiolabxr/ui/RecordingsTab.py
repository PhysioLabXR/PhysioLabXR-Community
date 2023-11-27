# This Python file uses the following encoding: utf-8
import os
import sys
import time

from PyQt6 import QtWidgets, uic

import numpy as np
from datetime import datetime

from PyQt6.QtCore import QTimer, QSettings
from PyQt6.QtWidgets import QDialogButtonBox

from physiolabxr.ui import ui_shared
from physiolabxr.configs.config import settings
from physiolabxr.configs.configs import AppConfigs, RecordingFileFormat
from physiolabxr.ui.RecordingConversionDialog import RecordingPostProcessDialog
from physiolabxr.ui.ui_shared import stop_recording_text, start_recording_text
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.ui.dialogs import dialog_popup
import subprocess

from physiolabxr.utils.buffers import DataBuffer


class RecordingsTab(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_RecordingsTab, self)
        self.settings = QSettings('TeamRena', 'RenaLabApp')  # load the user settings

        self.recording_buffer = DataBuffer()
        self.postprocess_dialog = None
        self.is_recording = False

        self.StartStopRecordingBtn.clicked.connect(self.start_stop_recording_pressed)
        self.RecordingOptionBtn.clicked.connect(self.on_option_button_clicked)

        self.parent = parent

        self.save_stream = None

        self.save_path = ''

        self.recording_byte_count = 0

        self.experimentNameTextEdit.textChanged.connect(self.update_ui_save_file)
        self.subjectTagTextEdit.textChanged.connect(self.update_ui_save_file)
        self.sessionTagTextEdit.textChanged.connect(self.update_ui_save_file)

        self.update_ui_save_file()

        self.timer = QTimer()

    def start_stop_recording_pressed(self):
        if self.is_recording:
            self.stop_recording_btn_pressed()
        else:
            self.start_recording_btn_pressed()

    def start_recording_btn_pressed(self):
        if not self.parent.is_any_streaming():
            self.parent.current_dialog = dialog_popup('You need at least one stream to be streaming to start recording.',
                                                      title='Warning', main_parent=self.parent, buttons=QDialogButtonBox.StandardButton.Ok)
            return
        self.save_path = self.generate_save_path()  # get a new save path
        self.save_stream = RNStream(self.save_path)
        self.recording_buffer.clear_buffer()  # clear buffer
        self.is_recording = True
        self.recording_byte_count = 0
        self.StartStopRecordingBtn.setText(stop_recording_text)
        self.StartStopRecordingBtn.setIcon(AppConfigs()._icon_stop)

        # disable the text edit fields
        self.experimentNameTextEdit.setEnabled(False)
        self.subjectTagTextEdit.setEnabled(False)
        self.sessionTagTextEdit.setEnabled(False)

        self.timer.setInterval(AppConfigs.eviction_interval)
        self.timer.timeout.connect(self.evict_buffer)
        self.timer.start()

    def stop_recording_btn_pressed(self):
        self.is_recording = False

        self.evict_buffer()
        self.timer.stop()

        self.recording_byte_count = 0
        self.update_file_size_label()

        # convert file format
        # if AppConfigs().recording_file_format != RecordingFileFormat.dats:
        self.postprocess_dialog = self.postprocess_recording(self.save_path, AppConfigs().recording_file_format, self.parent.fire_action_show_recordings)
        # else:
        #     dialog_popup('Saved to {0}'.format(self.save_path), title='Info', mode='modeless', buttons=QDialogButtonBox.StandardButton.Ok,
        #                  main_parent=self.parent, additional_buttons={'Show in directory': self.parent.fire_action_show_recordings})

        self.StartStopRecordingBtn.setText(start_recording_text)
        self.StartStopRecordingBtn.setIcon(AppConfigs()._icon_start)

        # reenable the text edit fields
        self.experimentNameTextEdit.setEnabled(True)
        self.subjectTagTextEdit.setEnabled(True)
        self.sessionTagTextEdit.setEnabled(True)

    def update_recording_buffer(self, data_dict: dict):
        # TODO: change lsl_data_type to stream_name?
        if self.is_recording:
            self.recording_buffer.update_buffer(data_dict)

    def update_camera_screen_buffer(self, cam_id, new_frame, timestamp):
        if self.is_recording:
            self.recording_buffer.update_buffer({'stream_name': cam_id, 'frames': np.expand_dims(new_frame, axis=-1), 'timestamps': [timestamp]})

    def update_ui_save_file(self):
        if AppConfigs().recording_file_format == RecordingFileFormat.csv:
            self.FileSaveLabel.setText(f'{ui_shared.recording_tab_file_save_label_prefix_csv} {self.generate_save_path().strip(RecordingFileFormat.get_default_file_extension())}')
        else:
            self.FileSaveLabel.setText(
                ui_shared.recording_tab_file_save_label_prefix + self.generate_save_path().replace(RecordingFileFormat.get_default_file_extension(), AppConfigs().recording_file_format.get_file_extension()))

    def on_option_button_clicked(self):
        self.parent.open_settings_tab('recording')

    def generate_save_path(self):
        # datetime object containing current date and time
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
        return os.path.join(settings.value('recording_file_location'),
                            '{0}-Exp_{1}-Sbj_{2}-Ssn_{3}{4}'.format(dt_string,
                                                                      self.experimentNameTextEdit.toPlainText(),
                                                                      self.subjectTagTextEdit.toPlainText(),
                                                                      self.sessionTagTextEdit.toPlainText(),
                                                                    RecordingFileFormat.get_default_file_extension()))

    def evict_buffer(self):
        # print(self.recording_buffer)
        self.recording_byte_count += self.save_stream.stream_out(self.recording_buffer.buffer)
        self.recording_buffer.clear_buffer()
        self.update_file_size_label()

    def update_file_size_label(self):
        self.parent.recording_file_size_label. \
            setText('    Recording file size: {0} Mb'.format(str(round(self.recording_byte_count / 10 ** 6, 2))))

    def open_recording_directory(self):
        try:
            if sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
                os.startfile(settings.value('recording_file_location'))
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, "-R", settings.value('recording_file_location')])
        except FileNotFoundError:
            self.parent.current_dialog = dialog_popup(msg="Recording directory does not exist. "
                             "Please use a valid directory in the Recording Tab.", title="Error")

    def postprocess_recording(self, file_path, file_format: RecordingFileFormat, open_directory_func):
        #first load the .dats back
        recording_postprocess_dialog = RecordingPostProcessDialog(file_path, file_format, open_directory_func)
        recording_postprocess_dialog.show()
        return recording_postprocess_dialog