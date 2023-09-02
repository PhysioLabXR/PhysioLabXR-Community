import pickle

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QObject, pyqtSignal, QThread
import pyqtgraph as pg
from scipy.io import savemat
import numpy as np
import os
import csv

from physiolabxr.configs.configs import RecordingFileFormat, AppConfigs
from physiolabxr.presets.Presets import Presets
from physiolabxr.utils.data_utils import CsvStoreLoad
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.xdf_utils import create_xml_string, XDF


class RecordingConversionDialog(QtWidgets.QWidget):
    def __init__(self,  file_path, file_format: RecordingFileFormat):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_RecordingConversionDialog, self)
        self.setWindowTitle(f'Please wait for converting to {file_format.value}')

        self.file_format = file_format
        self.file_path = file_path

        self.recording_convertion_worker = RecordingConversionWorker(RNStream(file_path), file_format, file_path)
        self.thread = QThread()
        self.thread.started.connect(self.recording_convertion_worker.run)
        self.recording_convertion_worker.moveToThread(self.thread)

        self.recording_convertion_worker.progress.connect(self.conversion_progress)
        self.recording_convertion_worker.finished_streamin.connect(self.streamin_finished)
        self.recording_convertion_worker.finished_conversion.connect(self.conversion_finished)

        self.finish_button.clicked.connect(self.on_finish_button_clicked)
        self.finish_button.hide()
        self.is_conversion_complete = False

        self.thread.start()

    def conversion_finished(self, newfile_path):
        print('Conversion finished, showing the finish button')
        self.setWindowTitle('Conversion completed')
        self.progress_label.setText('Complete saving file to {}'.format(newfile_path))
        self.finish_button.show()
        self.is_conversion_complete = True
        self.activateWindow()

    def conversion_progress(self, progresses):
        read_bytes, total_bytes = progresses
        self.progress_label.setText('Loading file back in: {} % loaded'.format(str(round(100 * read_bytes/total_bytes, 2))))
        self.progress_label.repaint()
        # print('updated progress label')

    def streamin_finished(self):
        self.progress_label.setText('Converting to {}'.format(self.file_format))

    def on_finish_button_clicked(self):
        self.close()
        self.thread.quit()


class RecordingConversionWorker(QObject):
    finished_streamin = pyqtSignal()
    finished_conversion = pyqtSignal(str)
    progress = pyqtSignal(list)

    def __init__(self, stream, file_format: RecordingFileFormat, file_path):
        super().__init__()
        self.stream = stream
        self.file_format = file_format
        self.file_path = file_path

    def run(self):
        print("RecordingConversionWorker started running")
        file, buffer, read_bytes_count = None, None, None
        while True:
            file, buffer, read_bytes_count, total_bytes, finished = self.stream.stream_in_stepwise(file, buffer, read_bytes_count, jitter_removal=False)
            if finished:
                break
            self.progress.emit([read_bytes_count, total_bytes])
        self.finished_streamin.emit()

        newfile_path = self.file_path
        if self.file_format == RecordingFileFormat.matlab:
            newfile_path = self.file_path.replace(RecordingFileFormat.get_default_file_extension(),
                                                  self.file_format.get_file_extension())
            # buffer_copy = {}
            # for stream_label, data_ts_array in buffer.items():
            #     buffer_copy[stream_label + ' timestamp'] = data_ts_array[1]
            #     buffer_copy[stream_label] = data_ts_array[0]
            buffer = [{f'{s_name} timestamp': timestamps, s_name: data} for s_name, (data, timestamps) in buffer.items()]
            buffer = {k: v for d in buffer for k, v in d.items()}
            savemat(newfile_path, buffer, oned_as='row')
        elif self.file_format == RecordingFileFormat.pickle:
            newfile_path = self.file_path.replace(RecordingFileFormat.get_default_file_extension(), self.file_format.get_file_extension())
            pickle.dump(buffer, open(newfile_path, 'wb'))
        elif self.file_format == RecordingFileFormat.csv:
            csv_store = CsvStoreLoad()
            csv_store.store_csv(buffer, self.file_path)
        elif self.file_format == RecordingFileFormat.xdf:
            newfile_path = self.file_path.replace(RecordingFileFormat.get_default_file_extension(), self.file_format.get_file_extension())
            file_header_info = {'name': 'Test', 'user': 'ixi'}
            file_header_xml = create_xml_string(file_header_info)
            stream_headers = {}
            stream_footers = {}
            idx = 0
            for stream_label, data_ts_array in buffer.items():
                if stream_label == 'monitor 0':
                    stream_header_info = {'name': stream_label, 'nominal_srate': str(
                        len(data_ts_array[1])/(data_ts_array[1][-1]-data_ts_array[1][0])),
                                          'channel_count': str(data_ts_array[0].shape[0] * data_ts_array[0].shape[1] * data_ts_array[0].shape[2]),
                                          'channel_format': 'int8'}
                    stream_header_xml = create_xml_string(stream_header_info)
                    stream_headers[stream_label] = stream_header_xml
                    stream_footer_info = {'first_timestamp': str(data_ts_array[1][0]),
                                          'last_timestamp': str(data_ts_array[1][-1]),
                                          'sample_count': str(len(data_ts_array[1])), 'stream_name': stream_label,
                                          'stream_id': idx,
                                          'frame_dimension': data_ts_array[0].shape[0:3]}
                    stream_footers[stream_label] = stream_footer_info
                    idx += 1
                else:
                    stream_header_info = {'name': stream_label, 'nominal_srate': str(Presets().stream_presets[stream_label].nominal_sampling_rate),
                                          'channel_count': str(data_ts_array[0].shape[0]), 'channel_format': 'double64' if Presets().stream_presets[stream_label].data_type.value == 'float64' else Presets().stream_presets[stream_label].data_type.value}
                    stream_header_xml = create_xml_string(stream_header_info)
                    stream_headers[stream_label] = stream_header_xml
                    stream_footer_info = {'first_timestamp': str(data_ts_array[1][0]), 'last_timestamp': str(data_ts_array[1][-1]), 'sample_count': str(len(data_ts_array[1])), 'stream_name': stream_label, 'stream_id': idx}
                    # stream_footer_xml = create_xml_string(stream_footer_info)
                    stream_footers[stream_label] = stream_footer_info
                    idx += 1

            xdffile = XDF(file_header_xml, stream_headers, stream_footers)
            xdffile.store_xdf(newfile_path, buffer)
        else:
            raise NotImplementedError
        self.finished_conversion.emit(newfile_path)
