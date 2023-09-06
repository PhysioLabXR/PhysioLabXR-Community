import time
from collections import deque

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QThread

from physiolabxr.interfaces.AudioInputInterface import AudioInputInterface, create_audio_input_interface
from physiolabxr.threadings.workers import RenaWorker


class AudioInputDeviceWorker(QObject, RenaWorker):
    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, stream_name, *args, **kwargs):
        super(AudioInputDeviceWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self._audio_device_interface: AudioInputInterface = create_audio_input_interface(stream_name)
        # self._lslInlet_interface = create_lsl_interface(stream_name, num_channels)
        self.is_streaming = False
        self.timestamp_queue = deque(maxlen=1024)

        self.start_time = time.time()
        self.num_samples = 0

        self.previous_availability = None

        # self.init_dsp_client_server(self._lslInlet_interface.lsl_stream_name)
        self.interface_mutex = QMutex()

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if QThread.currentThread().isInterruptionRequested():
            return
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            self.interface_mutex.lock()
            frames, timestamps = self._audio_device_interface.process_frames()  # get all data and remove it from internal buffer
            self.timestamp_queue.extend(timestamps)
            if len(self.timestamp_queue) > 1:
                sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
            else:
                sampling_rate = np.nan

            self.interface_mutex.unlock()

            if frames.shape[-1] == 0:
                return

            self.num_samples += len(timestamps)

            data_dict = {'stream_name': self._audio_device_interface._device_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

    @QtCore.pyqtSlot()
    def process_stream_availability(self):
        """
        only emit when the stream is not available
        """
        if QThread.currentThread().isInterruptionRequested():
            return
        is_stream_availability = self._audio_device_interface.is_stream_available()
        if self.previous_availability is None:  # first time running
            self.previous_availability = is_stream_availability
            self.signal_stream_availability.emit(self._audio_device_interface.is_stream_available())
        else:
            if is_stream_availability != self.previous_availability:
                self.previous_availability = is_stream_availability
                self.signal_stream_availability.emit(is_stream_availability)

    def reset_interface(self, stream_name, num_channels):
        self.interface_mutex.lock()
        self._audio_device_interface = create_audio_input_interface(stream_name)
        self.interface_mutex.unlock()

    def start_stream(self):
        self._audio_device_interface.start_stream()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()
        self.signal_stream_availability.emit(self._audio_device_interface.is_stream_available())  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self._audio_device_interface.stop_stream()
        self.is_streaming = False

    def is_stream_available(self):
        return self._audio_device_interface.is_stream_available()
