import time
import traceback
from collections import deque

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QThread

from physiolabxr.interfaces.DeviceInterface.utils import create_custom_device_classes
from physiolabxr.threadings.workers import RenaWorker


class DeviceWorker(QObject, RenaWorker):
    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, stream_name, *args, **kwargs):
        super(DeviceWorker, self).__init__()

        self.stream_name = stream_name
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self.device_interface, self.device_options_widget_class = create_custom_device_classes(device_name=stream_name)

        # check if an Options class exists for the custom device
        # first check if a fil

        self.is_streaming = False
        self.interrupted = False
        self.timestamp_queue = deque(maxlen=1024)

        self.start_time = time.time()
        self.num_samples = 0

        self.previous_availability = None

        # self.init_dsp_client_server(self._lslInlet_interface.lsl_stream_name)
        self.interface_mutex = QMutex()

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        """
        if self._custom_device_interface.process_frames() raises an exception, it means the device has disconnected.

        In the emitted dictionary

        frames: must be a ndarray of shape (num_channels, num_timesteps) or a empty ndarray if no data is available
        """
        if QThread.currentThread().isInterruptionRequested():
            return
        if self.is_streaming and not self.interrupted:
            frames, timestamps, messages = [], [], []
            error_message = None
            pull_data_start_time = time.perf_counter()
            self.interface_mutex.lock()
            try:
                frames, timestamps, messages = self.device_interface.process_frames()  # get all data and remove it from internal buffer
            except Exception as e:
                traceback.print_exc()
                error_message = str(e)
                self.interrupted = True

            self.interface_mutex.unlock()

            if len(messages) > 0:  # first emit all the messages
                for message in messages:
                    self.signal_data.emit({'stream_name': self.stream_name, 'frames': np.empty(0), 'timestamps': [], 'sampling_rate': [], 'i': message})
            if error_message is None:
                try:
                    if len(frames) == 0:
                        return # no data available
                    frames = np.array(frames) # convert to numpy array in case it is not
                    self.timestamp_queue.extend(timestamps)
                    if len(self.timestamp_queue) > 1:  # compute the sampling rate
                        sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
                    else:
                        sampling_rate = np.nan
                    self.num_samples += len(timestamps)
                    data_dict = {'stream_name': self.stream_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
                    self.signal_data.emit(data_dict)
                    self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
                except Exception as e:  # in case there's something wrong with the frames or timestamps
                    traceback.print_exc()
                    error_message = str(e)
                    self.interrupted = True
                    self.signal_data.emit({'stream_name': self.stream_name, 'frames': np.empty(0), 'timestamps': [], 'sampling_rate': np.nan, 'e': error_message})
            else:
                self.signal_data.emit({'stream_name': self.stream_name, 'frames': np.empty(0), 'timestamps': [], 'sampling_rate': np.nan, 'e': error_message})

    @QtCore.pyqtSlot()
    def process_stream_availability(self):
        """ only emit when the stream becomes available
        """
        if QThread.currentThread().isInterruptionRequested():
            return
        is_stream_availability = self.device_interface.is_stream_available()
        if self.previous_availability is None:  # first time running
            self.previous_availability = is_stream_availability
            self.signal_stream_availability.emit(self.device_interface.is_stream_available())
        else:
            if is_stream_availability != self.previous_availability:
                self.previous_availability = is_stream_availability
                self.signal_stream_availability.emit(is_stream_availability)

    def reset_interface(self, stream_name, num_channels):
        self.interface_mutex.lock()
        self.device_interface = create_custom_device_classes(stream_name)
        self.interface_mutex.unlock()

    def start_stream(self, *args, **kwargs):
        self.device_interface.start_stream(*args, **kwargs)
        self.is_streaming = True
        self.interrupted = False

        self.num_samples = 0
        self.start_time = time.time()
        self.signal_stream_availability.emit(self.device_interface.is_stream_available())  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self.device_interface.stop_stream()
        self.is_streaming = False

    def is_stream_available(self):
        return self.device_interface.is_stream_available()
