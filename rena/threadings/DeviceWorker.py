import time
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, pyqtSignal, QMutex
from pylsl import local_clock

from rena.interfaces.AudioInputInterface import RenaAudioInputInterface
from rena.interfaces.DeviceInterface import DeviceInterface
from rena.presets.Presets import VideoDeviceChannelOrder
from rena.threadings.workers import RenaWorker
from rena.utils.image_utils import process_image



class DeviceWorker(QObject, RenaWorker):
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()
    def __init__(self,device_interface:DeviceInterface):
        super(DeviceWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self._device_interface = device_interface
        self.is_streaming = False
        self.interface_mutex = QMutex()

        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self.timestamp_queue = deque(maxlen=self._device_interface.get_device_nominal_sampling_rate() * 10)


    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()

            self.interface_mutex.lock()

            frames, timestamps = self._audio_input_interface.process_frames()

            self.timestamp_queue.extend(timestamps)
            if len(self.timestamp_queue) > 1:
                sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
            else:
                sampling_rate = np.nan


            data_dict = {'stream_name': self._device_interface._device_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

            self.interface_mutex.unlock()


    def start_stream(self):
        self.interface_mutex.lock()
        self._device_interface.start_sensor()
        self.is_streaming = True
        self.interface_mutex.unlock()

    def stop_stream(self):
        self.interface_mutex.lock()
        self._device_interface.stop_sensor()
        self.is_streaming = False
        self.interface_mutex.unlock()


    @pg.QtCore.pyqtSlot()
    def process_stream_availability(self):
        self.signal_stream_availability.emit(self.is_stream_available())

    def is_stream_available(self):
        return True