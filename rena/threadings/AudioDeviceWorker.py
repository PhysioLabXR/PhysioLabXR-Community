import time
from collections import deque

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, pyqtSignal, QMutex
from pylsl import local_clock

from rena.interfaces.AudioInputInterface import RenaAudioInputInterface
from rena.presets.Presets import VideoDeviceChannelOrder
from rena.threadings.workers import RenaWorker
from rena.utils.image_utils import process_image


class AudioDeviceWorker(QObject, RenaWorker):
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()
    def __init__(self,stream_name, audio_device_index, channel_num):
        super(AudioDeviceWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)

        self._audio_input_interface = RenaAudioInputInterface(stream_name=stream_name, audio_device_index=audio_device_index, channels=channel_num)
        self.is_streaming = False
        self.interface_mutex = QMutex()

        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self.timestamp_queue = deque(maxlen=self._audio_input_interface.get_sampling_rate() * 10)


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


            data_dict = {'stream_name': self._audio_input_interface.audio_device_index, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

            self.interface_mutex.unlock()


    def start_stream(self):
        self.interface_mutex.lock()
        self._audio_input_interface.start_sensor()
        self.is_streaming = True
        self.interface_mutex.unlock()

    def stop_stream(self):
        self.interface_mutex.lock()
        self._audio_input_interface.stop_sensor()
        self.is_streaming = False
        self.interface_mutex.unlock()


    @pg.QtCore.pyqtSlot()
    def process_stream_availability(self):
        self.signal_stream_availability.emit(self.is_stream_available())

    def is_stream_available(self):
        return True

    # def is_streaming(self):
    #     return self.is_streaming

        # def __init__(self, input_device_index=0, frames_per_buffer=128, format=pyaudio.paInt16, channels=1, rate=4410):
        #     self.input_device_index = input_device_index
        #     self.frames_per_buffer = frames_per_buffer
        #     self.format = format
        #     self.channels = channels
        #     self.rate = rate
        #
        #     self.frame_duration = 1 / rate
        #
        #     self.audio = None
        #     self.stream = None


# class WebcamWorker(QObject, RenaWorker):
#     tick_signal = pyqtSignal()
#     change_pixmap_signal = pyqtSignal(tuple)
#
#     def __init__(self, cam_id, video_scale: float, channel_order: VideoDeviceChannelOrder):
#         super().__init__()
#         self.cap = None
#         self.cam_id = cam_id
#         self.cap = cv2.VideoCapture(self.cam_id)
#         self.tick_signal.connect(self.process_on_tick)
#         self.is_streaming = True
#
#         self.video_scale = video_scale
#         self.channel_order = channel_order
#
#     def stop_stream(self):
#         self.is_streaming = False
#         if self.cap is not None:
#             self.cap.release()
#
#     @pg.QtCore.pyqtSlot()
#     def process_on_tick(self):
#         if self.is_streaming:
#             pull_data_start_time = time.perf_counter()
#             ret, cv_img = self.cap.read()
#             if ret:
#                 cv_img = cv_img.astype(np.uint8)
#                 cv_img = process_image(cv_img, self.channel_order, self.video_scale)
#                 cv_img = np.flip(cv_img, axis=0)
#                 self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
#                 self.change_pixmap_signal.emit((self.cam_id, cv_img, local_clock()))  # uses lsl local clock for syncing
