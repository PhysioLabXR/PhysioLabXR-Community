import time

import pyscreeze
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtCore import QObject
from pylsl import local_clock

from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.threadings.workers import RenaWorker
from physiolabxr.utils.image_utils import process_image


class ScreenCaptureWorker(QObject, RenaWorker):

    def __init__(self, screen_label, video_scale: float, channel_order: VideoDeviceChannelOrder):
        super().__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.screen_label = screen_label
        self.is_streaming = True

        self.video_scale = video_scale
        self.channel_order = channel_order

    def stop_stream(self):
        self.is_streaming = False

    def start_stream(self):
        self.is_streaming = True

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            img = pyscreeze.screenshot()
            frame = np.array(img)
            frame = frame.astype(np.uint8)
            frame = process_image(frame, self.channel_order, self.video_scale)
            frame = np.flip(frame, axis=0)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
            self.signal_data.emit({"frame": frame, "timestamp": local_clock()})  # uses lsl local clock for syncing
