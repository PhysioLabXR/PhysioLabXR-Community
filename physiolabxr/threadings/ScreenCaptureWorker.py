import time

from mss import mss
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QObject

from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.threadings.workers import RenaWorker
from physiolabxr.utils.image_utils import process_image
from physiolabxr.utils.time_utils import get_clock_time

def get_screen_capture_size():
    img = mss().grab(mss().monitors[1])
    frame = np.array(img)
    return frame.shape[0], frame.shape[1]

class ScreenCaptureWorker(QObject, RenaWorker):

    def __init__(self, screen_label, video_scale: float, channel_order: VideoDeviceChannelOrder):
        super().__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.screen_label = screen_label
        self.is_streaming = True

        self.video_scale = video_scale
        self.channel_order = channel_order
        # self.bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}

    def stop_stream(self):
        self.is_streaming = False

    def start_stream(self):
        self.is_streaming = True

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            with mss() as sct:
                pull_data_start_time = time.perf_counter()
                # img = self.sct.grab(self.bounding_box)
                img = sct.grab(self.sct.monitors[1])
                frame = np.array(img)
                frame = frame.astype(np.uint8)
                frame = process_image(frame, self.channel_order, self.video_scale)
                frame = np.flip(frame, axis=0)
                self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
                self.signal_data.emit({"frame": frame, "timestamp": get_clock_time()})  # uses lsl local clock for syncing
