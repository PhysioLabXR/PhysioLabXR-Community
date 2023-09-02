import time

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, pyqtSignal
from pylsl import local_clock

from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.threadings.workers import RenaWorker
from physiolabxr.utils.image_utils import process_image


class WebcamWorker(QObject, RenaWorker):

    def __init__(self, cam_id, video_scale: float, channel_order: VideoDeviceChannelOrder):
        super().__init__()
        self.cap = None
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(self.cam_id)
        self.signal_data_tick.connect(self.process_on_tick)
        self.is_streaming = True

        self.video_scale = video_scale
        self.channel_order = channel_order

    def stop_stream(self):
        self.is_streaming = False
        if self.cap is not None:
            self.cap.release()

    def start_stream(self):
        self.is_streaming = True
        self.cap = cv2.VideoCapture(self.cam_id)

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            ret, cv_img = self.cap.read()
            if ret:
                cv_img = cv_img.astype(np.uint8)
                cv_img = process_image(cv_img, self.channel_order, self.video_scale)
                cv_img = np.flip(cv_img, axis=0)
                self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
                self.signal_data.emit({"camera id": self.cam_id, "frame": cv_img, "timestamp": local_clock()})  # uses lsl local clock for syncing
