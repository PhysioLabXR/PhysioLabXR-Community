# This Python file uses the following encoding: utf-8
import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer

import rena.threadings.WebcamWorker
from rena.configs.configs import AppConfigs
from rena.presets.Presets import PresetType
from rena.presets.presets_utils import get_video_scale, get_video_channel_order, is_video_webcam, get_video_device_id
from rena.ui.BaseStreamWidget import BaseStreamWidget
from rena.ui.VideoDeviceOptions import VideoDeviceOptions
from rena.utils.ui_utils import dialog_popup
from rena.threadings.ScreenCaptureWorker import ScreenCaptureWorker
from rena.threadings.WebcamWorker import WebcamWorker


class VideoWidget(BaseStreamWidget):
    def __init__(self, parent_widget, parent_layout, video_preset_type: PresetType, video_device_name, insert_position=None):
        """
        This class overrides create_visualization_component() from BaseStreamWidget.
        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(parent_widget, parent_layout, video_preset_type, video_device_name, data_timer_interval=AppConfigs().video_device_refresh_interval, use_viz_buffer=False, insert_position=insert_position, options_widget=VideoDeviceOptions(parent_stream_widget=self, video_device_name=self.video_device_name))
        self.plot_widget = None
        self.StartStopStreamBtn.hide()  # video widget does not have start/stop button
        self.video_preset_type = video_preset_type

        # check if the video device is a camera or screen capture ####################################
        self.video_device_long_name = ('Webcam ' if self.video_preset_type == PresetType.WEBCAM else 'Screen Capture ') + str(video_device_name)

        # worker and worker threads ##########################################
        video_scale, channel_order = get_video_scale(self.stream_name), get_video_channel_order(self.stream_name)
        if self.video_preset_type == PresetType.WEBCAM:
            self.worker = WebcamWorker(get_video_device_id(video_device_name), video_scale, channel_order)
        else:
            self.worker = ScreenCaptureWorker(video_device_name, video_scale, channel_order)
        self.worker.signal_data.connect(self.visualize)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self.is_image_fitted_to_frame = False

        self.data_timer.start()

    def create_visualization_component(self):
        self.plot_widget = pg.PlotWidget()
        self.splitter.addWidget(self.plot_widget)
        self.plot_widget.enableAutoRange(enable=False)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

    def visualize(self, cam_id_cv_img_timestamp):
        self.viz_times.append(time.time())
        cam_id, image, timestamp = cam_id_cv_img_timestamp
        # qt_img = convert_rgb_to_qt_image(image)
        image = np.swapaxes(image, 0, 1)
        self.image_item.setImage(image)

        if not self.is_image_fitted_to_frame:
            self.plot_widget.setXRange(0, image.shape[0])
            self.plot_widget.setYRange(0, image.shape[1])
            self.is_image_fitted_to_frame = True

        # self.ImageLabel.setPixmap(qt_img)
        self.main_parent.recording_tab.update_camera_screen_buffer(cam_id, image, timestamp)

    def video_preset_changed(self):
        self.worker.video_scale = get_video_scale(self.video_device_name)
        self.worker.channel_order = get_video_channel_order(self.video_device_name)
        self.is_image_fitted_to_frame = False

