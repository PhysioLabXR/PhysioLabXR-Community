# This Python file uses the following encoding: utf-8
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, QThread

import physiolabxr.threadings.WebcamWorker
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType
from physiolabxr.presets.presets_utils import get_video_scale, get_video_channel_order, is_video_webcam, get_video_device_id
from physiolabxr.ui.BaseStreamWidget import BaseStreamWidget
from physiolabxr.ui.VideoDeviceOptions import VideoDeviceOptions
from physiolabxr.utils.ui_utils import dialog_popup
from physiolabxr.threadings.ScreenCaptureWorker import ScreenCaptureWorker
from physiolabxr.threadings.WebcamWorker import WebcamWorker


class VideoWidget(BaseStreamWidget):
    def __init__(self, parent_widget, parent_layout, video_preset_type: PresetType, video_device_name, insert_position=None):
        """
        This class overrides create_visualization_component() from BaseStreamWidget.
        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(parent_widget, parent_layout, video_preset_type, video_device_name, data_timer_interval=AppConfigs().video_device_refresh_interval,
                         use_viz_buffer=False, insert_position=insert_position, option_widget_call=lambda: VideoDeviceOptions(parent_stream_widget=self, video_device_name=video_device_name))
        self.StartStopStreamBtn.hide()  # video widget does not have start/stop button
        self.video_preset_type = video_preset_type
        self.is_stream_available = True  # a video device is always available

        # check if the video device is a camera or screen capture ####################################
        self.video_device_long_name = ('Webcam ' if self.video_preset_type == PresetType.WEBCAM else 'Screen Capture ') + str(video_device_name)

        # worker and worker threads ##########################################
        video_scale, channel_order = get_video_scale(self.stream_name), get_video_channel_order(self.stream_name)
        if self.video_preset_type == PresetType.WEBCAM:
            self.worker = WebcamWorker(get_video_device_id(video_device_name), video_scale, channel_order)
        else:
            self.worker = ScreenCaptureWorker(video_device_name, video_scale, channel_order)
        self.connect_worker(self.worker, False)
        self.is_image_fitted_to_frame = False
        self.data_timer.start()

    def create_visualization_component(self):
        self.plot_widget = pg.PlotWidget()
        self.splitter.addWidget(self.plot_widget)
        self.plot_widget.enableAutoRange(enable=False)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

    def process_stream_data(self, cam_id_cv_img_timestamp):
        self.viz_times.append(time.time())
        image, timestamp = cam_id_cv_img_timestamp["frame"], cam_id_cv_img_timestamp["timestamp"]
        image = np.swapaxes(image, 0, 1)
        self.image_item.setImage(image)

        if not self.is_image_fitted_to_frame:
            self.plot_widget.setXRange(0, image.shape[0])
            self.plot_widget.setYRange(0, image.shape[1])
            self.is_image_fitted_to_frame = True

        self.main_parent.recording_tab.update_camera_screen_buffer(self.stream_name, image, timestamp)

    def video_preset_changed(self):
        self.worker.video_scale = get_video_scale(self.stream_name)
        self.worker.channel_order = get_video_channel_order(self.stream_name)
        self.is_image_fitted_to_frame = False

