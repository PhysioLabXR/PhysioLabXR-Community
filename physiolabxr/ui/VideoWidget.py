# This Python file uses the following encoding: utf-8
import time
from collections import deque

import numpy as np
import pyqtgraph as pg

from physiolabxr.configs import config_ui
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType
from physiolabxr.presets.presets_utils import get_video_scale, get_video_channel_order, get_video_device_id
from physiolabxr.ui.BaseStreamWidget import BaseStreamWidget
from physiolabxr.ui.VideoDeviceOptions import VideoDeviceOptions
from physiolabxr.threadings.ScreenCaptureWorker import ScreenCaptureWorker
from physiolabxr.threadings.WebcamWorker import WebcamWorker
from physiolabxr.utils.image_utils import process_image


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
        self.timestamp_queue = deque(maxlen=1024)

    def create_visualization_component(self):
        self.plot_widget = pg.PlotWidget()
        self.splitter.addWidget(self.plot_widget)
        self.plot_widget.enableAutoRange(enable=False)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

    def process_stream_data(self, cam_id_cv_img_timestamp):
        self.viz_times.append(time.time())
        display_image, timestamp = cam_id_cv_img_timestamp["frame"].copy(), cam_id_cv_img_timestamp["timestamp"]

        # alter the image for display
        display_image = process_image(display_image, self.worker.channel_order)  # scale is already changed by the worker
        display_image = np.flip(display_image, axis=0)

        display_image = np.swapaxes(display_image, 0, 1)


        self.image_item.setImage(display_image)

        # compute and display the frame rate
        self.timestamp_queue.append(timestamp)
        if len(self.timestamp_queue) > 1:
            sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
        else:
            sampling_rate = np.nan
        self.fs_label.setText(
            'fps: {:.3f}'.format(round(sampling_rate, config_ui.sampling_rate_decimal_places)))
        self.ts_label.setText('timestamp: {:.3f}'.format(timestamp))

        if not self.is_image_fitted_to_frame:
            self.plot_widget.setXRange(0, display_image.shape[0])
            self.plot_widget.setYRange(0, display_image.shape[1])
            self.is_image_fitted_to_frame = True

        data_dict = {"stream_name": self.stream_name, "frames": np.expand_dims(display_image.reshape(-1), -1), "timestamps": np.array([timestamp])}
        self.main_parent.scripting_tab.forward_data(data_dict)
        self.main_parent.recording_tab.update_camera_screen_buffer(self.stream_name, cam_id_cv_img_timestamp["frame"], timestamp)

    def video_preset_changed(self):
        self.worker.video_scale = get_video_scale(self.stream_name)
        self.worker.channel_order = get_video_channel_order(self.stream_name)
        self.is_image_fitted_to_frame = False

