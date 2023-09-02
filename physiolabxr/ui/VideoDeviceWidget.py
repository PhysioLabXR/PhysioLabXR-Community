# This Python file uses the following encoding: utf-8
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer

import physiolabxr.threadings.ScreenCaptureWorker
import physiolabxr.threadings.WebcamWorker
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.presets_utils import get_video_scale, get_video_channel_order, is_video_webcam, get_video_device_id
from physiolabxr.ui.PoppableWidget import Poppable
from physiolabxr.ui.VideoDeviceOptions import VideoDeviceOptions
from physiolabxr.utils.ui_utils import dialog_popup


class VideoDeviceWidget(Poppable, QtWidgets.QWidget):
    def __init__(self, parent_widget, parent_layout, video_device_name, insert_position=None):
        """

        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(video_device_name, parent_widget, parent_layout, self.remove_video_device)
        self.ui = uic.loadUi(AppConfigs()._ui_VideoDeviceWidget, self)
        self.set_pop_button(self.PopWindowBtn)

        if type(insert_position) == int:
            parent_layout.insertWidget(insert_position, self)
        else:
            parent_layout.addWidget(self)
        self.parent_layout = parent_layout
        self.main_parent = parent_widget
        self.video_device_name = video_device_name
        self.VideoDeviceNameLabel.setText(self.video_device_name)

        # check if the video device is a camera or screen capture ####################################
        self.is_webcam = is_video_webcam(self.video_device_name)
        self.video_device_long_name = ('Webcam ' if self.is_webcam else 'Screen Capture ') + str(video_device_name)
        # Connect UIs ##########################################
        self.RemoveVideoBtn.clicked.connect(self.remove_video_device)
        self.OptionsBtn.setIcon(AppConfigs()._icon_options)
        self.RemoveVideoBtn.setIcon(AppConfigs()._icon_remove_stream)

        # FPS counter``
        self.tick_times = deque(maxlen=10 * AppConfigs().video_device_refresh_interval)

        # image label
        self.plot_widget = pg.PlotWidget()
        self.ImageWidget.layout().addWidget(self.plot_widget)
        self.plot_widget.enableAutoRange(enable=False)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        # create options window ##########################################
        self.video_options_window = VideoDeviceOptions(parent_stream_widget=self, video_device_name=self.video_device_name)
        self.video_options_window.hide()
        self.OptionsBtn.clicked.connect(lambda: (self.video_options_window.show(), self.video_options_window.activateWindow()))

        # worker and worker threads ##########################################
        self.worker_thread = pg.QtCore.QThread(self)

        video_scale, channel_order = get_video_scale(self.video_device_name), get_video_channel_order(self.video_device_name)
        if self.is_webcam:
            self.worker = physiolabxr.threadings.WebcamWorker.WebcamWorker(get_video_device_id(video_device_name), video_scale, channel_order)
        else:
            self.worker = physiolabxr.threadings.ScreenCaptureWorker.ScreenCaptureWorker(video_device_name, video_scale, channel_order)
        self.worker.change_pixmap_signal.connect(self.visualize)
        self.worker.moveToThread(self.worker_thread)

        # define timer ##########################################
        self.timer = QTimer()
        self.timer.setInterval(AppConfigs().video_device_refresh_interval)
        self.timer.timeout.connect(self.ticks)

        self.worker_thread.start()
        self.timer.start()

        self.is_image_fitted_to_frame = False


    def visualize(self, cam_id_cv_img_timestamp):
        self.tick_times.append(time.time())
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

    def remove_video_device(self):
        if self.main_parent.recording_tab.is_recording:
            dialog_popup(msg='Cannot remove stream while recording.')
            return False
        self.timer.stop()
        self.worker.stop_stream()
        self.worker_thread.exit()
        self.worker_thread.wait()  # wait for the thread to exit

        self.main_parent.video_device_widgets.pop(self.video_device_name)
        self.main_parent.remove_stream_widget(self)

        # close window if popped
        if self.is_popped:
            self.delete_window()
        self.deleteLater()
        return True

    def ticks(self):
        self.worker.tick_signal.emit()

    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except (ZeroDivisionError, IndexError) as e:
            return 0

    def get_pull_data_delay(self):
        return self.worker.get_pull_data_delay()

    def is_widget_streaming(self):
        return self.worker.is_streaming

    def video_preset_changed(self):
        self.worker.video_scale = get_video_scale(self.video_device_name)
        self.worker.channel_order = get_video_channel_order(self.video_device_name)
        self.is_image_fitted_to_frame = False

    def try_close(self):
        return self.remove_video_device()