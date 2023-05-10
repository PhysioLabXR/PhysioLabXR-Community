# This Python file uses the following encoding: utf-8
import time
from collections import deque

import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer

from rena.config import settings
from rena.threadings import workers
from rena.ui.PoppableWidget import Poppable
from rena.ui_shared import remove_stream_icon, \
    options_icon
from rena.utils.ui_utils import dialog_popup, convert_rgb_to_qt_image


class VideoDeviceWidget(Poppable, QtWidgets.QWidget):
    def __init__(self, parent_widget, parent_layout, video_device_name, insert_position=None):
        """

        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(video_device_name, parent_widget, parent_layout, self.remove_video_device)
        self.ui = uic.loadUi("ui/VideoDeviceWidget.ui", self)
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
        self.is_webcam = video_device_name.isnumeric()
        self.video_device_long_name = ('Webcam ' if self.is_webcam else 'Screen Capture ') + str(video_device_name)
        # Connect UIs ##########################################
        self.RemoveVideoBtn.clicked.connect(self.remove_video_device)
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveVideoBtn.setIcon(remove_stream_icon)

        # FPS counter``
        self.tick_times = deque(maxlen=10 * settings.value('video_device_refresh_interval'))

        # worker and worker threads ##########################################
        self.worker_thread = pg.QtCore.QThread(self)

        self.worker = workers.WebcamWorker(cam_id=video_device_name) if self.is_webcam else workers.ScreenCaptureWorker(video_device_name)
        self.worker.change_pixmap_signal.connect(self.visualize)
        self.worker.moveToThread(self.worker_thread)

        # define timer ##########################################
        self.timer = QTimer()
        self.timer.setInterval(settings.value('video_device_refresh_interval'))
        self.timer.timeout.connect(self.ticks)

        self.worker_thread.start()
        self.timer.start()

    def visualize(self, cam_id_cv_img_timestamp):
        self.tick_times.append(time.time())
        cam_id, cv_img, timestamp = cam_id_cv_img_timestamp
        qt_img = convert_rgb_to_qt_image(cv_img)
        self.ImageLabel.setPixmap(qt_img)
        self.main_parent.recording_tab.update_camera_screen_buffer(cam_id, cv_img, timestamp)

    def remove_video_device(self):
        self.timer.stop()
        if self.main_parent.recording_tab.is_recording:
            dialog_popup(msg='Cannot remove stream while recording.')
            return False
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
