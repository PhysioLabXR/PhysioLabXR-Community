# This Python file uses the following encoding: utf-8
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer

from rena.config import settings
from rena.threadings import workers
from rena.ui_shared import pop_window_icon, dock_window_icon, remove_stream_icon, \
    options_icon
from rena.utils.ui_utils import AnotherWindow, dialog_popup, convert_rgb_to_qt_image


class VideoDeviceWidget(QtWidgets.QWidget):
    def __init__(self, main_parent, parent_layout, video_device_name, insert_position=None):
        """

        @param main_parent:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__()
        self.ui = uic.loadUi("ui/VideoDeviceWidget.ui", self)
        if type(insert_position) == int:
            parent_layout.insertWidget(insert_position, self)
        else:
            parent_layout.addWidget(self)
        self.parent_layout = parent_layout
        self.main_parent = main_parent
        self.video_device_name = video_device_name

        # check if the video device is a camera or screen capture ####################################
        self.is_webcam = video_device_name.isnumeric()
        self.video_device_long_name = ('Webcam ' if self.is_webcam else 'Screen Capture ') + str(video_device_name)

        # Connect UIs ##########################################
        self.RemoveVideoBtn.clicked.connect(self.remove_video_device)
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveVideoBtn.setIcon(remove_stream_icon)
        self.set_button_icons()

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
        cam_id, cv_img, timestamp = cam_id_cv_img_timestamp
        qt_img = convert_rgb_to_qt_image(cv_img)
        self.ImageLabel.setPixmap(qt_img)
        self.main_parent.recording_tab.update_camera_screen_buffer(cam_id, cv_img, timestamp)

    def remove_video_device(self):
        self.timer.stop()
        if self.main_parent.recording_tab.is_recording:
            dialog_popup(msg='Cannot remove stream while recording.')
            return False
        self.worker.stop_video()
        self.worker_thread.exit()
        self.worker_thread.wait()  # wait for the thread to exit

        self.main_parent.video_device_widgets.pop(self.video_device_name)
        self.main_parent.remove_stream_widget(self)

        # loop = QEventLoop()
        # QTimer.singleShot(250, loop.quit)
        # loop.exec_()
        # close window if popped
        if self.video_device_name in self.main_parent.pop_windows.keys():
            self.main_parent.pop_windows[self.video_device_name].hide()
            self.deleteLater()
        else:  # use recursive delete if docked
            self.deleteLater()
        return True

    def pop_window(self):
        w = AnotherWindow(self, close_function=self.remove_video_device)
        self.main_parent.pop_windows[self.video_device_name] = w
        w.setWindowTitle(self.video_device_long_name)
        self.PopWindowBtn.setText('Dock Window')
        w.show()
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.dock_window)
        self.set_button_icons()

    def dock_window(self):
        self.parent_layout.insertWidget(self.parent_layout.count() - 1, self)
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.PopWindowBtn.setText('Pop Window')
        self.main_parent.pop_windows[self.video_device_name].hide()
        self.main_parent.pop_windows.pop(self.video_device_name)
        self.set_button_icons()
        self.main_parent.activateWindow()

    def ticks(self):
        self.worker.tick_signal.emit()

    def set_button_icons(self):
        if 'Pop' in self.PopWindowBtn.text():
            self.PopWindowBtn.setIcon(pop_window_icon)
        else:
            self.PopWindowBtn.setIcon(dock_window_icon)