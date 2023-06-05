from rena.ui.StreamWidget import StreamWidget
from rena.ui.device_ui.DeviceSettingsWindow import DeviceOptionsWindow
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QIntValidator


class DeviceWidget(StreamWidget):

    def __init__(self,
                 parent_widget,
                 parent_layout,
                 stream_name,
                 data_type,
                 worker,
                 networking_interface,
                 port_number,
                 insert_position):

        super().__init__(parent_widget,
                         parent_layout,
                         stream_name,
                         data_type,
                         worker,
                         networking_interface,
                         port_number,
                         insert_position)

        # enable device settings widget
        self.DeviceOptionsButtonWidget.show()
        self.device_options_window: DeviceOptionsWindow = DeviceOptionsWindow(stream_name=self.stream_name, parent_widget=self)
        self.device_options_window.show()





    def start_stop_stream_btn_clicked(self):
        # self.init_device_worker()
        super(DeviceWidget, self).start_stop_stream_btn_clicked()

    # update the worker with settings
    def init_device_worker_with_settings(self):
        pass



