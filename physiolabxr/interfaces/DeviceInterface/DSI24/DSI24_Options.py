from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.presets.presets_utils import set_video_scale, set_video_channel_order
from physiolabxr.ui.ScriptConsoleLog import ScriptConsoleLog
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel


class DSI24_Options(QWidget):
    def __init__(self, parent_device_widget, device_worker):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_DSI24_Options, self)
        window_icon = QIcon(AppConfigs()._app_logo)
        self.setWindowIcon(window_icon)
        self.setWindowTitle('Options for DSI24')

        self.parent_device_widget = parent_device_widget  # this is not used in this class
        self.device_worker = device_worker
        self.device_worker.signal_data.connect(self.process_data)

        self.console_log_widget = ScriptConsoleLog(self)
        self.console_message_widget.layout().addWidget(self.console_log_widget)

        # the signal_data may be emitted console log, as defined in DSI24Interface

    def update_video_scale(self):
        set_video_scale(self.video_device_name, self.slider_video_scale.value() / 100)
        self.parent.video_preset_changed()

    def update_channel_order(self):
        set_video_channel_order(self.video_device_name, VideoDeviceChannelOrder[self.channel_order_combobox.currentText()])
        self.parent.video_preset_changed()

    def process_data(self, data):
        """
        The message is defined in DSI24_process.py
        """
        if 'i' in data:
            self.console_log_widget.print_msg('info', data['message'])