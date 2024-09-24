from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.presets.presets_utils import set_video_scale, set_video_channel_order
from physiolabxr.ui.ScriptConsoleLog import ScriptConsoleLog
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel


class VideoDeviceOptions(QWidget):
    def __init__(self, parent_stream_widget):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_DSI24_Options, self)
        window_icon = QIcon(AppConfigs()._app_logo)
        self.setWindowIcon(window_icon)
        self.setWindowTitle('Options for DSI24')

        self.console_log_widget = ScriptConsoleLog(self)
        self.console_message_widget.layout().addWidget(self.console_log_widget)

    def update_video_scale(self):
        set_video_scale(self.video_device_name, self.slider_video_scale.value() / 100)
        self.parent.video_preset_changed()

    def update_channel_order(self):
        set_video_channel_order(self.video_device_name, VideoDeviceChannelOrder[self.channel_order_combobox.currentText()])
        self.parent.video_preset_changed()