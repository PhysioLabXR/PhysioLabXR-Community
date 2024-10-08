from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.presets.presets_utils import set_video_scale, set_video_channel_order
from physiolabxr.ui.ScriptConsoleLog import ScriptConsoleLog


class DeviceMessageConsole(ScriptConsoleLog):
    def __init__(self, parent_device_widget, device_worker):
        super().__init__(parent_device_widget)
        window_icon = QIcon(AppConfigs()._app_logo)
        self.setWindowIcon(window_icon)
        self.setWindowTitle(f'Device Message Console: {parent_device_widget.stream_name}')

        self.parent_device_widget = parent_device_widget  # this is not used in this class
        self.device_worker = device_worker
        self.device_worker.signal_data.connect(self.process_message)

        # the signal_data may be emitted console log, as defined in DSI24Interface

    def process_message(self, data):
        """
        The message is defined in DSI24_process.py
        """
        if 'i' in data:
            self.print_msg('info', data['i'])