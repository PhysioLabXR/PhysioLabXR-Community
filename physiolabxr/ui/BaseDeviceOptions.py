from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import VideoDeviceChannelOrder
from physiolabxr.presets.presets_utils import set_video_scale, set_video_channel_order
from physiolabxr.ui.ScriptConsoleLog import ScriptConsoleLog
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel


class BaseDeviceOptions(QWidget):
    """ Base class for device options UI. This class is inherited by the custom device options UI classes.

    This class has handles for
    * the device interface you defined for your device

    """
    def __init__(self, stream_name, device_interface):
        super().__init__()
        self.stream_name = stream_name
        self.device_interface = device_interface  # this is not used in this class
        window_icon = QIcon(AppConfigs()._app_logo)
        self.setWindowIcon(window_icon)
        self.setWindowTitle(f'Options for {stream_name}')

        # get the _ui_{stream_name}}_Options from AppConfigs()

        try:
            ui_file = AppConfigs().__getattribute__(f'_ui_{stream_name}_Options')
        except AttributeError:
            raise AttributeError(f'No UI file is defined for {stream_name} at physiolabxr/_ui/{stream_name}_Options.ui')
        self.ui = uic.loadUi(ui_file, self)

