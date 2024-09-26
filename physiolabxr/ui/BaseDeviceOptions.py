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

        try:
            ui_file = AppConfigs().__getattribute__(f'_ui_{stream_name}_Options')
        except AttributeError:
            raise AttributeError(f'No UI file is defined for {stream_name} at physiolabxr/_ui/{stream_name}_Options.ui')
        self.ui = uic.loadUi(ui_file, self)

        self.setWindowTitle(f'Options for {stream_name}')

    def start_stream_args(self):
        """ Get the arguments to start the stream.

        This method is called when the user clicks the Start Stream button. You can override this function
        in your device options UI class to return the arguments to start the stream.

        the arguments here will be passed to the start_stream method of the device interface that you defined

        Returns:
            dict: The arguments to start the stream.

        Example:

            if you define a custom device interface like this:

            class MyAwesomeDevice(DeviceInterface):
                ...
                def start_stream(self, com_port, baud_rate):
                    pass

            then you can define the start_stream_args method in your custom device options UI class like this:

            class MyAwesomeDeviceOptions(BaseDeviceOptions):
                ...
                def start_stream_args(self):
                    return {
                        'com_port': self.com_port_line_edit.value(),
                        'baud_rate': self.baud_rate_line_edit.value()
                    }

            , given that you have com_port_line_edit and baud_rate_line_edit in your UI file. User can
            enter the values in the UI and when they click the Start Stream button, the values will be passed
            to the start_stream method of the device interface.
        """
        return {}


