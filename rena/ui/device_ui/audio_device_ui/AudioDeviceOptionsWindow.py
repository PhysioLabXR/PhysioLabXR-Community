from PyQt5 import QtCore
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QIcon
from PyQt5.QtWidgets import QPushButton, QWidget

from rena import config
from rena.config import app_logo_path
from rena.config_ui import *
from rena.presets.GroupEntry import PlotFormat
from rena.presets.presets_utils import get_stream_preset_info, set_stream_preset_info
from rena.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from rena.ui.StreamGroupView import StreamGroupView
from rena.ui_shared import num_points_shown_text
from rena.utils.ui_utils import dialog_popup
from rena.ui.device_ui.DeviceSettingsWindow import DeviceOptionsWindow


class AudioDeviceOptionsWindow(DeviceOptionsWindow):
    def __init__(self, stream_name, parent_widget):

        super().__init__(stream_name, parent_widget)
        self.ui = uic.loadUi("ui/device_ui/audio_device_ui/AudioDeviceOptionsWindow.ui", self)

