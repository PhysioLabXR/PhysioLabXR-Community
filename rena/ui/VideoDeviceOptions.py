from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from rena.config import app_logo_path
from rena.presets.Presets import VideoDeviceChannelOrder
from rena.presets.presets_utils import set_video_scale, set_video_channel_order
from rena.ui.SliderWithValueLabel import SliderWithValueLabel


class VideoDeviceOptions(QWidget):
    def __init__(self, parent_stream_widget, video_device_name):
        super().__init__()
        self.ui = uic.loadUi("ui/VideoDeviceOptions.ui", self)

        window_icon = QIcon(app_logo_path)
        self.setWindowIcon(window_icon)
        self.setWindowTitle('Options for {}'.format(video_device_name))

        self.parent = parent_stream_widget
        self.video_device_name = video_device_name

        self.slider_video_scale = SliderWithValueLabel(minimum=10, maximum=100, value=100)
        self.main_widget.layout().addWidget(self.slider_video_scale, 0, 1)
        self.slider_video_scale.valueChanged.connect(self.update_video_scale)

        channel_order_names = [name for name, _ in VideoDeviceChannelOrder.__members__.items()]
        self.channel_order_combobox.addItems(channel_order_names)
        self.channel_order_combobox.currentTextChanged.connect(self.update_channel_order)

    def update_video_scale(self):
        set_video_scale(self.video_device_name, self.slider_video_scale.value() / 100)
        self.parent.video_preset_changed()

    def update_channel_order(self):
        set_video_channel_order(self.video_device_name, VideoDeviceChannelOrder[self.channel_order_combobox.currentText()])
        self.parent.video_preset_changed()