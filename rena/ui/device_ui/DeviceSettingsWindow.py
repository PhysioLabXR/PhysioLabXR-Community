from PyQt5.QtWidgets import QWidget


class DeviceOptionsWindow(QWidget):
    def __init__(self, stream_name, parent_widget):
        super().__init__()
        self.stream_name = stream_name
        self.parent_widget = parent_widget

        self.device_options_valid = False
