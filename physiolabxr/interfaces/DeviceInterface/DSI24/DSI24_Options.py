from physiolabxr.ui.BaseDeviceOptions import BaseDeviceOptions


class DSI24_Options(BaseDeviceOptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_impedance_btn.clicked.connect(self.check_impedance_btn_clicked)

    def check_impedance_btn_clicked(self):
        raise NotImplementedError

    def start_stream_args(self):
        return {'bluetooth_port': self.device_port_lineedit.text()}
