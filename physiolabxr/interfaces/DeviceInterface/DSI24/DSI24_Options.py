from physiolabxr.ui.BaseDeviceOptions import BaseDeviceOptions


class DSI24_Options(BaseDeviceOptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_impedance_btn.clicked.connect(self.check_impedance_btn_clicked)

    def check_impedance_btn_clicked(self):
        raise NotImplementedError