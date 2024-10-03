from physiolabxr.ui.BaseDeviceOptions import BaseDeviceOptions


class DSI24_Options(BaseDeviceOptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_impedance_chkbx.clicked.connect(self.check_impedance_chkbx_clicked)

    def check_impedance_chkbx_clicked(self):
        # This method will be called when the checkbox is clicked
        if self.check_impedance_chkbx.isChecked():
            print("Check Impedance is checked.")
            # Add logic to handle the case when the checkbox is checked
        else:
            print("Check Impedance is unchecked.")
            # Add logic to handle the case when the checkbox is unchecked

    def start_stream_args(self):
        return {'bluetooth_port': self.device_port_lineedit.text()}
