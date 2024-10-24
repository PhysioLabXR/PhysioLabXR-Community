from physiolabxr.ui.BaseDeviceOptions import BaseDeviceOptions
import os
import subprocess
from physiolabxr.configs.configs import AppConfigs
from PyQt5 import QtWidgets

class TobiiProFusion_Options(BaseDeviceOptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Connect the Calibration button to the click handler
        self.calibration_btn.clicked.connect(self.calibration_btn_clicked)

    def calibration_btn_clicked(self):
        try:
            if not AppConfigs().tobii_app_path and not self.executable_path_input.text():
                # Warning input your path
                QtWidgets.QMessageBox.warning(self, "Missing Path", "Please input the path to the Tobii application.")
                return
            elif not AppConfigs().tobii_app_path:
                AppConfigs().tobii_app_path = self.executable_path_input.text()

            executable_path = os.path.abspath(
                os.path.join("physiolabxr", "interfaces", "DeviceInterface", "TobiiProFusion", "x64", "Debug",
                             "CallManagerApp.exe"))

            result = subprocess.run([executable_path, AppConfigs().tobii_app_path], capture_output=True, text=True)

            if result.returncode == 0:
                print("Executable ran successfully!")
                print("Output:", result.stdout)
            else:
                print(f"Executable failed with error code: {result.returncode}")
                print("Error:", result.stderr)

        except Exception as e:
            print(f"An error occurred: {e}")

    def start_stream_args(self):
        return {}
