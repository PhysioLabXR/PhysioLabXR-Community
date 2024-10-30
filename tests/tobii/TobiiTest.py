import os
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtCore
from unittest.mock import patch
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.interfaces.DeviceInterface.TobiiProFusion.TobiiProFusion_Options import TobiiProFusion_Options


@pytest.fixture
def tobii_options(qtbot):
    # Set the path to the UI file
    setattr(AppConfigs(), '_ui_TobiiProFusion_Options',
            r'C:\Users\Zeyi Tong\Desktop\PhysioLabXR\PhysioLabXR-Community\physiolabxr\_ui\TobiiProFusion_Options.ui')

    options = TobiiProFusion_Options(stream_name="TobiiProFusion", device_interface=None)
    options.show()
    qtbot.addWidget(options)

    yield options


def test_open_calibration(qtbot, tobii_options):
    qtbot.wait(8000)  # Adjust as needed for testing
    print("UI is open for manual testing.")

