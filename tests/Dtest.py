"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""
import faulthandler
# reference https://www.youtube.com/watch?v=WjctCBjHvmA
from multiprocessing import Process

import pytest
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget
from physiolabxr.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from tests.TestStream import LSLTestStream
from tests.test_utils import app_fixture, \
    ContextBot, get_random_test_stream_names


@pytest.fixture
def app_main_window(qtbot):
    print("here")

    app, test_renalabapp_main_window = app_fixture(qtbot)
    faulthandler.disable()  # disable the faulthandler to avoid the error message
    yield test_renalabapp_main_window
    app.quit()


@pytest.fixture
def context_bot(app_main_window, qtbot):
    test_context = ContextBot(app=app_main_window, qtbot=qtbot)
    yield test_context
    test_context.clean_up()

