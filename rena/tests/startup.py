"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
import uuid
from multiprocessing import Process

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QMessageBox
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import lsl_stream_availability_wait_time
from rena.startup import load_settings
from rena.tests.TestStream import LSLTestStream, ZMQTestStream
from rena.tests.test_utils import get_random_test_stream_names, update_test_cwd
from rena.utils.data_utils import RNStream
from rena.utils.settings_utils import create_default_preset
from rena.utils.ui_utils import CustomDialog


@pytest.fixture
def app(qtbot: QtBot):
    print('Initializing test fixture for ' + 'Visualization Features')
    update_test_cwd()
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initialization
    load_settings(revert_to_default=True, reload_presets=True)  # load the default settings
    test_renalabapp = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    test_renalabapp.show()
    qtbot.addWidget(test_renalabapp)
    return test_renalabapp

def teardown_function(function):
    """ teardown any state that was previously setup with a setup_method
    call.
    """
    pass