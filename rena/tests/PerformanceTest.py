"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""
import itertools
import math
# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
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
from rena.tests.test_utils import TestContext
from rena.utils.data_utils import RNStream
from rena.utils.settings_utils import create_default_preset
from rena.utils.ui_utils import CustomDialog


@pytest.fixture
def app(qtbot: QtBot):
    print('Initializing test fixture for ' + 'Visualization Features')
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


def test_stream_visualization_single_stream_performance(app, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName0'
    test_time_second_per_stream = 10
    sampling_rates_to_test = np.linspace(1, 2048, 1)
    num_channels_to_test = np.linspace(1, 500, 1)

    num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
    sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]
    num_tests = len(sampling_rates_to_test) * len(num_channels_to_test)
    print(f"Testing performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {num_tests * (test_time_second_per_stream + 3)}")

    test_space = itertools.product(num_channels_to_test, sampling_rates_to_test)
    result_update_buffer_time = []
    result_plot_data_time = []
    test_context = TestContext(app, qtbot)
    for num_channels, sampling_rate in test_space:
        print(f"Testing #channels {num_channels} and srate {sampling_rate}...", end='')
        start_time = time.perf_counter()
        test_context.start_stream(test_stream_name, num_channels, sampling_rate)
        qtbot.wait(test_time_second_per_stream * 1e3)
        test_context.close_stream(test_stream_name)

        result_update_buffer_time.append((np.mean(app.stream_widgets[test_stream_name].update_buffer_times),
                                          np.std(app.stream_widgets[test_stream_name].update_buffer_times)))
        result_plot_data_time.append((np.mean(app.stream_widgets[test_stream_name].plot_data_times),
                                      np.std(app.stream_widgets[test_stream_name].plot_data_times)))

        test_context.remove_stream(test_stream_name)
        # print(f"Took {timestart_time}.", end='')

    print(test_space)
    print(result_update_buffer_time)
    print(result_plot_data_time)