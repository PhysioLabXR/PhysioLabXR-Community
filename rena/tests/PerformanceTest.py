"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""
import itertools
import math
# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import pickle
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from multiprocessing import Process

import numpy as np
import matplotlib.pyplot as plt
import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QMessageBox
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import stream_availability_wait_time
from rena.startup import load_settings
from rena.tests.TestStream import LSLTestStream, ZMQTestStream
from rena.tests.test_utils import TestContext, update_test_cwd, get_random_test_stream_names, run_benchmark, \
    visualize_benchmark_results
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


def test_stream_visualization_single_stream_performance(app, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_time_second_per_stream = 15
    sampling_rates_to_test = np.linspace(1, 2048, 10)
    num_channels_to_test = np.linspace(1, 500, 10)
    metrics = 'update buffer time', 'plot data time', 'viz fps'

    num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
    sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

    test_axes = {"number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}

    num_tests = len(sampling_rates_to_test) * len(num_channels_to_test)
    test_stream_names = get_random_test_stream_names(num_tests * 2)

    print(f"Testing performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {num_tests * (test_time_second_per_stream + 3)}")

    test_context = TestContext(app, qtbot)

    # test without recording
    results_with_recording = run_benchmark(test_context, test_stream_names[:int(len(test_stream_names)/2)], num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=True)
    visualize_benchmark_results(results_with_recording, test_axes=test_axes, metrics=metrics, notes="With recording")

    results_without_recording = run_benchmark(test_context, test_stream_names[int(len(test_stream_names)/2):], num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False)
    visualize_benchmark_results(results_without_recording, test_axes=test_axes, metrics=metrics, notes="Without recording")

    pickle.dump({'results_with_recording': results_with_recording, 'results_without_recording': results_without_recording, 'test_axes': test_axes}, open("single_stream_benchmark.p", 'wb'))


