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
from rena.tests.test_utils import ContextBot, update_test_cwd, get_random_test_stream_names, run_visualization_benchmark, \
    visualize_benchmark_results, app_fixture
from rena.utils.data_utils import RNStream
from rena.presets.presets_utils import create_default_preset
from rena.utils.ui_utils import CustomDialog


@pytest.fixture
def app_main_window(qtbot):
    app, test_renalabapp_main_window = app_fixture(qtbot)
    yield test_renalabapp_main_window
    app.quit()

@pytest.fixture
def context_bot(app_main_window, qtbot):
    test_context = ContextBot(app=app_main_window, qtbot=qtbot)
    yield test_context
    test_context.clean_up()


def test_stream_visualization_single_stream_performance(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_time_second_per_stream = 60
    num_streams_to_test = [1, 3, 7]
    sampling_rates_to_test = np.linspace(1, 2048, 10)
    num_channels_to_test = np.linspace(1, 128, 10)
    metrics = 'update buffer time', 'plot data time', 'viz fps'

    num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
    sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

    test_axes = {"number of streams": num_streams_to_test, "number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}

    num_tests = len(num_streams_to_test) * len(sampling_rates_to_test) * len(num_channels_to_test)
    test_stream_names = get_random_test_stream_names(np.sum([n_stream * len(sampling_rates_to_test) * len(num_channels_to_test) for n_stream in num_streams_to_test]))

    print(f"Testing performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {num_tests * (test_time_second_per_stream + 3)} seconds.")

    test_context = ContextBot(app_main_window, qtbot)

    results_without_recording = run_visualization_benchmark(app_main_window, test_context, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False)
    pickle.dump({'results_without_recording': results_without_recording, 'test_axes': test_axes}, open("single_stream_benchmark.p", 'wb'))

    visualize_benchmark_results(results_without_recording, test_axes=test_axes, metrics=metrics, notes="")



