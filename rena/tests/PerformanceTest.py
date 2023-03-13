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
from rena.config import lsl_stream_availability_wait_time
from rena.startup import load_settings
from rena.tests.TestStream import LSLTestStream, ZMQTestStream
from rena.tests.test_utils import TestContext, update_test_cwd, get_random_test_stream_names
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
    test_time_second_per_stream = 10
    sampling_rates_to_test = np.linspace(1, 2048, 10)
    num_channels_to_test = np.linspace(1, 500, 10)
    metrics = 'update buffer time', 'plot data time'
    results = defaultdict(list)

    num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
    sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]
    num_tests = len(sampling_rates_to_test) * len(num_channels_to_test)
    test_stream_names = get_random_test_stream_names(num_tests)

    print(f"Testing performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {num_tests * (test_time_second_per_stream + 3)}")

    test_space = list(itertools.product(num_channels_to_test, sampling_rates_to_test))

    test_context = TestContext(app, qtbot)
    for stream_name, (num_channels, sampling_rate) in zip(test_stream_names, test_space):
        print(f"Testing #channels {num_channels} and srate {sampling_rate} with random stream name {stream_name}...", end='')
        start_time = time.perf_counter()
        test_context.start_stream(stream_name, num_channels, sampling_rate)
        qtbot.wait(test_time_second_per_stream * 1e3)
        test_context.close_stream(stream_name)

        for measure in metrics:
            if measure == 'update buffer time':
                update_buffer_time_mean = np.mean(app.stream_widgets[stream_name].update_buffer_times)
                update_buffer_time_std = np.std(app.stream_widgets[stream_name].update_buffer_times)
                if np.isnan(update_buffer_time_mean) or np.isnan(update_buffer_time_std):
                    raise ValueError()
                results[measure].append((update_buffer_time_mean, update_buffer_time_std))
            elif measure == 'plot data time':
                plot_data_time_mean = np.mean(app.stream_widgets[stream_name].plot_data_times)
                plot_data_time_std = np.std(app.stream_widgets[stream_name].plot_data_times)
                if np.isnan(plot_data_time_mean) or np.isnan(plot_data_time_std):
                    raise ValueError()
                results[measure].append((plot_data_time_mean, plot_data_time_std))
            else:
                raise ValueError(f"Unknown metric: {measure}")

        test_context.remove_stream(stream_name)
        print(f"Took {time.perf_counter() - start_time}.", end='')


    # for measure in metrics:
    #     result_matrix = np.zeros((len(sampling_rates_to_test), len(num_channels_to_test), 2))  # last dimension is mean and std
    #     for i, num_channels in enumerate(num_channels_to_test):
    #         for j, sampling_rate in enumerate(sampling_rates_to_test):
    #             result_matrix[i, j] = results[measure][test_space.index((num_channels, sampling_rate))]

    for measure in metrics:
        num_channels_in_test_space = np.array([num_channels for num_channels, _ in test_space])
        means = np.zeros(len(num_channels_to_test))
        for i, num_chan in enumerate(num_channels_to_test):
            means[i] = np.mean(np.array(results[measure])[num_channels_in_test_space == num_chan][:, 0])
        plt.scatter(num_channels_to_test, means)
        plt.plot(num_channels_to_test, means)
        plt.title(f"Rena Benchmark: single stream {measure} across number of channels")
        plt.xlabel("Number of channels")
        plt.ylabel(f'{measure} (seconds)')
        plt.show()

        srates_in_test_space = np.array([srate for _, srate in test_space])
        means = np.zeros(len(sampling_rates_to_test))
        for i, srate in enumerate(sampling_rates_to_test):
            means[i] = np.mean(np.array(results[measure])[srates_in_test_space == srate][:, 0])
        plt.scatter(sampling_rates_to_test, means)
        plt.plot(sampling_rates_to_test, means)
        plt.title(f"Rena Benchmark: single stream {measure} across sampling rates")
        plt.xlabel("Sampling Rate (Hz)")
        plt.ylabel(f'{measure} (seconds)')
        plt.show()

    print(test_space)
    print(results)
    pickle.dump({'results': results, 'test_space': test_space}, open("single_stream_benchmark_results.p", 'wb'))

