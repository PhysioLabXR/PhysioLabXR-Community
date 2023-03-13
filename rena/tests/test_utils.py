import itertools
import os
import secrets
import string
import time
from collections import defaultdict
from multiprocessing import Process

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget
from matplotlib import pyplot as plt
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import lsl_stream_availability_wait_time
from rena.tests.TestStream import LSLTestStream


def stream_is_available(app: MainWindow, test_stream_name: str):
    # print(f"Stream name {test_stream_name} availability is {app.stream_widgets[test_stream_name].is_stream_available}")
    assert app.stream_widgets[test_stream_name].is_stream_available

class TestContext:
    """
    Helper class for carrying out the most performed actions in the tests

    """
    def __init__(self, app: MainWindow, qtbot: QtBot):
        self.send_data_processes = {}
        self.app = app
        self.qtbot = qtbot

        self.stream_availability_timeout = 4 * lsl_stream_availability_wait_time * 1e3

    def cleanup(self):
        pass

    def start_stream(self, stream_name: str, num_channels: int, srate:int):
        """
        start a stream as a separate process, add it to the app's streams, and start it once it becomes
        available
        @param stream_name:
        @param num_channels:
        """
        if stream_name in self.send_data_processes.keys():
            raise ValueError(f"Stream name {stream_name} is in keys for send_data_processes")
        p = Process(target=LSLTestStream, args=(stream_name, num_channels, srate))
        p.start()
        self.send_data_processes[stream_name] = p
        self.app.create_preset(stream_name, 'float', None, 'LSL', num_channels=num_channels)  # add a default preset

        self.app.ui.tabWidget.setCurrentWidget(self.app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
        self.qtbot.mouseClick(self.app.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
        self.qtbot.keyPress(self.app.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
        self.qtbot.keyClicks(self.app.addStreamWidget.stream_name_combo_box, stream_name)
        self.qtbot.mouseClick(self.app.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

        self.qtbot.waitUntil(lambda: stream_is_available(app=self.app, test_stream_name=stream_name), timeout=self.stream_availability_timeout)  # wait until the LSL stream becomes available
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.LeftButton)


    def close_stream(self, stream_name: str):
        if stream_name not in self.send_data_processes.keys():
            raise ValueError(f"Founding repeating test_stream_name : {stream_name}")
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.LeftButton)
        self.send_data_processes[stream_name].kill()

    def remove_stream(self, stream_name: str):
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].RemoveStreamBtn, QtCore.Qt.LeftButton)

    def clean_up(self):
        [p.kill() for _, p in self.send_data_processes.items()]

    def __del__(self):
        self.clean_up()

def secrets_random_choice(alphabet):
    return ''.join(secrets.choice(alphabet) for _ in range(8))

def get_random_test_stream_names(num_names: int, alphabet = string.ascii_lowercase + string.digits):
    names = []
    for i in range(num_names):
        redraw = True
        while redraw:
            rand_name = secrets_random_choice(alphabet)
            redraw = rand_name in names
        names.append(rand_name)
    return names

def update_test_cwd():
    if os.getcwd().endswith('rena/tests'):
        os.chdir('../')

def run_benchmark(test_context, test_stream_names, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False):
    results = defaultdict(lambda: defaultdict(dict))
    for stream_name, (num_channels, sampling_rate) in zip(test_stream_names, itertools.product(num_channels_to_test, sampling_rates_to_test)):
        print(f"Testing #channels {num_channels} and srate {sampling_rate} with random stream name {stream_name}...", end='')
        start_time = time.perf_counter()
        test_context.start_stream(stream_name, num_channels, sampling_rate)
        test_context.qtbot.wait(test_time_second_per_stream * 1e3)
        test_context.close_stream(stream_name)

        for measure in metrics:
            if measure == 'update buffer time':
                update_buffer_time_mean = np.mean(test_context.app.stream_widgets[stream_name].update_buffer_times)
                update_buffer_time_std = np.std(test_context.app.stream_widgets[stream_name].update_buffer_times)
                if np.isnan(update_buffer_time_mean) or np.isnan(update_buffer_time_std):
                    raise ValueError()
                results[measure][num_channels, sampling_rate][measure] = update_buffer_time_mean
                # results[measure][num_channels, sampling_rate]['update_buffer_time_std'] = update_buffer_time_std
            elif measure == 'plot data time':
                plot_data_time_mean = np.mean(test_context.app.stream_widgets[stream_name].plot_data_times)
                plot_data_time_std = np.std(test_context.app.stream_widgets[stream_name].plot_data_times)
                if np.isnan(plot_data_time_mean) or np.isnan(plot_data_time_std):
                    raise ValueError()
                results[measure][num_channels, sampling_rate][measure] = plot_data_time_mean
                # results[measure][num_channels, sampling_rate]['plot_data_time_std'] = plot_data_time_std
            elif measure == 'viz fps':
                results[measure][num_channels, sampling_rate][measure] = test_context.app.stream_widgets[stream_name].get_fps()
            else:
                raise ValueError(f"Unknown metric: {measure}")

        test_context.remove_stream(stream_name)
        print(f"Took {time.perf_counter() - start_time}.", end='')
    return results


def visualize_benchmark_results(results, test_axes, metrics, notes=''):
    """
    the key for results[measure] are the test axes, these keys must be in the same order as test axes
    @param results:
    @param test_axes:
    @param metrics:
    @return:
    """
    for i, (axis_name, test_variables) in enumerate(test_axes.items()):
        visualize_metric_across_test_space_axis(results, i, axis_name, test_variables, metrics, notes=notes)

    sampling_rates_to_test = test_axes["sampling rate (Hz)"]
    num_channels_to_test = test_axes["number of channels"]
    visualize_metrics_across_num_chan_sampling_rate(results, metrics, sampling_rates_to_test, num_channels_to_test, notes=notes)

def visualize_metrics_across_num_chan_sampling_rate(results, metrics, sampling_rates_to_test, num_channels_to_test, notes=''):
    for measure in metrics:
        result_matrix = np.zeros((len(sampling_rates_to_test), len(num_channels_to_test), 2))  # last dimension is mean and std
        for i, num_channels in enumerate(num_channels_to_test):
            for j, sampling_rate in enumerate(sampling_rates_to_test):
                result_matrix[i, j] = results[measure][num_channels, sampling_rate][measure]
        plt.imshow(result_matrix[:, :, 0])
        plt.xticks(ticks=list(range(len(sampling_rates_to_test))), labels=sampling_rates_to_test)
        plt.yticks(ticks=list(range(len(num_channels_to_test))), labels=num_channels_to_test)
        plt.xlabel("Sampling Rate (Hz)")
        plt.ylabel("Number of channels")
        plt.title(f'Rena Benchmark: single stream: {measure}. {notes}')
        plt.colorbar()
        plt.show()

def visualize_metric_across_test_space_axis(results, axis_index, axis_name, test_variables, metrics, notes=''):
    for measure in metrics:
        means = np.zeros(len(test_variables))
        for j, test_variable_value in enumerate(test_variables):
            this_test_variable_measure_means = [value[measure] for key, value in results[measure].items() if key[axis_index] == test_variable_value]
            means[j] = np.mean(this_test_variable_measure_means)
        plt.scatter(test_variables, means)
        plt.plot(test_variables, means)
        plt.title(f"Rena Benchmark: single stream {measure} across number of channels. {notes}")
        plt.xlabel(axis_name)
        plt.ylabel(f'{measure} (seconds)')
        plt.show()