import itertools
import os
import secrets
import string
import sys
import threading
import time
from collections import defaultdict
from multiprocessing import Process
from typing import Union, Iterable

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget
from matplotlib import pyplot as plt
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import stream_availability_wait_time
from rena.startup import load_settings
from rena.tests.TestStream import LSLTestStream
from rena.utils.ui_utils import CustomDialog


def app_fixture(qtbot, show_window=True, revert_to_default=True, reload_presets=True):
    print('Initializing test fixture for ' + 'Visualization Features')
    update_test_cwd()
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initializationfix type error
    load_settings(revert_to_default=revert_to_default, reload_presets=reload_presets)  # load the default settings
    test_renalabapp_main_window = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    if show_window:
        test_renalabapp_main_window.show()
    qtbot.addWidget(test_renalabapp_main_window)

    return app, test_renalabapp_main_window

def stream_is_available(app: MainWindow, test_stream_name: str):
    # print(f"Stream name {test_stream_name} availability is {app.stream_widgets[test_stream_name].is_stream_available}")
    assert app.stream_widgets[test_stream_name].is_stream_available

def handle_custom_dialog_ok(qtbot, patience_second=0, click_delay_second=0):
    if patience_second == 0:
        w = QtWidgets.QApplication.activeWindow()
        if isinstance(w, CustomDialog):
            yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=int(click_delay_second * 1e3))
    else:
        time_started = time.time()
        while not isinstance(w := QtWidgets.QApplication.activeWindow(), CustomDialog):
            time_waited = time.time() - time_started
            if time_waited > patience_second:
                raise TimeoutError
            qtbot.wait(100)  # wait for 100 ms between tries
            print(f"Waiting for the activate window to be a CustomDialog: {w}")
        print(f": {w} is a CustomDialog, trying to click ok button")
        yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=int(click_delay_second * 1e3))
def handle_current_dialog_ok(app: MainWindow, qtbot: QtBot, patience_second=0, click_delay_second=0):
    """
    This is compatible with CustomDialogue creation that also sets the current_dialog in MainWindow
    @param app:
    @param qtbot:
    @param patience_second:
    @param delay:
    """
    if patience_second == 0:
        if isinstance(app.current_dialog, CustomDialog):
            yes_button = app.current_dialog.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=int(click_delay_second * 1e3))  # delay 1 second for the data to come in
        else:
            raise ValueError(f"current dialog in main window is not CustomDialog. It is {type(app.current_dialog)}")
    else:
        time_started = time.time()
        while not isinstance(app.current_dialog, CustomDialog):
            time_waited = time.time() - time_started
            if time_waited > patience_second:
                raise TimeoutError
            qtbot.wait(100)  # wait for 100 ms between tries
            print(f"Waiting for the current dialogue to be a CustomDialog: {app.current_dialog}")
        print(f": {app.current_dialog} is a CustomDialog, trying to click ok button")
        yes_button = app.current_dialog.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=click_delay_second * 1e3)

class TestContext:
    """
    Helper class for carrying out the most performed actions in the tests

    """
    def __init__(self, app: MainWindow, qtbot: QtBot):
        self.send_data_processes = {}
        self.app = app
        self.qtbot = qtbot

        self.stream_availability_timeout = 4 * stream_availability_wait_time * 1e3

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
        self.app.create_preset(stream_name, 'float', None, 'LSL', num_channels=num_channels, nominal_sample_rate=srate)  # add a default preset

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

    def stop_recording(self):
        if not self.app.recording_tab.is_recording:
            raise ValueError("App is not recording when calling stop_recording from test_context")
        t = threading.Timer(1, lambda: handle_current_dialog_ok(app=self.app, qtbot=self.qtbot, patience_second=30000))
        t.start()
        self.qtbot.mouseClick(self.app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording
        t.join()  # wait until the dialog is closed

    def start_recording(self):
        if self.app.recording_tab.is_recording:
            raise ValueError("App is already recording when calling stop_recording from test_context")
        self.app.settings_tab.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)
        self.app.ui.tabWidget.setCurrentWidget(self.app.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
        self.qtbot.mouseClick(self.app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

    def start_streams_and_recording(self, num_stream_to_test: int, num_channels: Union[int, Iterable[int]]=1, sampling_rate: Union[int, Iterable[int]]=1, stream_availability_timeout=2 * stream_availability_wait_time * 1e3):
        """
        start a given number of streams with given number of channels and sampling rate, and start recording.
        @param num_stream_to_test: int, the number of streams to test
        @param num_channels: int or iterable of int, the number of channels in the stream
        @rtype: object
        """

        if isinstance(num_channels, int):
            num_channels = [num_channels] * num_stream_to_test
        if isinstance(sampling_rate, int):
            sampling_rate = [sampling_rate] * num_stream_to_test

        test_stream_names = []
        ts_names = get_random_test_stream_names(num_stream_to_test)
        for i, ts_name in enumerate(ts_names):
            test_stream_names.append(ts_name)
            self.start_stream(ts_name, num_channels[i], sampling_rate[i])
        self.start_recording()
        return test_stream_names

    def get_active_send_data_stream_names(self):
        return [stream_name for stream_name, process in self.send_data_processes.items() if process.is_alive()]


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
    if os.getcwd().endswith(os.path.join('rena', 'tests')):
        os.chdir('../')

def run_benchmark(test_context, test_stream_names, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False):
    results = defaultdict(defaultdict(dict).copy)  # use .copy for pickle friendly one-liner
    for stream_name, (num_channels, sampling_rate) in zip(test_stream_names, itertools.product(num_channels_to_test, sampling_rates_to_test)):
        print(f"Testing #channels {num_channels} and srate {sampling_rate} with random stream name {stream_name}...", end='')
        start_time = time.perf_counter()
        test_context.start_stream(stream_name, num_channels, sampling_rate)
        if is_reocrding:
            test_context.app_main_window.settings_tab.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)
            test_context.qtbot.mouseClick(test_context.app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

        test_context.qtbot.wait(int(test_time_second_per_stream * 1e3))
        test_context.close_stream(stream_name)

        for measure in metrics:
            if measure == 'update buffer time':
                update_buffer_time_mean = np.mean(test_context.app_main_window.stream_widgets[stream_name].update_buffer_times)
                update_buffer_time_std = np.std(test_context.app_main_window.stream_widgets[stream_name].update_buffer_times)
                if np.isnan(update_buffer_time_mean) or np.isnan(update_buffer_time_std):
                    raise ValueError()
                results[measure][num_channels, sampling_rate][measure] = update_buffer_time_mean
                # results[measure][num_channels, sampling_rate]['update_buffer_time_std'] = update_buffer_time_std
            elif measure == 'plot data time':
                plot_data_time_mean = np.mean(test_context.app_main_window.stream_widgets[stream_name].plot_data_times)
                plot_data_time_std = np.std(test_context.app_main_window.stream_widgets[stream_name].plot_data_times)
                if np.isnan(plot_data_time_mean) or np.isnan(plot_data_time_std):
                    raise ValueError()
                results[measure][num_channels, sampling_rate][measure] = plot_data_time_mean
                # results[measure][num_channels, sampling_rate]['plot_data_time_std'] = plot_data_time_std
            elif measure == 'viz fps':
                results[measure][num_channels, sampling_rate][measure] = test_context.app_main_window.stream_widgets[stream_name].get_fps()
            else:
                raise ValueError(f"Unknown metric: {measure}")
        if is_reocrding:
            test_context.stop_recording()
            recording_file_name = test_context.app_main_window.recording_tab.save_path
            assert os.stat(recording_file_name).st_size != 0  # make sure recording file has content
            os.remove(recording_file_name)

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


