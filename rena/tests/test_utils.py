import itertools
import os
import pickle
import secrets
import string
import sys
import threading
import time
from collections import defaultdict
from multiprocessing import Process
from typing import Union, Iterable, List

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget
from matplotlib import pyplot as plt
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import stream_availability_wait_time
from rena.startup import load_settings
from rena.sub_process.pyzmq_utils import can_connect_to_port
from rena.tests.TestStream import LSLTestStream, ZMQTestStream
from rena.utils.buffers import flatten
from rena.utils.data_utils import RNStream
from rena.utils.ui_utils import CustomDialog


def app_fixture(qtbot, show_window=True, revert_to_default=True, reload_presets=True):
    print('Initializing test fixture for ' + 'Visualization Features')
    update_test_cwd()
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    load_settings(revert_to_default=revert_to_default, reload_presets=reload_presets)  # load the default settings
    test_renalabapp_main_window = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    if show_window:
        test_renalabapp_main_window.show()
    qtbot.addWidget(test_renalabapp_main_window)

    return app, test_renalabapp_main_window

def stream_is_available(app: MainWindow, test_stream_name: str):
    # print(f"Stream name {test_stream_name} availability is {app.stream_widgets[test_stream_name].is_stream_available}")
    assert app.stream_widgets[test_stream_name].is_stream_available

def streams_are_available(app: MainWindow, test_stream_names: List[str]):
    # print(f"Stream name {test_stream_name} availability is {app.stream_widgets[test_stream_name].is_stream_available}")
    for ts_name in test_stream_names:
        assert app.stream_widgets[ts_name].is_stream_available

def stream_is_unavailable(app_main_window, stream_name):
    assert not app_main_window.stream_widgets[stream_name].is_stream_available

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
    @param app: the main window
    @param qtbot: qtbot instance of the testing fixture
    @param patience_second: how long to wait for the current dialog to be a CustomDialog
    @param delay: how long to wait before clicking the button
    """
    handle_current_dialog_button(QtWidgets.QDialogButtonBox.Ok, app, qtbot, patience_second, click_delay_second)

def handle_current_dialog_button(button, app: MainWindow, qtbot: QtBot, patience_second=0, click_delay_second=0):
    """
    This is compatible with CustomDialogue creation that also sets the current_dialog in MainWindow
    @param app: the main window
    @param qtbot: qtbot instance of the testing fixture
    @param patience_second: how long to wait for the current dialog to be a CustomDialog
    @param delay: how long to wait before clicking the button
    """
    if patience_second == 0:
        if isinstance(app.current_dialog, CustomDialog):
            yes_button = app.current_dialog.buttonBox.button(button)
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
        yes_button = app.current_dialog.buttonBox.button(button)
        qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=click_delay_second * 1e3)

class ContextBot:
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

    def create_zmq_stream(self, stream_name: str, num_channels: int, srate:int, port_range=(5000, 5100)):
        using_port = None
        for port in range(*port_range):
            if not can_connect_to_port(port):
                using_port = port
                break
        if using_port is None:
            raise ValueError(f"Could not find a port in range {port_range}. Consider use a different range.")
        if stream_name in self.send_data_processes.keys():
            raise ValueError(f"Stream name {stream_name} is in keys for send_data_processes")
        p = Process(target=ZMQTestStream, args=(stream_name, using_port, num_channels, srate))
        p.start()
        self.send_data_processes[stream_name] = p
        return using_port

    def close_stream(self, stream_name: str):
        if stream_name not in self.send_data_processes.keys():
            raise ValueError(f"Founding repeating test_stream_name : {stream_name}")
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.LeftButton)
        self.send_data_processes[stream_name].kill()

        self.qtbot.waitUntil(lambda: stream_is_unavailable(self.app, stream_name), timeout=self.stream_availability_timeout)  # wait until the stream becomes unavailable

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
        self.app.settings_widget.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)
        # self.app.ui.tabWidget.setCurrentWidget(self.app.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
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
    print('update_test_cwd: current working directory is', os.getcwd())
    if os.getcwd().endswith(os.path.join('rena', 'tests')):
        os.chdir('../')
    elif 'rena' in os.listdir(os.getcwd()):
        os.chdir('rena')
    # else:
    #     raise Exception('update_test_cwd: RenaLabApp test must be run from either <project_root>/rena/tests or <project_root>. Instead cwd is', os.getcwd())

def run_visualization_benchmark(app_main_window, test_context, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False):
    results = defaultdict(defaultdict(dict).copy)  # use .copy for pickle friendly one-liner
    for n_streams, num_channels, sampling_rate in itertools.product(num_streams_to_test, num_channels_to_test, sampling_rates_to_test):
        stream_names = [test_stream_names.pop(0) for _ in range(n_streams)]
        print(f"Testing #channels {num_channels} and srate {sampling_rate} with random stream name(s) {stream_names}...", end='')
        start_time = time.perf_counter()
        for s_name in stream_names:
            test_context.start_stream(s_name, num_channels, sampling_rate)
        if is_reocrding:
            app_main_window.settings_widget.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)
            test_context.qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

        test_context.qtbot.wait(int(test_time_second_per_stream * 1e3))
        for s_name in stream_names:
            test_context.close_stream(s_name)
        for measure in metrics:
            if measure == 'update buffer time':
                update_buffer_times = flatten([app_main_window.stream_widgets[s_name].update_buffer_times for s_name in stream_names])
                update_buffer_time_mean = np.mean(update_buffer_times)
                update_buffer_time_std = np.std(update_buffer_times)
                if np.isnan(update_buffer_time_mean) or np.isnan(update_buffer_time_std):
                    raise ValueError()
                results[measure][n_streams, num_channels, sampling_rate][measure] = update_buffer_time_mean
                # results[measure][num_channels, sampling_rate]['update_buffer_time_std'] = update_buffer_time_std
            elif measure == 'plot data time':
                plot_data_times = flatten([app_main_window.stream_widgets[s_name].plot_data_times for s_name in stream_names])
                plot_data_time_mean = np.mean(plot_data_times)
                plot_data_time_std = np.std(plot_data_times)
                if np.isnan(plot_data_time_mean) or np.isnan(plot_data_time_std):
                    raise ValueError()
                results[measure][n_streams, num_channels, sampling_rate][measure] = plot_data_time_mean
                # results[measure][num_channels, sampling_rate]['plot_data_time_std'] = plot_data_time_std
            elif measure == 'viz fps':
                results[measure][n_streams, num_channels, sampling_rate][measure] = np.mean([app_main_window.stream_widgets[s_name].get_fps() for s_name in stream_names])
            else:
                raise ValueError(f"Unknown metric: {measure}")
        [app_main_window.stream_widgets[s_name].reset_performance_measures() for s_name in stream_names]

        if is_reocrding:
            test_context.stop_recording()
            recording_file_name = app_main_window.recording_tab.save_path
            assert os.stat(recording_file_name).st_size != 0  # make sure recording file has content
            os.remove(recording_file_name)
        for s_name in stream_names:
            test_context.remove_stream(s_name)
        print(f"Took {time.perf_counter() - start_time}.", end='')

    return results


def plot_viz_benchmark_results(results, test_axes, metrics, notes=''):
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
    num_streams_to_test = test_axes["number of streams"]

    for n_streams in num_streams_to_test:
        visualize_metrics_across_num_chan_sampling_rate(results, metrics, n_streams, sampling_rates_to_test, num_channels_to_test, notes=notes)

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

def visualize_metrics_across_num_chan_sampling_rate(results, metrics, number_of_streams, sampling_rates_to_test, num_channels_to_test, notes=''):
    for measure in metrics:
        result_matrix = np.zeros((len(sampling_rates_to_test), len(num_channels_to_test), 2))  # last dimension is mean and std
        for i, num_channels in enumerate(num_channels_to_test):
            for j, sampling_rate in enumerate(sampling_rates_to_test):
                result_matrix[i, j] = results[measure][number_of_streams, num_channels, sampling_rate][measure]
        plt.imshow(result_matrix[:, :, 0], cmap='plasma')
        plt.xticks(ticks=list(range(len(sampling_rates_to_test))), labels=sampling_rates_to_test)
        plt.yticks(ticks=list(range(len(num_channels_to_test))), labels=num_channels_to_test)
        plt.xlabel("Sampling Rate (Hz)")
        plt.ylabel("Number of channels")
        plt.title(f'{number_of_streams} stream{"s" if number_of_streams > 1 else ""}: {measure}. {notes}')
        plt.colorbar()
        plt.show()


def run_replay_benchmark(app_main_window, test_context: ContextBot, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, results_path):
    results = defaultdict(defaultdict(dict).copy)  # use .copy for pickle friendly one-liner
    start_time = time.perf_counter()
    test_axes = {"number of streams": num_streams_to_test, "number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}

    for n_streams, num_channels, sampling_rate in itertools.product(num_streams_to_test, num_channels_to_test, sampling_rates_to_test):
        this_stream_names = [test_stream_names.pop(0) for _ in range(n_streams)]
        print(f"Testing #channels {num_channels} and srate {sampling_rate} with random stream name(s) {this_stream_names}...", end='')

        # play streams and record them #################################################################################
        # start the designated number of streams with given number of channels and sampling rate
        for s_name in this_stream_names:
            test_context.start_stream(s_name, num_channels, sampling_rate)
        test_context.start_recording()

        # wait for experiment time
        test_context.qtbot.wait(int(test_time_second_per_stream * 1e3))

        # close the all the sterams
        for s_name in this_stream_names:
            test_context.close_stream(s_name)
        test_context.stop_recording()

        # replay the streams ##########################################################################################
        recording_file_name = app_main_window.recording_tab.save_path
        data_original = RNStream(recording_file_name).stream_in()  # this original data will be compared with replayed data later

        app_main_window.replay_tab.select_file(recording_file_name)
        test_context.qtbot.mouseClick(app_main_window.replay_tab.StartStopReplayBtn, QtCore.Qt.LeftButton)  # stop the recording
        test_context.qtbot.waitUntil(lambda: streams_are_available(app_main_window, this_stream_names), timeout=test_context.stream_availability_timeout)  # wait until the streams becomes available from replay
        # start the streams from replay and record them ################################################
        for ts_name in this_stream_names:
            test_context.qtbot.mouseClick(app_main_window.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.LeftButton)
        test_context.start_recording()
        test_context.qtbot.wait(int(test_time_second_per_stream * 1e3))

        # test if the data are being received as they are being replayed
        for ts_name in this_stream_names:
            assert app_main_window.stream_widgets[ts_name].viz_data_buffer.has_data()
        test_context.qtbot.waitUntil(lambda: not app_main_window.replay_tab.is_replaying, timeout=test_time_second_per_stream * 1e3)  # wait until the replay completes, need to ensure that the replay can finish

        # remove the streams from the stream widgets
        test_context.stop_recording()
        for ts_name in this_stream_names:
            test_context.remove_stream(ts_name)

        # load the replay file
        replayed_file_name = app_main_window.recording_tab.save_path
        data_replayed = RNStream(replayed_file_name).stream_in()  # this original data will be compared with replayed data later

        for measure in metrics:
            if measure == 'replay push data loop time':
                average_loop_time = app_main_window.replay_tab._request_replay_performance()
                if average_loop_time == 0:
                    raise ValueError()
                results[measure][n_streams, num_channels, sampling_rate][measure] = average_loop_time
            elif measure == 'timestamp reenactment accuracy':
                results[measure][n_streams, num_channels, sampling_rate][measure] = get_replay_time_reenactment_accuracy(data_original, data_replayed, this_stream_names)
            else:
                raise ValueError(f"Unknown metric: {measure}")

        os.remove(replayed_file_name)
        os.remove(recording_file_name)
        pickle.dump({'results': results, 'test_axes': test_axes}, open(results_path, 'wb'))

    print(f"Took {time.perf_counter() - start_time}.", end='')

    return results

def get_replay_time_reenactment_accuracy(data_original, data_replayed, stream_names):
    tick_time_discrepancies = np.empty(0)
    for s_name in stream_names:
        a = data_original[s_name][0]
        b = data_replayed[s_name][0]
        assert np.all(a[:, -b.shape[1]:] == b)

        a = data_original[s_name][1]
        b = data_replayed[s_name][1]
        c = a[-b.shape[0]:]

        d = np.diff(c)
        e = np.diff(b)

        tick_time_discrepancies = np.concatenate([tick_time_discrepancies, np.abs(e - d).flatten()])
    return tick_time_discrepancies


def plot_replay_benchmark_results(results, test_axes, metrics, notes=''):
    """
    the key for results[measure] are the test axes, these keys must be in the same order as test axes
    @param results:
    @param test_axes:
    @param metrics:
    @return:
    """
    sampling_rates_to_test = test_axes["sampling rate (Hz)"]
    num_channels_to_test = test_axes["number of channels"]
    num_streams_to_test = test_axes["number of streams"]

    if 'replay push data loop time' in metrics:
        for n_streams in num_streams_to_test:
            visualize_metrics_across_num_chan_sampling_rate(results, ['replay push data loop time'], n_streams, sampling_rates_to_test, num_channels_to_test, notes=notes)

    if 'timestamp reenactment accuracy' in metrics:
        print()
        timestamps_renactments_discrepencies = []
        for n_streams in num_streams_to_test:
            results_this_n_streams = np.empty(0)
            for i, num_channels in enumerate(num_channels_to_test):
                for j, sampling_rate in enumerate(sampling_rates_to_test):
                    results_this_n_streams = np.concatenate([results_this_n_streams, results['timestamp reenactment accuracy'][n_streams, num_channels, sampling_rate]['timestamp reenactment accuracy']])
            timestamps_renactments_discrepencies.append(results_this_n_streams)
        plt.boxplot(timestamps_renactments_discrepencies)
        plt.xlabel('number of streams')
        plt.ylabel('discrepancy between original and replayed stream timestamps (second)')
        plt.show()
