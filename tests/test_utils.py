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
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import QWidget, QDialogButtonBox
from pytestqt.qtbot import QtBot

from physiolabxr.presets.PresetEnums import DataType, PresetType
from tests.TestStream import LSLTestStream, ZMQTestStream, SampleDefinedLSLStream, SampleDefinedZMQStream
from tests.test_viz import visualize_metrics_across_num_chan_sampling_rate


def app_fixture(qtbot, show_window=True, revert_to_default=True, reload_presets=True):
    print('Initializing test fixture for ' + 'Visualization Features')
    # update_test_cwd()
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    from physiolabxr.startup.startup import load_settings
    load_settings(revert_to_default=revert_to_default, reload_presets=reload_presets)  # load the default settings
    from physiolabxr.startup.startup import apply_patches
    apply_patches()
    from physiolabxr.ui.MainWindow import MainWindow
    test_renalabapp_main_window = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    if show_window:
        test_renalabapp_main_window.show()
    qtbot.addWidget(test_renalabapp_main_window)

    return app, test_renalabapp_main_window

def stream_is_available(app, test_stream_name: str):
    # print(f"Stream name {test_stream_name} availability is {app.stream_widgets[test_stream_name].is_stream_available}")
    assert app.stream_widgets[test_stream_name].is_stream_available

def streams_are_available(app, test_stream_names: List[str]):
    # print(f"Stream name {test_stream_name} availability is {app.stream_widgets[test_stream_name].is_stream_available}")
    for ts_name in test_stream_names:
        assert app.stream_widgets[ts_name].is_stream_available

def stream_is_unavailable(app_main_window, stream_name):
    assert not app_main_window.stream_widgets[stream_name].is_stream_available

def handle_custom_dialog_ok(qtbot, patience_second=0, click_delay_second=0):
    from physiolabxr.utils.ui_utils import CustomDialog
    if patience_second == 0:
        w = QtWidgets.QApplication.activeWindow()
        if isinstance(w, CustomDialog):
            yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.MouseButton.LeftButton, delay=int(click_delay_second * 1e3))
    else:
        time_started = time.time()
        while not isinstance(w := QtWidgets.QApplication.activeWindow(), CustomDialog):
            time_waited = time.time() - time_started
            if time_waited > patience_second:
                raise TimeoutError
            qtbot.wait(100)  # wait for 100 ms between tries
            print(f"Waiting for the activate window to be a CustomDialog: {w}")
        print(f": {w} is a CustomDialog, trying to click ok button")
        yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        qtbot.mouseClick(yes_button, QtCore.Qt.MouseButton.LeftButton, delay=int(click_delay_second * 1e3))


def handle_current_dialog_ok(app, qtbot: QtBot, patience_second=0, click_delay_second=0):
    """
    This is compatible with CustomDialogue creation that also sets the current_dialog in MainWindow
    @param app: the main window
    @param qtbot: qtbot instance of the testing fixture
    @param patience_second: how long to wait for the current dialog to be a CustomDialog
    @param delay: how long to wait before clicking the button
    """
    handle_current_dialog_button(QtWidgets.QDialogButtonBox.StandardButton.Ok, app, qtbot, patience_second, click_delay_second)

def handle_current_dialog_button(button, app, qtbot: QtBot, patience_second=0, click_delay_second=0, expected_message_include=None):
    """
    This is compatible with CustomDialogue creation that also sets the current_dialog in MainWindow
    @param app: the main window
    @param qtbot: qtbot instance of the testing fixture
    @param patience_second: how long to wait for the current dialog to be a CustomDialog
    @param delay: how long to wait before clicking the button
    """
    from physiolabxr.utils.ui_utils import CustomDialog

    if expected_message_include is not None:
        qtbot.waitUntil(lambda: expected_message_include in app.current_dialog.msg, timeout=patience_second * 1e3)
    if patience_second == 0:
        if isinstance(app.current_dialog, CustomDialog):
            yes_button = app.current_dialog.buttonBox.button(button)
            qtbot.mouseClick(yes_button, QtCore.Qt.MouseButton.LeftButton, delay=int(click_delay_second * 1e3))  # delay 1 second for the data to come in
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
        qtbot.mouseClick(yes_button, QtCore.Qt.MouseButton.LeftButton, delay=int(click_delay_second * 1e3))

class ContextBot:
    """
    Helper class for carrying out the most performed actions in the tests

    """
    def __init__(self, app, qtbot: QtBot):
        self.send_data_processes = {}
        self.app = app
        self.qtbot = qtbot

        from physiolabxr.configs.config import stream_availability_wait_time
        self.stream_availability_timeout = int(20 * stream_availability_wait_time * 1e3)

        self.monitor_stream_name = "monitor 0"

    def cleanup(self):
        pass

    def create_add_start_stream(self, stream_name: str, num_channels: int, srate:int):
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
        from physiolabxr.presets.PresetEnums import PresetType
        self.app.create_preset(stream_name, PresetType.LSL, num_channels=num_channels, nominal_sample_rate=srate)  # add a default preset

        self.app.ui.tabWidget.setCurrentWidget(self.app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
        self.qtbot.mouseClick(self.app.addStreamWidget.stream_name_combo_box, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
        self.qtbot.keyPress(self.app.addStreamWidget.stream_name_combo_box, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
        self.qtbot.keyClicks(self.app.addStreamWidget.stream_name_combo_box, stream_name)
        self.qtbot.mouseClick(self.app.addStreamWidget.add_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

        self.qtbot.waitUntil(lambda: stream_is_available(app=self.app, test_stream_name=stream_name), timeout=self.stream_availability_timeout)  # wait until the LSL stream becomes available
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)

    def add_and_start_stream(self, stream_name: str, num_channels:int, interface_type=PresetType.LSL, port=None, dtype: DataType=None, *args, **kwargs):
        self.add_stream(stream_name, interface_type, port, dtype, *args, **kwargs)
        self.start_a_stream(stream_name, num_channels, *args, **kwargs)

    def create_add_start_predefined_stream(self, stream_name: str, num_channels: int, srate:int, stream_time:float, dtype: DataType, interface_type=PresetType.LSL, port=None):
        from physiolabxr.presets.Presets import Presets

        samples = np.random.random((num_channels, stream_time * srate)).astype(dtype.get_data_type())
        if interface_type == PresetType.LSL:
            p = Process(target=SampleDefinedLSLStream, args=(stream_name, samples), kwargs={'n_channels': num_channels, 'srate': srate, 'dtype': dtype.get_lsl_type()})
        elif interface_type == PresetType.ZMQ:
            assert port is not None, "port must be specified for ZMQ interface"
            p = Process(target=SampleDefinedZMQStream, args=(stream_name, samples), kwargs={'srate': srate, 'port': port})
        else:
            raise ValueError(f"create_add_start_predefined_stream: interface_type {interface_type} is not supported")
        p.start()
        self.send_data_processes[stream_name] = p
        self.add_stream(stream_name, interface_type=interface_type, port=port, dtype=dtype)
        self.start_a_stream(stream_name, num_channels)

        this_stream_widget = self.app.stream_widgets[stream_name]
        # go to the stream option window and set the sampling rate
        self.qtbot.mouseClick(this_stream_widget.OptionsBtn, QtCore.Qt.MouseButton.LeftButton)
        # check if the option window is open
        assert this_stream_widget.option_window.isVisible()
        self.qtbot.mouseClick(this_stream_widget.option_window.nominalSamplingRateIineEdit, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
        self.qtbot.keyPress(this_stream_widget.option_window.nominalSamplingRateIineEdit, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
        self.qtbot.keyClicks(this_stream_widget.option_window.nominalSamplingRateIineEdit, str(srate))

        # check the sampling rate has been changed in preset
        assert Presets().stream_presets[stream_name].nominal_sampling_rate == srate
        return samples

    def start_a_stream(self, stream_name: str, num_channels: int, thread_timer_second= 4, *args, **kwargs):
        """
        start a stream once it becomes available.
        It also checks if the number of channels in the stream is the same as the number of channels in the preset,
        if not, it clicks the yes button in the dialog box so that it resets the number of channels in the preset
        @param stream_name:
        @param num_channels:
        @return:
        """
        from physiolabxr.presets.Presets import Presets
        self.qtbot.waitUntil(lambda: stream_is_available(app=self.app, test_stream_name=stream_name), timeout=self.stream_availability_timeout)  # wait until the LSL stream becomes available

        if num_channels != Presets().stream_presets[stream_name].num_channels:
            def waitForCurrentDialog():
                assert self.app.current_dialog
            t = threading.Timer(thread_timer_second, lambda: handle_current_dialog_button(QDialogButtonBox.StandardButton.Yes, self.app, self.qtbot, click_delay_second=1, patience_second=4, expected_message_include=stream_name))   # get the messagebox about channel mismatch
            t.start()
            self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)
            self.qtbot.waitUntil(waitForCurrentDialog)
            t.join()
        else:
            self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)

    def add_stream(self, stream_name, interface_type=PresetType.LSL, port=None, dtype=None, *args, **kwargs):
        self.app.ui.tabWidget.setCurrentWidget(self.app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
        self.qtbot.mouseClick(self.app.addStreamWidget.stream_name_combo_box, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
        self.qtbot.keyPress(self.app.addStreamWidget.stream_name_combo_box, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
        self.qtbot.keyClicks(self.app.addStreamWidget.stream_name_combo_box, stream_name)

        preset_type_combo_box = self.app.ui.addStreamWidget.preset_type_combobox
        assert (preset_type_index := preset_type_combo_box.findText(interface_type.name)) != -1
        preset_type_combo_box.setCurrentIndex(preset_type_index)
        # self.qtbot.mouseClick(preset_type_combo_box.view().viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(0, 0))
        # for i in range(preset_type_index + 1):
        #     self.qtbot.keyPress(preset_type_combo_box.view().viewport(), Qt.Key.Key_Down)
        # self.qtbot.keyPress(preset_type_combo_box.view().viewport(), Qt.Key.Key_Return)
        assert preset_type_combo_box.currentText() == interface_type.name

        if interface_type == PresetType.ZMQ:
            assert port is not None, "port must be specified for ZMQ interface"
            assert dtype is not None, "dtype must be specified for ZMQ interface"
            self.qtbot.mouseClick(self.app.addStreamWidget.PortLineEdit, QtCore.Qt.MouseButton.LeftButton)
            self.qtbot.keyPress(self.app.addStreamWidget.PortLineEdit, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
            self.qtbot.keyClicks(self.app.addStreamWidget.PortLineEdit, str(port))
            assert (dtype_index := self.app.addStreamWidget.data_type_combo_box.findText(dtype.name)) != -1
            self.app.addStreamWidget.data_type_combo_box.setCurrentIndex(dtype_index)

        self.qtbot.mouseClick(self.app.addStreamWidget.add_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    def create_zmq_stream(self, stream_name: str, num_channels: int, srate:int, port_range=(5000, 5100)):
        from physiolabxr.sub_process.pyzmq_utils import can_connect_to_port
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
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)
        self.send_data_processes[stream_name].kill()

        self.qtbot.waitUntil(lambda: stream_is_unavailable(self.app, stream_name), timeout=self.stream_availability_timeout)  # wait until the stream becomes unavailable

    def remove_stream(self, stream_name: str):
        self.qtbot.mouseClick(self.app.stream_widgets[stream_name].RemoveStreamBtn, QtCore.Qt.MouseButton.LeftButton)

    def clean_up(self):
        [p.kill() for _, p in self.send_data_processes.items() if p.is_alive()]

    def stop_recording(self):
        if not self.app.recording_tab.is_recording:
            raise ValueError("App is not recording when calling stop_recording from test_context")
        # t = threading.Timer(1, lambda: handle_current_dialog_ok(app=self.app, qtbot=self.qtbot, patience_second=30000))
        # t.start()
        self.qtbot.mouseClick(self.app.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # start the recording
        # t.join()  # wait until the dialog is closed
        ok_button = self.app.current_dialog.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.qtbot.mouseClick(ok_button, QtCore.Qt.MouseButton.LeftButton)

    def start_recording(self):
        if self.app.recording_tab.is_recording:
            raise ValueError("App is already recording when calling stop_recording from test_context")
        self.app.settings_widget.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)
        # self.app._ui.tabWidget.setCurrentWidget(self.app._ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
        self.qtbot.mouseClick(self.app.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # start the recording

    def start_streams_and_recording(self, num_stream_to_test: int, num_channels: Union[int, Iterable[int]]=1, sampling_rate: Union[int, Iterable[int]]=1, stream_availability_timeout=2 * 1e3):
        """
        start a given number of streams with given number of channels and sampling rate, and start recording.
        @param num_stream_to_test: int, the number of streams to test
        @param num_channels: int or iterable of int, the number of channels in the stream
        @rtype: object
        """
        from physiolabxr.configs.config import stream_availability_wait_time
        stream_availability_timeout = stream_availability_timeout * stream_availability_wait_time
        if isinstance(num_channels, int):
            num_channels = [num_channels] * num_stream_to_test
        if isinstance(sampling_rate, int):
            sampling_rate = [sampling_rate] * num_stream_to_test

        test_stream_names = []
        ts_names = get_random_test_stream_names(num_stream_to_test)
        for i, ts_name in enumerate(ts_names):
            test_stream_names.append(ts_name)
            self.create_add_start_stream(ts_name, num_channels[i], sampling_rate[i])
        self.start_recording()
        return test_stream_names

    def get_active_send_data_stream_names(self):
        return [stream_name for stream_name, process in self.send_data_processes.items() if process.is_alive()]

    def connect_to_monitor_0(self):
        self.qtbot.mouseClick(self.app.addStreamWidget.stream_name_combo_box, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
        self.qtbot.keyPress(self.app.addStreamWidget.stream_name_combo_box, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
        self.qtbot.keyClicks(self.app.addStreamWidget.stream_name_combo_box, self.monitor_stream_name)
        self.qtbot.mouseClick(self.app.addStreamWidget.add_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

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

# def update_test_cwd():
#     print('update_test_cwd: current working directory is', os.getcwd())
#     if os.getcwd().endswith('tests'):
#         os.chdir('..')
#     elif 'physiolabxr' in os.listdir(os.getcwd()):
#         os.chdir('physiolabxr')
    # else:
    #     raise Exception('update_test_cwd: RenaLabApp test must be run from either <project_root>/physiolabxr/tests or <project_root>. Instead cwd is', os.getcwd())

def run_visualization_benchmark(app_main_window, test_context, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False):
    from physiolabxr.utils.buffers import flatten

    results = defaultdict(defaultdict(dict).copy)  # use .copy for pickle friendly one-liner
    for n_streams, num_channels, sampling_rate in itertools.product(num_streams_to_test, num_channels_to_test, sampling_rates_to_test):
        stream_names = [test_stream_names.pop(0) for _ in range(n_streams)]
        print(f"Testing #channels {num_channels} and srate {sampling_rate} with random stream name(s) {stream_names}...", end='')
        start_time = time.perf_counter()
        for s_name in stream_names:
            test_context.create_add_start_stream(s_name, num_channels, sampling_rate)
        if is_reocrding:
            app_main_window.settings_widget.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)
            test_context.qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # start the recording

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

        from matplotlib import pyplot as plt
        plt.scatter(test_variables, means)
        plt.plot(test_variables, means)
        plt.title(f"Rena Benchmark: single stream {measure} across number of channels. {notes}")
        plt.xlabel(axis_name)
        plt.ylabel(f'{measure} (seconds)')
        plt.show()


def run_replay_benchmark(app_main_window, test_context: ContextBot, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, results_path):
    from physiolabxr.utils.RNStream import RNStream

    results = defaultdict(defaultdict(dict).copy)  # use .copy for pickle friendly one-liner
    start_time = time.perf_counter()
    test_axes = {"number of streams": num_streams_to_test, "number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}

    for n_streams, num_channels, sampling_rate in itertools.product(num_streams_to_test, num_channels_to_test, sampling_rates_to_test):
        this_stream_names = [test_stream_names.pop(0) for _ in range(n_streams)]
        print(f"test: testing #channels {num_channels} and srate {sampling_rate} with random stream name(s) {this_stream_names}...", end='')

        # play streams and record them #################################################################################
        # start the designated number of streams with given number of channels and sampling rate
        print("test: starting the original streams as separate processes")
        for s_name in this_stream_names:
            test_context.create_add_start_stream(s_name, num_channels, sampling_rate)
        test_context.start_recording()

        # wait for experiment time
        test_context.qtbot.wait(int(test_time_second_per_stream * 1e3))

        # close the all the streams
        for s_name in this_stream_names:
            print(f"test: closing original stream process with name {s_name}")
            test_context.close_stream(s_name)
        test_context.stop_recording()
        print("test: recording stopped for the original streams")

        # replay the streams ##########################################################################################
        print("test: loading back the recorded streams")
        recording_file_name = app_main_window.recording_tab.save_path
        data_original = RNStream(recording_file_name).stream_in()  # this original data will be compared with replayed data later
        print("test: recorded streams successfully loaded. now moving on to replaying")

        app_main_window.replay_tab.select_file(recording_file_name)

        print(f"test: selected file at {recording_file_name} for replaying. starting replay")

        def replay_reloaded():
            assert app_main_window.replay_tab.StartStopReplayBtn.isVisible()
        test_context.qtbot.waitUntil(replay_reloaded, timeout=test_context.stream_availability_timeout)
        test_context.qtbot.mouseClick(app_main_window.replay_tab.StartStopReplayBtn, QtCore.Qt.MouseButton.LeftButton)
        test_context.qtbot.waitUntil(lambda: streams_are_available(app_main_window, this_stream_names), timeout=test_context.stream_availability_timeout)  # wait until the streams becomes available from replay
        # start the streams from replay and record them ################################################
        print("test: replayed streams is now available, start the streams in their StreamWidget")
        for ts_name in this_stream_names:
            test_context.qtbot.mouseClick(app_main_window.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)
        print("test: start recording replayed streams")
        test_context.start_recording()
        test_context.qtbot.wait(int(test_time_second_per_stream * 1e3))

        # test if the data are being received as they are being replayed
        for ts_name in this_stream_names:
            print(f"test: checking if the stream widgets are receiving data from replayed streams. Working on stream name {ts_name}")
            assert app_main_window.stream_widgets[ts_name].viz_data_buffer.has_data()
        test_context.qtbot.waitUntil(lambda: not app_main_window.replay_tab.is_replaying, timeout=int(1.2 * test_time_second_per_stream * 1e3))  # wait until the replay completes, need to ensure that the replay can fin=gesrtnigog mck./
        print("test: stop recording replayed streams")
        test_context.stop_recording()
        # remove the streams from the stream widgets
        for ts_name in this_stream_names:
            test_context.remove_stream(ts_name)
            print(f"test: after replay completes, removed stream widget with stream name {ts_name}")

        # load the replay file
        print("test: loading replayed file")
        replayed_file_name = app_main_window.recording_tab.save_path
        data_replayed = RNStream(replayed_file_name).stream_in()  # this original data will be compared with replayed data later

        print("test: completed loading replayed file, now computing performance metrics")
        for measure in metrics:
            if measure == 'replay push data loop time':
                average_loop_time = app_main_window.replay_tab._request_replay_performance()
                if average_loop_time == 0:
                    raise ValueError()
                results[measure][n_streams, num_channels, sampling_rate][measure] = average_loop_time
            elif measure == 'timestamp reenactment accuracy':
                results[measure][n_streams, num_channels, sampling_rate]['timestamp reenactment accuracy'],\
                    results[measure][n_streams, num_channels, sampling_rate]['timestamp reenactment accuracy pairwise'] = get_replay_time_reenactment_accuracy(data_original, data_replayed, this_stream_names)
            else:
                raise ValueError(f"Unknown metric: {measure}")

        print("test: metrics computed, now removing replayed file")
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
        try:
            assert np.isin(b, a).all()
        except AssertionError:
            print("a and b are not equal, saving them to a_b.pkl")
            pickle.dump({'a': a, 'b': b}, open('a_b.pkl', 'wb'))
            raise AssertionError()

        a = data_original[s_name][1]
        b = data_replayed[s_name][1]
        c = a[-len(b):]

        d = np.diff(c)
        e = np.diff(b)

        tick_time_discrepancies = np.concatenate([tick_time_discrepancies, np.abs(e - d).flatten()])

    # pair-wise synchronization
    tick_time_discrepancies_pairwise = np.empty(0)
    for s_name_anchor in stream_names:
        for s_name_compare in stream_names:
            if s_name_anchor != s_name_compare:
                timestamp_original_anchor = data_original[s_name_anchor][1]
                timestamp_replayed_anchor = data_replayed[s_name_anchor][1]
                timestamp_original_anchor = timestamp_original_anchor[-len(timestamp_replayed_anchor):]

                timestamp_original_compare = data_original[s_name_compare][1]
                timestamp_replayed_compare = data_replayed[s_name_compare][1]
                timestamp_original_compare = timestamp_original_compare[-len(timestamp_replayed_compare):]

                original_length = min(len(timestamp_original_anchor), len(timestamp_original_compare))
                original_pairwise = timestamp_original_anchor[:original_length] - timestamp_original_compare[:original_length]

                replay_length = min(len(timestamp_replayed_anchor), len(timestamp_replayed_compare))
                replay_pairwise = timestamp_replayed_anchor[:replay_length] - timestamp_replayed_compare[:replay_length]

                tick_time_discrepancies_pairwise = np.concatenate([tick_time_discrepancies_pairwise, np.abs(original_pairwise - replay_pairwise).flatten()])

    return tick_time_discrepancies, tick_time_discrepancies_pairwise


