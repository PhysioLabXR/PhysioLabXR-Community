"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import threading
import time
from multiprocessing import Process

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

from rena.configs.configs import AppConfigs
from rena.presets.Presets import PresetType

AppConfigs(_reset=True)  # create the singleton app configs object
from rena.config import stream_availability_wait_time
from rena.tests.TestStream import LSLTestStream
from rena.tests.test_utils import get_random_test_stream_names, app_fixture, ContextBot
from rena.utils.data_utils import RNStream
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

# def test_plot_format_change(app, qtbot) -> None:
#     '''
#     Adding active stream
#     :param app:
#     :param qtbot:
#     :return:
#     '''
#     test_stream_name = 'TestStreamName'
#     p = Process(target=LSLTestStream, args=(test_stream_name,))
#     p.start()
#     print("Test complete, killing sending-data process")
#     p.kill()  # stop the dummy LSL process


def test_replay_multi_streams(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    num_stream_to_test = 3
    recording_time_second = 6
    replay_file_session_name = 'replayed'
    stream_availability_timeout = 2 * stream_availability_wait_time * 1e3

    test_stream_names = []
    test_stream_processes = []
    ts_names = get_random_test_stream_names(num_stream_to_test)
    for ts_name in ts_names:
        test_stream_names.append(ts_name)
        p = Process(target=LSLTestStream, args=(ts_name,))
        test_stream_processes.append(p)
        p.start()
        app_main_window.create_preset(ts_name, PresetType.LSL, num_channels=81)  # add a default preset

    for ts_name in test_stream_names:
        app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
        qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
        qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
        qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, ts_name)
        qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    app_main_window.settings_widget.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)

    def stream_is_available():
        for ts_name in test_stream_names:
            assert app_main_window.stream_widgets[ts_name].is_stream_available
    def stream_is_unavailable():
        for ts_name in test_stream_names:
            assert not app_main_window.stream_widgets[ts_name].is_stream_available

    qtbot.waitUntil(stream_is_available, timeout=stream_availability_timeout)  # wait until the LSL stream becomes available

    for ts_name in test_stream_names:
        qtbot.mouseClick(app_main_window.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
    qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # start the recording

    qtbot.wait(int(recording_time_second * 1e3))
    # time.sleep(recording_time_second)
    # test if the data are being received
    for ts_name in test_stream_names:
        assert app_main_window.stream_widgets[ts_name].viz_data_buffer.has_data()
    def handle_custom_dialog_ok(patience=0):
        w = QtWidgets.QApplication.activeWindow()
        if patience == 0:
            if isinstance(w, CustomDialog):
                yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
                qtbot.mouseClick(yes_button, QtCore.Qt.MouseButton.LeftButton, delay=1000)  # delay 1 second for the data to come in
        else:
            time_started = time.time()
            while not isinstance(w, CustomDialog):
                time_waited = time.time() - time_started
                if time_waited > patience:
                    raise TimeoutError
                time.sleep(0.5)
            yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.MouseButton.LeftButton, delay=1000)

    t = threading.Timer(1, handle_custom_dialog_ok)
    t.start()
    qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # stop the recording
    t.join()  # wait until the dialog is closed
    #
    qtbot.mouseClick(app_main_window.stop_all_btn, QtCore.Qt.MouseButton.LeftButton)  # stop all the streams, so we don't need to handle stream lost
    #
    print("Waiting for test stream processes to close")
    [p.kill() for p in test_stream_processes]
    qtbot.waitUntil(stream_is_unavailable, timeout=stream_availability_timeout)  # wait until the lsl processes are closed

    # start the streams from replay and record them ################################################
    recording_file_name = app_main_window.recording_tab.save_path
    # load in the data
    data_original = RNStream(recording_file_name).stream_in()  # this original data will be compared with replayed data later
    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'replay_tab'))  # switch to the replay widget
    app_main_window.replay_tab.select_file(recording_file_name)
    qtbot.mouseClick(app_main_window.replay_tab.StartStopReplayBtn, QtCore.Qt.MouseButton.LeftButton)  # stop the recording

    print("Waiting for replay streams to become available")
    qtbot.waitUntil(stream_is_available, timeout=stream_availability_timeout)  # wait until the streams becomes available from replay

    # start the streams from replay and record them ################################################
    for ts_name in test_stream_names:
        qtbot.mouseClick(app_main_window.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)

    # change the recording file name
    qtbot.mouseClick(app_main_window.recording_tab.sessionTagTextEdit, QtCore.Qt.MouseButton.LeftButton)
    qtbot.keyPress(app_main_window.recording_tab.sessionTagTextEdit, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(app_main_window.recording_tab.sessionTagTextEdit, replay_file_session_name)

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
    qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # start the recording

    wait_for_replay_finishes_time = (recording_time_second * 2) * 1e3

    # test if the data are being received as they are being replayed
    for ts_name in test_stream_names:
        assert app_main_window.stream_widgets[ts_name].viz_data_buffer.has_data()

    qtbot.waitUntil(lambda: not app_main_window.replay_tab.is_replaying, timeout=wait_for_replay_finishes_time)  # wait until the replay completes, need to ensure that the replay can finish
    print("replay is over")
    # the streams are stopped at this point

    t = threading.Timer(1, handle_custom_dialog_ok)
    t.start()
    qtbot.mouseClick(app_main_window.recording_tab.StartStopRecordingBtn, QtCore.Qt.MouseButton.LeftButton)  # stop the recording

    # qtbot.mouseClick(app.stop_all_btn, QtCore.Qt.LeftButton)  # stop all the streams

    # replay is completed and the data file saved ################################################
    replayed_file_name = app_main_window.recording_tab.save_path
    data_replayed = RNStream(replayed_file_name).stream_in()  # this original data will be compared with replayed data later

    for ts_name in test_stream_names:
        # test the data
        a = data_original[ts_name][0]
        b = data_replayed[ts_name][0]
        assert np.all(a[:, -b.shape[1]:] == b)

        # test the timestamps
        a = data_original[ts_name][1]
        b = data_replayed[ts_name][1]
        c = a[-b.shape[0]:]

        d = np.diff(c)
        e = np.diff(b)
        assert np.mean(np.abs(e - d)) < 1e-9
        assert np.max(np.abs(e - d)) < 1e-9
        assert np.std(np.abs(e - d)) < 1e-9
    os.remove(app_main_window.recording_tab.save_path)
    print("Replay completed")
