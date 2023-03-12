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

def test_plot_format_change(app, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name,))
    p.start()
    print("Test complete, killing sending-data process")
    p.kill()  # stop the dummy LSL process


def test_replay_multi_streams(app, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    num_stream_to_test = 3
    recording_time_second = 3
    replay_file_session_name = 'replayed'
    stream_availability_timeout = 2 * lsl_stream_availability_wait_time * 1e3

    test_stream_names = []
    test_stream_processes = []
    for i in range(num_stream_to_test):
        ts_name = f'TestStreamName{i}'
        test_stream_names.append(ts_name)
        p = Process(target=LSLTestStream, args=(ts_name,))
        test_stream_processes.append(p)
        p.start()
        app.create_preset(f'TestStreamName{i}', 'float', None, 'LSL',num_channels=81)  # add a default preset

    for ts_name in test_stream_names:
        app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
        qtbot.mouseClick(app.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
        qtbot.keyPress(app.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
        qtbot.keyClicks(app.addStreamWidget.stream_name_combo_box, ts_name)
        qtbot.mouseClick(app.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

    app.settings_tab.set_recording_file_location(os.getcwd())  # set recording file location (not through the system's file dialog)

    def stream_is_available():
        for ts_name in test_stream_names:
            assert app.stream_widgets[ts_name].is_stream_available
    def stream_is_unavailable():
        for ts_name in test_stream_names:
            assert not app.stream_widgets[ts_name].is_stream_available

    qtbot.waitUntil(stream_is_available, timeout=stream_availability_timeout)  # wait until the LSL stream becomes available

    for ts_name in test_stream_names:
        qtbot.mouseClick(app.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.LeftButton)

    # test if the data are being received
    for ts_name in test_stream_names:
        assert app.stream_widgets[ts_name].viz_data_buffer.has_data()

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
    qtbot.mouseClick(app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

    qtbot.wait(recording_time_second * 1e3)
    # time.sleep(recording_time_second)

    def handle_custom_dialog_ok(patience=0):
        w = QtWidgets.QApplication.activeWindow()
        if patience == 0:
            if isinstance(w, CustomDialog):
                yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
                qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)  # delay 1 second for the data to come in
        else:
            time_started = time.time()
            while not isinstance(w, CustomDialog):
                time_waited = time.time() - time_started
                if time_waited > patience:
                    raise TimeoutError
                time.sleep(0.5)
            yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)

    t = threading.Timer(1, handle_custom_dialog_ok)
    t.start()
    qtbot.mouseClick(app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # stop the recording
    t.join()  # wait until the dialog is closed
    #
    qtbot.mouseClick(app.stop_all_btn, QtCore.Qt.LeftButton)  # stop all the streams, so we don't need to handle stream lost
    #
    print("Waiting for test stream processes to close")
    [p.kill() for p in test_stream_processes]
    qtbot.waitUntil(stream_is_unavailable, timeout=stream_availability_timeout)  # wait until the lsl processes are closed

    # start the streams from replay and record them ################################################
    recording_file_name = app.recording_tab.save_path
    # load in the data
    data_original = RNStream(recording_file_name).stream_in()  # this original data will be compared with replayed data later
    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'replay_tab'))  # switch to the replay widget
    app.replay_tab.select_file(recording_file_name)
    qtbot.mouseClick(app.replay_tab.StartStopReplayBtn, QtCore.Qt.LeftButton)  # stop the recording

    print("Waiting for replay streams to become available")
    [p.kill() for p in test_stream_processes]
    qtbot.waitUntil(stream_is_available, timeout=stream_availability_timeout)  # wait until the streams becomes available from replay

    # start the streams from replay and record them ################################################
    for ts_name in test_stream_names:
        qtbot.mouseClick(app.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.LeftButton)

    # test if the data are being received
    for ts_name in test_stream_names:
        assert app.stream_widgets[ts_name].viz_data_buffer.has_data()

    # change the recording file name
    qtbot.mouseClick(app.recording_tab.sessionTagTextEdit, QtCore.Qt.LeftButton)
    qtbot.keyPress(app.recording_tab.sessionTagTextEdit, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app.recording_tab.sessionTagTextEdit, replay_file_session_name)

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
    qtbot.mouseClick(app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

    wait_for_replay_finishes_time = (recording_time_second * 2) * 1e3

    qtbot.waitUntil(lambda: not app.replay_tab.is_replaying, timeout=wait_for_replay_finishes_time)  # wait until the replay completes, need to ensure that the replay can finish
    print("replay is over")
    # the streams are stopped at this point

    t = threading.Timer(1, handle_custom_dialog_ok)
    t.start()
    qtbot.mouseClick(app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # stop the recording

    # qtbot.mouseClick(app.stop_all_btn, QtCore.Qt.LeftButton)  # stop all the streams

    # replay is completed and the data file saved ################################################
    replayed_file_name = app.recording_tab.save_path
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
        break
    os.remove(app.recording_tab.save_path)
    print("Replay completed")
