"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import random
import sys
import threading
import unittest
from multiprocessing import Process

import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QMessageBox
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import lsl_stream_availability_wait_time
from rena.startup import load_default_settings
from rena.tests.TestStream import LSLTestStream, ZMQTestStream
from rena.utils.settings_utils import create_default_preset
from rena.utils.ui_utils import CustomDialog


@pytest.fixture
def app(qtbot: QtBot):
    print('Initializing test fixture for ' + 'Visualization Features')
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initialization
    load_default_settings(revert_to_default=True, reload_presets=True)  # load the default settings
    test_renalabapp = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    # test_renalabapp.show()
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
    recording_time = 3

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

    app.settings_tab.set_recording_file_location('/')  # set recording file location (not through the system's file dialog)

    def stream_is_available():
        for ts_name in test_stream_names:
            assert app.stream_widgets[ts_name].is_stream_available
    qtbot.waitUntil(stream_is_available, timeout=2 * lsl_stream_availability_wait_time * 1e3)  # wait until the LSL stream becomes available

    for ts_name in test_stream_names:
        qtbot.mouseClick(app.stream_widgets[ts_name].StartStopStreamBtn, QtCore.Qt.LeftButton)

    # test if the data are being received
    for ts_name in test_stream_names:
        assert app.stream_widgets[ts_name].viz_data_buffer.has_data()

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'recording_tab'))  # switch to the recoding widget
    qtbot.mouseClick(app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # start the recording

    qtbot.wait(recording_time * 1e3)

    def handle_dialog():
        w = QtWidgets.QApplication.activeWindow()
        if isinstance(w, CustomDialog):
            yes_button = w.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
            qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)  # delay 1 second for the data to come in

    threading.Timer(1, handle_dialog).start()
    qtbot.mouseClick(app.recording_tab.StartStopRecordingBtn, QtCore.Qt.LeftButton)  # stop the recording


    recording_file_name = app.recording_tab.save_path
    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'replay_tab'))  # switch to the replay widget
    app.replay_tab.select_file(recording_file_name)

    print("Test complete, killing sending-data process")
    [p.kill() for p in test_stream_processes]

