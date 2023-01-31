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

def test_lsl_channel_mistmatch(app, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name, random.randint(100, 200)))
    p.start()

    app.create_preset('TestStreamName', 'float', None, 'LSL', num_channels=random.randint(1, 99))  # add a default preset

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app.addStreamWidget.stream_name_combo_box, test_stream_name)

    qtbot.mouseClick(app.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box
    def stream_is_available():
        assert app.stream_widgets[test_stream_name].is_stream_available
    qtbot.waitUntil(stream_is_available, timeout=2 * lsl_stream_availability_wait_time * 1e3)  # wait until the LSL stream becomes available

    # get the messagebox about channel mismatch
    def handle_dialog():
        w = QtWidgets.QApplication.activeWindow()
        if isinstance(w, QMessageBox):
            yes_button = w.button(QtWidgets.QMessageBox.Yes)
            qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)  # delay 1 second for the data to come in

    threading.Timer(1, handle_dialog).start()
    qtbot.mouseClick(app.stream_widgets[test_stream_name].StartStopStreamBtn, QtCore.Qt.LeftButton)
    # check if data is being plotted
    assert app.stream_widgets[test_stream_name].viz_data_buffer.has_data()

    p.kill()  # stop the dummy LSL process
