"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import random
import threading
import uuid
from multiprocessing import Process

import pytest
import zmq
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialogButtonBox

from rena.config import stream_availability_wait_time
from rena.tests.TestStream import LSLTestStream, ZMQTestStream
from rena.tests.test_utils import handle_current_dialog_ok, app_fixture, \
    ContextBot, handle_current_dialog_button


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


def test_zmq_channel_mistmatch(app_main_window, context_bot, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName' + str(uuid.uuid4())
    streaming_time_second = 3

    port = context_bot.create_zmq_stream(test_stream_name, num_channels=random.randint(100, 200), srate=30)
    app_main_window.create_preset(test_stream_name, 'uint8', port, 'ZMQ', num_channels=random.randint(1, 99))  # add a default preset

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, test_stream_name)

    qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.wait(int(stream_availability_wait_time * 1e3))

    def stream_is_available():
        assert app_main_window.stream_widgets[test_stream_name].is_stream_available
    qtbot.waitUntil(stream_is_available, timeout=int(2 * stream_availability_wait_time * 1e3))  # wait until the LSL stream becomes available

    def waitForCurrentDialog():
        assert app_main_window.current_dialog
    t = threading.Timer(4, lambda: handle_current_dialog_button(QDialogButtonBox.Yes, app_main_window, qtbot, click_delay_second=1))   # get the messagebox about channel mismatch
    t.start()
    qtbot.mouseClick(app_main_window.stream_widgets[test_stream_name].StartStopStreamBtn, QtCore.Qt.LeftButton)
    qtbot.waitUntil(waitForCurrentDialog)
    t.join()

    qtbot.wait(int(streaming_time_second * 1e3))

    # check if data is being plotted
    assert app_main_window.stream_widgets[test_stream_name].viz_data_buffer.has_data()

