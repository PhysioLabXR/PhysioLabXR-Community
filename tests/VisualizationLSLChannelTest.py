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
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QDialogButtonBox
from rena.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from rena.presets.PresetEnums import PresetType
from rena.config import stream_availability_wait_time
from tests.TestStream import LSLTestStream
from tests.test_utils import app_fixture, \
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

def test_lsl_channel_mistmatch(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName' + str(uuid.uuid4())
    actual_num_chan = random.randint(100, 200)
    preset_num_chan = random.randint(1, 99)
    streaming_time_second = 3

    p = Process(target=LSLTestStream, args=(test_stream_name, actual_num_chan))
    p.start()

    app_main_window.create_preset(test_stream_name, PresetType.LSL, num_channels=preset_num_chan)  # add a default preset

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, test_stream_name)
    assert app_main_window.addStreamWidget.add_btn.isEnabled()
    qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.wait(int(stream_availability_wait_time * 1e3))
    def stream_is_available():
        assert app_main_window.stream_widgets[test_stream_name].is_stream_available
    qtbot.waitUntil(stream_is_available, timeout=int(2 * stream_availability_wait_time * 1e3))  # wait until the LSL stream becomes available

    # def handle_dialog():
    #     w = app.current_dialog
    #     if isinstance(w, CustomDialog):
    #         yes_button = w.button(QtWidgets.QMessageBox.Yes)
    #         qtbot.mouseClick(yes_button, QtCore.Qt.LeftButton, delay=1000)  # delay 1 second for the data to come in

    t = threading.Timer(1, lambda: handle_current_dialog_button(QDialogButtonBox.StandardButton.Yes, app_main_window, qtbot, click_delay_second=1))   # get the messagebox about channel mismatch
    t.start()
    qtbot.mouseClick(app_main_window.stream_widgets[test_stream_name].StartStopStreamBtn, QtCore.Qt.MouseButton.LeftButton)
    t.join()

    qtbot.wait(int(streaming_time_second * 1e3))

    # check if data is being plotted
    assert app_main_window.stream_widgets[test_stream_name].viz_data_buffer.has_data()

    print("Test complete, killing sending-data process")
    p.kill()  # stop the dummy LSL process
