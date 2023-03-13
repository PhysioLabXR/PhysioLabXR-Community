import secrets
import string
from multiprocessing import Process

from PyQt5 import QtCore, Qt
from PyQt5.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.config import lsl_stream_availability_wait_time
from rena.tests.TestStream import LSLTestStream


def stream_is_available(app: MainWindow, test_stream_name: str):
    assert app.stream_widgets[test_stream_name].is_stream_available

class TestContext:
    """
    Helper class for carrying out the most performed actions in the tests

    """
    def __init__(self, app: MainWindow, qtbot: QtBot):
        self.send_data_processes = {}
        self.app = app
        self.qtbot = qtbot

        self.stream_availability_timeout = 2 * lsl_stream_availability_wait_time * 1e3

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

        self.qtbot.waitUntil(stream_is_available, timeout=self.stream_availability_timeout)  # wait until the LSL stream becomes available
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
