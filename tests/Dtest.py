"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""
print("!!! importing faulthandler")
import faulthandler
# reference https://www.youtube.com/watch?v=WjctCBjHvmA
print("!!! importing multiprocessing")
from multiprocessing import Process

print("!!! importing pytest")
import pytest

print("!!! from PyQt6 import QtCore")
from PyQt6 import QtCore

print("!!! from PyQt6.QtCore import Qt")
from PyQt6.QtCore import Qt

print("!!! from PyQt6.QtWidgets import QWidget")
from PyQt6.QtWidgets import QWidget

print("!!! from physiolabxr.configs.configs import AppConfigs")
from physiolabxr.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from tests.TestStream import LSLTestStream
from tests.test_utils import app_fixture, ContextBot, get_random_test_stream_names





@pytest.fixture
def app_main_window(qtbot):

    print(QtCore.PYQT_VERSION_STR)
    app, test_renalabapp_main_window = app_fixture(qtbot)
    faulthandler.disable()  # disable the faulthandler to avoid the error message
    yield test_renalabapp_main_window
    app.quit()


@pytest.fixture
def context_bot(app_main_window, qtbot):
    print("here")
    test_context = ContextBot(app=app_main_window, qtbot=qtbot)
    yield test_context
    test_context.clean_up()


def test_button_click(qtbot,app_main_window):
    print(QtCore.PYQT_VERSION_STR)
    print("here")



def test_add_inactive_unknown_stream_in_added_stream_widgets(app_main_window, qtbot) -> None:
    '''
    Adding inactive stream
    :param app:
    :param qtbot:
    :return:
    '''
    print("here")
    test_stream_name = 'TestStreamName'

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    # qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    # qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, test_stream_name)
    # assert app_main_window.addStreamWidget.add_btn.isEnabled()
    # qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    #
    # assert test_stream_name in app_main_window.get_added_stream_names()

