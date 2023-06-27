"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA
from multiprocessing import Process

import pytest
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from rena.configs.configs import AppConfigs

AppConfigs(_reset=True)  # create the singleton app configs object
from rena.tests.TestStream import LSLTestStream
from rena.tests.test_utils import app_fixture, \
    ContextBot


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

def test_add_inactive_unknown_stream_in_added_stream_widgets(app_main_window, qtbot) -> None:
    '''
    Adding inactive stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, test_stream_name)
    qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

    assert test_stream_name in app_main_window.get_added_stream_names()

def test_add_active_unknown_stream_in_added_stream_widgets(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name,))
    p.start()

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, test_stream_name)
    qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

    assert test_stream_name in app_main_window.get_added_stream_names()
    p.kill()  # stop the dummy LSL process
#
# def test_stream_availablity(app, qtbot):
#     pass

# def test_running_random_stream(app, qtbot):
#     pass
#     # TODO

# def test_label_after_click(app, qtbot):
#     qtbot.mouseClick(app.button, QtCore.Qt.LeftButton)
#     assert app.text_label.text() == "Changed!"

def test_add_active_unknown_stream_stream_widgets_and_close(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name,))
    p.start()

    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app_main_window.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app_main_window.addStreamWidget.stream_name_combo_box, test_stream_name)
    qtbot.mouseClick(app_main_window.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box
    assert test_stream_name in app_main_window.get_added_stream_names()

    qtbot.mouseClick(app_main_window.stream_widgets[test_stream_name].RemoveStreamBtn, QtCore.Qt.LeftButton)  # click the add widget combo box

    p.kill()  # stop the dummy LSL process