# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import sys
import unittest
from multiprocessing import Process

import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from rena.MainWindow import MainWindow
from rena.startup import load_default_settings
from rena.tests.LSLTestStream import LSLTestStream

@pytest.fixture
def app(qtbot: QtBot):
    print('Initializing test fixature for ' + 'Visualization Features')
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initialization
    load_default_settings(revert_to_default=True, reload_presets=True)  # load the default settings
    test_renalabapp = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    test_renalabapp.show()
    qtbot.addWidget(test_renalabapp)
    return test_renalabapp

def teardown_function(function):
    """ teardown any state that was previously setup with a setup_method
    call.
    """
    pass

def test_add_inactive_unknown_stream_in_added_stream_wigets(app, qtbot) -> None:
    '''
    Adding inactive stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app.addStreamWidget.stream_name_combo_box, test_stream_name)
    qtbot.mouseClick(app.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

    assert test_stream_name in app.get_added_stream_names()

def test_add_active_unknown_stream_in_added_stream_wigets(app, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name,))
    p.start()

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app.addStreamWidget.stream_name_combo_box, QtCore.Qt.LeftButton)  # click the add widget combo box
    qtbot.keyPress(app.addStreamWidget.stream_name_combo_box, 'a', modifier=Qt.ControlModifier)
    qtbot.keyClicks(app.addStreamWidget.stream_name_combo_box, test_stream_name)
    qtbot.mouseClick(app.addStreamWidget.add_btn, QtCore.Qt.LeftButton)  # click the add widget combo box

    assert test_stream_name in app.get_added_stream_names()
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