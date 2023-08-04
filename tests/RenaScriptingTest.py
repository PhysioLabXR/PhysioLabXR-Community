# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import importlib
import os
import sys
from multiprocessing import Process

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget
from rena.configs.configs import AppConfigs
from tests.TestStream import SampleDefinedTestStream

AppConfigs(_reset=True)  # create the singleton app configs object

from rena.MainWindow import MainWindow
from rena.startup import load_settings
from tests.test_utils import ContextBot, app_fixture, get_random_test_stream_names


@pytest.fixture
def app_main_window(qtbot):
    app, test_renalabapp_main_window = app_fixture(qtbot)
    yield test_renalabapp_main_window
    app.quit()


@pytest.fixture
def app(qtbot):
    print('Initializing test fixture for ' + 'Visualization Features')
    # update_test_cwd()
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initialization
    load_settings(revert_to_default=True, reload_presets=True)  # load the default settings
    test_renalabapp = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    test_renalabapp.show()
    qtbot.addWidget(test_renalabapp)
    return test_renalabapp

@pytest.fixture
def context_bot(app_main_window, qtbot):
    test_context = ContextBot(app=app_main_window, qtbot=qtbot)

    yield test_context
    test_context.clean_up()


def test_create_script(app_main_window, qtbot):
    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.scripting_tab.AddScriptBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    class_name = 'ScriptTest'
    this_scripting_widget = app_main_window.scripting_tab.script_widgets[-1]
    script_path = os.path.join(os.getcwd(), class_name + '.py')  # TODO also need to test without .py
    this_scripting_widget.create_script(script_path, is_open_file=False)

    assert os.path.exists(script_path)
    try:
        importlib.import_module(class_name)
    except ImportError:
        raise AssertionError

    # delete the file and remove the script from rena as clean up steps
    qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    os.remove(script_path)

def test_script_input_output(context_bot, qtbot):
    test_stream_name = get_random_test_stream_names(1)[0]
    n_channels = 2
    recording_time_second = 8
    srate = 2048

    context_bot.start_predefined_stream(test_stream_name, n_channels, srate, recording_time_second)

    class_name = 'ScriptTest'
    script_path = os.path.join(os.getcwd(), class_name + '.py')
    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    qtbot.mouseClick(context_bot.app.scripting_tab.AddScriptBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    this_scripting_widget = context_bot.app.scripting_tab.script_widgets[-1]
    this_scripting_widget.create_script(script_path, is_open_file=False)

    # adding input
    qtbot.mouseClick(this_scripting_widget.inputComboBox, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.keyPress(this_scripting_widget.inputComboBox, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(this_scripting_widget.inputComboBox, test_stream_name)
    qtbot.mouseClick(this_scripting_widget.addInputBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    # check the input has been added
    expected_shape = (n_channels, int(int(this_scripting_widget.timeWindowLineEdit.text()) * srate))
    assert this_scripting_widget.get_input_shape_dict()[test_stream_name] == expected_shape

    # delete the file and remove the script from rena as clean up steps
    qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    os.remove(script_path)

