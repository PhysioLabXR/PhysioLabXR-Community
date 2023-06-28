# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import importlib
import os
import sys

import pytest
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QWidget
from rena.configs.configs import AppConfigs

AppConfigs(_reset=True)  # create the singleton app configs object
from rena.MainWindow import MainWindow
from rena.startup import load_settings
from rena.tests.test_utils import update_test_cwd


@pytest.fixture
def app(qtbot):
    print('Initializing test fixture for ' + 'Visualization Features')
    update_test_cwd()
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initialization
    load_settings(revert_to_default=True, reload_presets=True)  # load the default settings
    test_renalabapp = MainWindow(app=app, ask_to_close=False)  # close without asking so we don't pend on human input at the end of each function test fixatire
    test_renalabapp.show()
    qtbot.addWidget(test_renalabapp)
    return test_renalabapp


def test_create_script(app_main_window, qtbot):
    app_main_window.ui.tabWidget.setCurrentWidget(app_main_window.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    qtbot.mouseClick(app_main_window.scripting_tab.AddScriptBtn, QtCore.Qt.LeftButton)  # click the add widget combo box

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
    qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.LeftButton)  # click the add widget combo box
    os.remove(script_path)

# def test_script_input_output(app, qtbot):
#     test_stream_name = 'TestStreamName'
#     p = Process(target=LSLTestStream, args=(test_stream_name,))
#     p.start()
#
#     app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
#     qtbot.mouseClick(app.scripting_tab.AddScriptBtn, QtCore.Qt.LeftButton)  # click the add widget combo box
#
#     class_name = 'ScriptTest'
#     this_scripting_widget = app.scripting_tab.script_widgets[-1]
#     script_path = os.path.join(os.getcwd(), class_name + '.py')
#     this_scripting_widget.create_script(script_path, is_open_file=False)
#
#     # TODO add test body here
#
#     # delete the file and remove the script from rena as clean up steps
#     qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.LeftButton)  # click the add widget combo box
#     os.remove(script_path)