# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import importlib
import os
import sys

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QWidget
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import DataType, PresetType

AppConfigs(_reset=True)  # create the singleton app configs object


from physiolabxr.utils.user_utils import stream_in
from physiolabxr.ui.MainWindow import MainWindow
from physiolabxr.startup.startup import load_settings
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
    script_ids = list(app_main_window.scripting_tab.script_widgets.keys())
    this_scripting_widget = app_main_window.scripting_tab.script_widgets[script_ids[-1]]
    script_path = os.path.join(os.getcwd(), class_name + '.py')  # TODO also need to test without .py
    this_scripting_widget.create_script(script_path, is_open_file=False)

    assert os.path.exists(script_path)
    try:
        importlib.import_module(class_name)
    except ImportError:
        raise AssertionError

    # delete the file and remove the script from physiolabxr as clean up steps
    qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    os.remove(script_path)

def test_script_single_lsl_input_output(context_bot, qtbot):
    test_stream_name = get_random_test_stream_names(1)[0]
    n_channels = 2
    recording_time_second = 15
    wait_for_script_to_send_output_timeout = 5 * 1e3
    out_message = "Sent to output"
    srate = 2048
    dtype = DataType.float64

    sent_samples = context_bot.create_add_start_predefined_stream(test_stream_name, n_channels, srate, recording_time_second * 3, dtype)

    class_name = 'ScriptTest'
    script_path = os.path.join(os.getcwd(), class_name + '.py')
    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    qtbot.mouseClick(context_bot.app.scripting_tab.AddScriptBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    script_ids = list(context_bot.app.scripting_tab.script_widgets.keys())
    this_scripting_widget = context_bot.app.scripting_tab.script_widgets[script_ids[-1]]
    this_scripting_widget.create_script(script_path, is_open_file=False)

    # add script content
    code = f"""
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript

class ScriptTest(RenaScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit. 
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        self.outputs['output1'] = self.inputs[self.input_names[0]][0]
        self.inputs.clear_buffer()
        print("{out_message}")

    # cleanup is called when the stop button is hit    
    def cleanup(self):
        print('Cleanup function is called')
    
    """
    # output the same input stream
    with open(script_path, "w") as f:
        f.write(code)

    # adding input
    qtbot.mouseClick(this_scripting_widget.inputComboBox, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.keyPress(this_scripting_widget.inputComboBox, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(this_scripting_widget.inputComboBox, test_stream_name)
    qtbot.mouseClick(this_scripting_widget.addInputBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    # check the input has been added
    expected_shape = (n_channels, int(int(this_scripting_widget.timeWindowLineEdit.text()) * srate))
    assert this_scripting_widget.get_input_shape_dict()[test_stream_name] == expected_shape

    # adding outputs
    qtbot.mouseClick(this_scripting_widget.addOutput_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    # set the number of output channels
    qtbot.keyPress(this_scripting_widget.output_widgets[0].numChan_lineEdit, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(this_scripting_widget.output_widgets[0].numChan_lineEdit, str(n_channels))
    # change the output data type to match the sample's
    dtype_combobox = this_scripting_widget.output_widgets[0].data_type_comboBox
    assert (dtype_index := dtype_combobox.findText(dtype.name)) != -1
    qtbot.mouseClick(dtype_combobox.view().viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(0, 0))
    qtbot.keyPress(dtype_combobox.view().viewport(), Qt.Key.Key_Down)
    for i in range(dtype_index):
        qtbot.keyPress(dtype_combobox.view().viewport(), Qt.Key.Key_Down)
    qtbot.keyPress(dtype_combobox.view().viewport(), Qt.Key.Key_Return)
    assert dtype_combobox.currentText() == dtype.name

    # start the script
    qtbot.mouseClick(this_scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    # wait till the script is sending outputs

    qtbot.wait_until(lambda: this_scripting_widget.script_console_log.get_most_recent_msg() == out_message, timeout=wait_for_script_to_send_output_timeout)

    # add and start the output stream
    output_stream_name = this_scripting_widget.get_outputs()[0]
    context_bot.add_and_start_stream(output_stream_name, n_channels)

    # start recording
    context_bot.start_recording()
    qtbot.wait(1000 * recording_time_second)

    # stop recording
    context_bot.stop_recording()
    recording_file_path = context_bot.app.recording_tab.save_path
    # load the recording back
    recorded_data = stream_in(recording_file_path)

    assert np.isin(recorded_data[test_stream_name][0], sent_samples).all()
    assert np.isin(recorded_data[output_stream_name][0], sent_samples).all()

    # as clean up steps, delete the file and remove the script from physiolabxr
    qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.wait_until(lambda: context_bot.app.scripting_tab.script_widgets_empty(), timeout=wait_for_script_to_send_output_timeout)

    os.remove(recording_file_path)
    os.remove(script_path)


def test_script_single_zmq_input_output(context_bot, qtbot):
    """

    Notes:
    there's a known issue: if the number of channels is more than 1, that is, not default. The script won't be able to
    start the output stream because of custom dialog asking if the user wants to set the number of channels to match
    what's received. This dialog shows up too late for the thread used to click the yes button to be able to find it.

    @param context_bot:
    @param qtbot:
    @return:
    """
    test_stream_name = get_random_test_stream_names(1)[0]
    n_channels = 1
    recording_time_second = 30
    wait_for_script_to_send_output_timeout = 5 * 1e3
    out_message = "Sent to output"
    srate = 2048
    dtype = DataType.float64
    port = AppConfigs().test_port_starting_port

    sent_samples = context_bot.create_add_start_predefined_stream(test_stream_name, n_channels, srate, recording_time_second, dtype, interface_type=PresetType.ZMQ, port=port)

    class_name = 'ScriptTest'
    script_path = os.path.join(os.getcwd(), class_name + '.py')
    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    qtbot.mouseClick(context_bot.app.scripting_tab.AddScriptBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    script_ids = list(context_bot.app.scripting_tab.script_widgets.keys())
    this_scripting_widget = context_bot.app.scripting_tab.script_widgets[script_ids[-1]]
    this_scripting_widget.create_script(script_path, is_open_file=False)

    # add script content
    code = f"""
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript

class ScriptTest(RenaScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit. 
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        self.outputs['output1'] = self.inputs[self.input_names[0]][0]
        self.inputs.clear_buffer()
        print("{out_message}")

    # cleanup is called when the stop button is hit    
    def cleanup(self):
        print('Cleanup function is called')

    """
    # output the same input stream
    with open(script_path, "w") as f:
        f.write(code)

    # adding input
    qtbot.mouseClick(this_scripting_widget.inputComboBox,QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.keyPress(this_scripting_widget.inputComboBox, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(this_scripting_widget.inputComboBox, test_stream_name)
    qtbot.mouseClick(this_scripting_widget.addInputBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    # check the input has been added
    expected_shape = (n_channels, int(int(this_scripting_widget.timeWindowLineEdit.text()) * srate))
    assert this_scripting_widget.get_input_shape_dict()[test_stream_name] == expected_shape

    # adding outputs
    qtbot.mouseClick(this_scripting_widget.addOutput_btn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    # set the number of output channels
    qtbot.keyPress(this_scripting_widget.output_widgets[0].numChan_lineEdit, Qt.Key.Key_A, modifier=Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(this_scripting_widget.output_widgets[0].numChan_lineEdit, str(n_channels))
    # change the output data type to match the sample's
    dtype_combobox = this_scripting_widget.output_widgets[0].data_type_comboBox
    assert (dtype_index := dtype_combobox.findText(dtype.name)) != -1
    dtype_combobox.setCurrentIndex(0)
    qtbot.mouseClick(dtype_combobox.view().viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(0, 0))
    qtbot.keyPress(dtype_combobox.view().viewport(), Qt.Key.Key_Down)
    for i in range(dtype_index):
        qtbot.keyPress(dtype_combobox.view().viewport(), Qt.Key.Key_Down)
    qtbot.keyPress(dtype_combobox.view().viewport(), Qt.Key.Key_Return)
    assert dtype_combobox.currentText() == dtype.name

    # change the output interface type to ZMQ
    interface_combobox = this_scripting_widget.output_widgets[0].interface_type_comboBox
    assert (interface_index := interface_combobox.findText(PresetType.ZMQ.name)) != -1
    interface_combobox.setCurrentIndex(0)
    qtbot.mouseClick(interface_combobox.view().viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(0, 0))
    qtbot.keyPress(interface_combobox.view().viewport(), Qt.Key.Key_Down)
    for i in range(interface_index):
        qtbot.keyPress(interface_combobox.view().viewport(), Qt.Key.Key_Down)
    qtbot.keyPress(interface_combobox.view().viewport(), Qt.Key.Key_Return)
    assert interface_combobox.currentText() == PresetType.ZMQ.name
    # get the output port number
    script_output_port = this_scripting_widget.output_widgets[0].get_port_number()

    # start the script
    qtbot.mouseClick(this_scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    # wait till the script is sending outputs

    qtbot.wait_until(lambda: this_scripting_widget.script_console_log.get_most_recent_msg() == out_message, timeout=wait_for_script_to_send_output_timeout)

    # add and start the output stream
    output_stream_name = this_scripting_widget.get_outputs()[0]
    context_bot.add_and_start_stream(output_stream_name, n_channels,
                                     interface_type=PresetType.ZMQ, dtype=dtype, port=script_output_port,
                                     thread_timer_second=10)  # need to wait for the script to start sending data

    # start recording
    context_bot.start_recording()
    qtbot.wait(1000 * recording_time_second)

    # stop recording
    context_bot.stop_recording()
    recording_file_path = context_bot.app.recording_tab.save_path
    # load the recording back
    recorded_data = stream_in(recording_file_path)

    assert np.isin(recorded_data[test_stream_name][0], sent_samples).all()
    assert np.isin(recorded_data[output_stream_name][0], sent_samples).all()

    # as clean up steps, delete the file and remove the script from physiolabxr
    qtbot.mouseClick(this_scripting_widget.removeBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.wait_until(lambda: context_bot.app.scripting_tab.script_widgets_empty(), timeout=wait_for_script_to_send_output_timeout)

    os.remove(recording_file_path)
    os.remove(script_path)
