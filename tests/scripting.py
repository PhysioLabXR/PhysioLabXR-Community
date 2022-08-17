# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import sys
from multiprocessing import Process

import pytest
from PyQt5 import QtCore, QtWidgets

from rena.MainWindow import MainWindow
from rena.interfaces import InferenceInterface
from tests.LSLTestStream import LSLTestStream


@pytest.fixture
def app(qtbot):
    print('Initializing test fixature for ' + 'Scripting Features')
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # main window init
    inference_interface = InferenceInterface.InferenceInterface()
    test_renalab_app = MainWindow(app=app, inference_interface=inference_interface)

    return test_renalab_app


def test_add_random_stream(app, qtbot):
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name,))
    p.start()
    app.lslStream_name_input.setText()
    qtbot.mouseClick(app.add_lslStream_btn, QtCore.Qt.LeftButton)

    # check all the required GUI fields are added
    assert test_stream_name in app.stream_ui_elements
    assert 'lsl_widget' in app.stream_ui_elements[test_stream_name]
    app.stream_ui_elements[test_stream_name]['lsl_widget'].StreamNameLabel.text = test_stream_name


def test_stop_script_right_after_start(app, qtbot):
    test_stream_name = 'TestStopScriptRightAfterStart'

    pass
    # TODO

# def test_label_after_click(app, qtbot):
#     qtbot.mouseClick(app.button, QtCore.Qt.LeftButton)
#     assert app.text_label.text() == "Changed!"