# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import sys
from multiprocessing import Process

import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget

from rena.MainWindow import MainWindow
from rena.startup import load_default_settings
from rena.tests.LSLTestStream import LSLTestStream

@pytest.fixture
def app(qtbot):
    print('Initializing test fixature for ' + 'Visualization Features')
    print(os.getcwd())
    # ignore the splash screen and tree icon
    app = QtWidgets.QApplication(sys.argv)
    # app initialization
    load_default_settings(revert_to_default=True)  # load the default settings
    test_renalabapp = MainWindow(app=app)
    qtbot.addWidget(test_renalabapp)
    return test_renalabapp


def test_add_random_stream_in_added_stream_wigets(app, qtbot) -> None:
    test_stream_name = 'TestStreamName'
    p = Process(target=LSLTestStream, args=(test_stream_name,))
    p.start()

    app.ui.tabWidget.setCurrentWidget(app.ui.tabWidget.findChild(QWidget, 'visualization_tab'))
    qtbot.keyClicks(app.addStreamWidget.stream_name_combo_box(), '')
    qtbot.mouseClick(app.addStreamWidget.add_btn, QtCore.Qt.LeftButton)

    assert test_stream_name in app.get_added_stream_names()
    p.kill()  # stop the dummy LSL process

def test_running_random_stream(app, qtbot):
    pass
    # TODO

# def test_label_after_click(app, qtbot):
#     qtbot.mouseClick(app.button, QtCore.Qt.LeftButton)
#     assert app.text_label.text() == "Changed!"