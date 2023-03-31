# reference https://www.youtube.com/watch?v=WjctCBjHvmA
import os
import sys
from multiprocessing import Process

import pytest
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget

from rena.MainWindow import MainWindow
from rena.interfaces import InferenceInterface
from rena.startup import load_settings
from rena.tests.TestStream import LSLTestStream
import importlib

from rena.tests.test_utils import update_test_cwd, TestContext, app_fixture


@pytest.fixture
def app_main_window(qtbot):
    app, test_renalabapp_main_window = app_fixture(qtbot)
    yield test_renalabapp_main_window
    app.quit()

@pytest.fixture
def test_context(app_main_window, qtbot):
    test_context = TestContext(app=app_main_window, qtbot=qtbot)

    yield test_context
    test_context.clean_up()

def test_recording_text_field_disabled_on_start(app_main_window, test_context, qtbot):
    num_streams_to_start = 3

    # check that the text field is enabled when the app starts
    assert app_main_window.recording_tab.experimentNameTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.subjectTagTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.sessionTagTextEdit.isEnabled() is True

    test_context.start_streams_and_recording(num_streams_to_start)

    # check that the text field is disabled
    assert app_main_window.recording_tab.experimentNameTextEdit.isEnabled() is False
    assert app_main_window.recording_tab.subjectTagTextEdit.isEnabled() is False
    assert app_main_window.recording_tab.sessionTagTextEdit.isEnabled() is False

    test_context.stop_recording()
    assert app_main_window.recording_tab.experimentNameTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.subjectTagTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.sessionTagTextEdit.isEnabled() is True

