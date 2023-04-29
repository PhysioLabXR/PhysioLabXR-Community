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

from rena.tests.test_utils import update_test_cwd, ContextBot, app_fixture


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

def test_with_buggy_presets(app_main_window, context_bot, qtbot):
    """
    pass in a temporary preset root containing a buggy preset file
    @param app_main_window:
    @param context_bot:
    @param qtbot:
    @return:
    """
    pass

def test_reloading_modified_presets(app_main_window, context_bot, qtbot):
    pass

def test_reloading_new_presets(app_main_window, context_bot, qtbot):
    pass

def test_reloading_remove_presets(app_main_window, context_bot, qtbot):
    pass

def test_existing_presets(app_main_window, context_bot, qtbot):
    pass
