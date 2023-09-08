"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA

import pytest
from physiolabxr.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from tests.test_utils import app_fixture, ContextBot


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