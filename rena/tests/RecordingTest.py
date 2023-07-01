# reference https://www.youtube.com/watch?v=WjctCBjHvmA

import pytest

from rena.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from rena.tests.test_utils import ContextBot, app_fixture


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

def test_recording_text_field_disabled_on_start(app_main_window, context_bot, qtbot):
    num_streams_to_start = 3

    # check that the text field is enabled when the app starts
    assert app_main_window.recording_tab.experimentNameTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.subjectTagTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.sessionTagTextEdit.isEnabled() is True

    context_bot.start_streams_and_recording(num_streams_to_start)

    # check that the text field is disabled
    assert app_main_window.recording_tab.experimentNameTextEdit.isEnabled() is False
    assert app_main_window.recording_tab.subjectTagTextEdit.isEnabled() is False
    assert app_main_window.recording_tab.sessionTagTextEdit.isEnabled() is False

    context_bot.stop_recording()
    assert app_main_window.recording_tab.experimentNameTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.subjectTagTextEdit.isEnabled() is True
    assert app_main_window.recording_tab.sessionTagTextEdit.isEnabled() is True

