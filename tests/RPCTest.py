"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""
import os

# reference https://www.youtube.com/watch?v=WjctCBjHvmA

import pytest
from PyQt6 import QtCore
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.rpc.compiler import generate_proto_from_script_class, compile_rpc
from physiolabxr.scripting.script_utils import get_script_class

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

def check_generated_files(script_path):
    script_class = get_script_class(script_path)
    proto_file = f"{script_class.__name__}.proto"
    pb2_file = f"{script_class.__name__}_pb2.py"
    pb2_grpc_file = f"{script_class.__name__}_pb2_grpc.py"
    server_file = f"{script_class.__name__}Server.py"
    script_dir = os.path.dirname(script_path)
    assert os.path.exists(os.path.join(script_dir, proto_file))
    assert os.path.exists(os.path.join(script_dir, pb2_file))
    assert os.path.exists(os.path.join(script_dir, pb2_grpc_file))
    assert os.path.exists(os.path.join(script_dir, server_file))

def remove_generated_files(script_path):
    script_class = get_script_class(script_path)
    proto_file = f"{script_class.__name__}.proto"
    pb2_file = f"{script_class.__name__}_pb2.py"
    pb2_grpc_file = f"{script_class.__name__}_pb2_grpc.py"
    server_file = f"{script_class.__name__}Server.py"
    script_dir = os.path.dirname(script_path)
    if os.path.exists(os.path.join(script_dir, proto_file)):
        os.remove(os.path.join(script_dir, proto_file))
    if os.path.exists(os.path.join(script_dir, pb2_file)):
        os.remove(os.path.join(script_dir, pb2_file))
    if os.path.exists(os.path.join(script_dir, pb2_grpc_file)):
        os.remove(os.path.join(script_dir, pb2_grpc_file))
    if os.path.exists(os.path.join(script_dir, server_file)):
        os.remove(os.path.join(script_dir, server_file))

def test_rpc_compiler_standalone():
    script_path = "tests/assets/RPCTest.py"

    # remove everything in the script folder except the script
    remove_generated_files(script_path)

    assert compile_rpc(script_path)
    check_generated_files(script_path)

def test_rpc_compile_from_app(context_bot, qtbot):
    """
    """
    script_path = "tests/assets/RPCTest.py"
    remove_generated_files(script_path)

    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    scripting_widget = context_bot.add_existing_script(script_path)

    qtbot.mouseClick(scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.wait_until(lambda: scripting_widget.script_console_log.get_most_recent_msg() == 'Loop: rpc server', timeout=5000)
    check_generated_files(script_path)


def test_rpc_calls(context_bot, qtbot):
    """
    """
    script_path = "tests/assets/RPCTest.py"
    # TODO remove_generated_files(script_path)

    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    scripting_widget = context_bot.add_existing_script(script_path)

    qtbot.mouseClick(scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.wait_until(lambda: scripting_widget.script_console_log.get_most_recent_msg() == 'Loop: rpc server', timeout=10000)  # wait till the compile finishes and the loop is called
    check_generated_files(script_path)

    # call the rpc method



def test_rpc_with_unsupported_typehint():
    """
    TODO an error should be raised if the method has unsupported type hints for its arguments
    """
    pass