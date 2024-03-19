"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""
import os
import sys

import grpc
# reference https://www.youtube.com/watch?v=WjctCBjHvmA

import pytest
from PyQt6 import QtCore
from PyQt6.QtWidgets import QWidget
from google.protobuf.json_format import MessageToDict
from google.protobuf import empty_pb2

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import RPCLanguage
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

    server_file = f"{script_class.__name__}Server.py"
    script_dir = os.path.dirname(script_path)

    assert os.path.exists(os.path.join(script_dir, proto_file))
    assert os.path.exists(os.path.join(script_dir, server_file))

    # compare with the golden file for RPCTest.proto and RPCTestServer.py
    with open(os.path.join(script_dir, proto_file), 'r') as f:
        assert f.read() == open(f"tests/assets/{proto_file}", 'r').read()
    with open(os.path.join(script_dir, server_file), 'r') as f:
        assert f.read() == open(f"tests/assets/{server_file}", 'r').read()

def check_python_generated_files(script_path):
    script_class = get_script_class(script_path)
    script_dir = os.path.dirname(script_path)

    pb2_file = f"{script_class.__name__}_pb2.py"
    pb2_grpc_file = f"{script_class.__name__}_pb2_grpc.py"
    assert os.path.exists(os.path.join(script_dir, pb2_file))
    assert os.path.exists(os.path.join(script_dir, pb2_grpc_file))


def check_csharp_generated_files(script_path):
    script_class = get_script_class(script_path)
    script_dir = os.path.dirname(script_path)

    cs_file = f"{script_class.__name__}.cs"
    cs_grpc_file = f"{script_class.__name__}Grpc.cs"

    assert os.path.exists(os.path.join(script_dir, cs_file))
    assert os.path.exists(os.path.join(script_dir, cs_grpc_file))



def remove_generated_files(script_path):
    script_class = get_script_class(script_path)
    proto_file = f"{script_class.__name__}.proto"
    pb2_file = f"{script_class.__name__}_pb2.py"
    pb2_grpc_file = f"{script_class.__name__}_pb2_grpc.py"
    server_file = f"{script_class.__name__}Server.py"
    script_dir = os.path.dirname(script_path)

    cs_file = f"{script_class.__name__}.cs"
    cs_grpc_file = f"{script_class.__name__}Grpc.cs"

    if os.path.exists(os.path.join(script_dir, proto_file)):
        os.remove(os.path.join(script_dir, proto_file))
    if os.path.exists(os.path.join(script_dir, pb2_file)):
        os.remove(os.path.join(script_dir, pb2_file))
    if os.path.exists(os.path.join(script_dir, pb2_grpc_file)):
        os.remove(os.path.join(script_dir, pb2_grpc_file))
    if os.path.exists(os.path.join(script_dir, server_file)):
        os.remove(os.path.join(script_dir, server_file))

    if os.path.exists(os.path.join(script_dir, cs_file)):
        os.remove(os.path.join(script_dir, cs_file))
    if os.path.exists(os.path.join(script_dir, cs_grpc_file)):
        os.remove(os.path.join(script_dir, cs_grpc_file))

def test_rpc_compiler_standalone_python():
    script_path = "tests/assets/RPCTest.py"

    # remove everything in the script folder except the script
    remove_generated_files(script_path)

    assert compile_rpc(script_path)
    check_generated_files(script_path)
    check_python_generated_files(script_path)


def test_rpc_compiler_standalone_csharp():
    script_path = "tests/assets/RPCTest.py"

    # remove everything in the script folder except the script
    remove_generated_files(script_path)

    rpc_outputs = [{"language": RPCLanguage.CSHARP, "location": '.'}]
    csharp_plugin_path = AppConfigs().csharp_plugin_path

    assert compile_rpc(script_path, csharp_plugin_path=csharp_plugin_path, rpc_outputs=rpc_outputs)
    check_generated_files(script_path)
    check_csharp_generated_files(script_path)


def test_rpc_compile_from_app(context_bot, qtbot):
    """
    """
    script_path = "tests/assets/RPCTest.py"
    remove_generated_files(script_path)

    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    scripting_widget = context_bot.add_existing_script(script_path)

    qtbot.mouseClick(scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    qtbot.wait(5000)
    qtbot.wait_until(lambda: scripting_widget.script_console_log._check_message_exits('Loop: rpc server'), timeout=5000)
    check_generated_files(script_path)


def test_rpc_calls(context_bot, qtbot):
    """
    """
    script_path = "tests/assets/RPCTest.py"
    # TODO remove_generated_files(script_path)

    context_bot.app.ui.tabWidget.setCurrentWidget(context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    scripting_widget = context_bot.add_existing_script(script_path)

    qtbot.mouseClick(scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box
    qtbot.wait(5000)
    qtbot.wait_until(lambda: scripting_widget.script_console_log._check_message_exits('Loop: rpc server'), timeout=10000)  # wait till the compile finishes and the loop is called
    check_generated_files(script_path)

    # call the rpc method
    channel = grpc.insecure_channel(f'localhost:{scripting_widget.rpc_port}')
    # import the rpc pb from the generated file
    # add the script path to the sys path so that the generated pb2_grpc can find the module named pb2
    sys.path.append(os.path.dirname(script_path))
    import RPCTest_pb2_grpc, RPCTest_pb2
    stub = RPCTest_pb2_grpc.RPCTestStub(channel)
    response = stub.TestRPCOneArgOneReturn(RPCTest_pb2.TestRPCOneArgOneReturnRequest(input0='test'))
    assert response.message == 'Received input: test'

    response = stub.TestRPCTwoArgTwoReturn(RPCTest_pb2.TestRPCTwoArgTwoReturnRequest(input0='test', input1=1))
    assert response.message0 == 'received test'
    assert response.message1 == 1

    response = stub.TestRPCNoInputNoReturn(empty_pb2.Empty())
    assert response == empty_pb2.Empty()

    response = stub.TestRPCNoReturn(RPCTest_pb2.TestRPCNoReturnRequest(input0=1))
    assert response == empty_pb2.Empty()

    response = stub.TestRPCNoArgs(empty_pb2.Empty())
    assert response.message == 'No Args'


def test_rpc_with_unsupported_typehint_args(context_bot, qtbot):
    """
    TODO an error should be raised if the method has unsupported type hints for its return
    the main app should not stuck if the rpc compilation fails
    """
    script_path = "tests/assets/RPCTestUnsupportedArgType.py"
    remove_generated_files(script_path)

    context_bot.app.ui.tabWidget.setCurrentWidget( context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    scripting_widget = context_bot.add_existing_script(script_path)

    qtbot.mouseClick(scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    # wait five seconds for the rpc server to start
    qtbot.wait(5000)
    qtbot.wait_until(lambda: scripting_widget.script_console_log._check_message_exits('RPC method TestUnsupportedArgType has unsupported type hint for argument input1'), timeout=5000)
    qtbot.wait_until(lambda: scripting_widget.script_console_log._check_message_exits("Unsupported type 'set' in RPC method"), timeout=5000)

def test_rpc_with_unsupported_typehint_return(context_bot, qtbot):
    """
    TODO an error should be raised if the method has unsupported type hints for its return
    the main app should not stuck if the rpc compilation fails
    """
    script_path = "tests/assets/RPCTestUnsupportedReturnType.py"
    remove_generated_files(script_path)

    context_bot.app.ui.tabWidget.setCurrentWidget( context_bot.app.ui.tabWidget.findChild(QWidget, 'scripting_tab'))  # switch to the visualization widget
    scripting_widget = context_bot.add_existing_script(script_path)

    qtbot.mouseClick(scripting_widget.runBtn, QtCore.Qt.MouseButton.LeftButton)  # click the add widget combo box

    # check for the build failed message
    qtbot.wait(5000)
    qtbot.wait_until(lambda: scripting_widget.script_console_log._check_message_exits('RPC method TestUnsupportedReturnType has unsupported type hint for its return'), timeout=5000)
    qtbot.wait_until(lambda: scripting_widget.script_console_log._check_message_exits("Unsupported type 'set' in RPC method"), timeout=5000)
