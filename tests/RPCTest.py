"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""

# reference https://www.youtube.com/watch?v=WjctCBjHvmA

import pytest
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

def test_rpc_compiler():
    script_path = "tests/assets/RPCTest.py"
    compile_rpc(script_path)

def test_rpc_without_typehint():
    """
    TODO an error should be raised if the method does not have type hints for its arguments
    """
    pass

def test_rpc_with_unsupported_typehint():
    """
    TODO an error should be raised if the method has unsupported type hints for its arguments
    """
    pass