import importlib.util
import os.path
import os
from inspect import isclass
from exceptions.exceptions import InvalidScripPathError

from multiprocessing import Process

def start_script_server(script_class):
    print("script server Started")
    replay_client_thread = script_class()
    replay_client_thread.start()

def validate_script_path(script_path: str):
    try:
        assert os.path.exists(script_path)
    except AssertionError:
        raise InvalidScripPathError(script_path, 'File Not Found')
    try:
        assert script_path.endswith('.py')
    except AssertionError:
        raise InvalidScripPathError(script_path, 'File name must end with .py')

def get_target_class(script_path):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(script_path)), script_path)
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)

    classes = [x for x in dir(script_module) if
               isclass(getattr(script_module, x))]  # all the classes defined in the module
    target_class = classes[0]
    return script_module.__getattribute__(target_class)

def start_script(script_path):
    print('Script started')
    target_class = get_target_class(script_path)
    start_script_server(target_class)
    script_process = Process(target=start_script_server, args=(target_class))


def stop_script(script_path):
    print('Script stopped')
