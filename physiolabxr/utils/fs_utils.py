import importlib.util
import os
from inspect import isclass
from typing import List, Union

from physiolabxr.exceptions.exceptions import ScriptMissingModuleError


def get_json_file_paths_from_dir(dir_path: str) -> List[str]:
    files_paths = []
    preset_file_paths = [os.path.join(dir_path, x) for x in dir_path]
    for pf_path in preset_file_paths:
        files_paths.append(pf_path)
    return files_paths

def get_json_file_paths_from_multiple_dir(dir_paths: List[str], flatten=False) -> Union[list[list[str]], list[str]]:
    files_paths = []
    for dir_path in dir_paths:
        if flatten:
            files_paths += get_json_file_paths_from_dir(dir_path)
        else:
            files_paths.append(get_json_file_paths_from_dir(dir_path))
    return files_paths


def get_file_changes(dir_path, last_mod_times):
    files = os.listdir(dir_path)
    current_mod_times = {}
    modifed_files = []
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.json'):
            current_mod_times[file_name] = os.path.getmtime(file_path)
            if file_name not in last_mod_times or last_mod_times[file_name] != current_mod_times[file_name]:
                print('file {} has been modified'.format(file_name))
                modifed_files.append(file_path)
    return modifed_files, current_mod_times

def get_file_changes_multiple_dir(dir_paths, last_mod_times, flatten=False):
    modified_files = []
    current_mod_times = {}
    for dir_path in dir_paths:
        _modified_files, _current_mod_times = get_file_changes(dir_path, last_mod_times)
        if flatten:
            modified_files += _modified_files
        else:
            modified_files.append(_modified_files)
        current_mod_times.update(_current_mod_times)
    return modified_files, current_mod_times


def load_servicer_add_function(script_name, grpc_file_path):
    """
    Dynamically loads the add_<script name>Servicer_to_server function from a gRPC generated file.

    :param script_name: The name of the script (e.g., "MyScript" for MyScript_pb2_grpc.py)
    :param grpc_file_path: The file path to the *_pb2_grpc.py file.
    :return: The add_<script name>Servicer_to_server function, or None if not found.
    """
    module_name = os.path.basename(grpc_file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, grpc_file_path)
    grpc_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grpc_module)

    function_name = f"add_{script_name}Servicer_to_server"

    # Retrieve the function by name
    add_function = getattr(grpc_module, function_name, None)

    if add_function is None:
        raise AttributeError(f"The function {function_name} could not be found in {grpc_file_path}")

    return add_function


def load_file_classes(script_path):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(script_path)), script_path)
    script_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(script_module)
    except ImportError as e:
        raise ScriptMissingModuleError(script_path, e)
    classes = [x for x in dir(script_module) if
               isclass(getattr(script_module, x))]  # all the classes defined in the module
    classes = [script_module.__getattribute__(x) for x in classes if x != 'RenaScript']  # exclude RenaScript itself
    return classes

def import_file(script_path):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(script_path)), script_path)
    script_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(script_module)
    except ImportError as e:
        raise ScriptMissingModuleError(script_path, e)
    return script_module
