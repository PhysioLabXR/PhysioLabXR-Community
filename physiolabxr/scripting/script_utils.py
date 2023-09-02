import importlib.util
import os.path
import os
import pickle
from inspect import isclass
from typing import Type

from physiolabxr.exceptions.exceptions import InvalidScriptPathError, ScriptSyntaxError, ScriptMissingModuleError

from multiprocessing import Process

from physiolabxr.configs import config
from physiolabxr.scripting.RenaScript import RenaScript

debugging = False


def start_script_server(script_path, script_args):
    print("script process, starting script thread")
    # sys.stdout = Stream(newText=self.on_print)

    target_class = get_target_class(script_path)
    replay_client_thread = target_class(**script_args)
    replay_client_thread.start()

def validate_python_script_class(script_path: str, desired_class: Type):
    target_class, target_class_name = validate_python_script(script_path, desired_class)
    try:
        assert issubclass(target_class, desired_class)
    except AssertionError:
        raise InvalidScriptPathError(script_path, f'The first class ({target_class_name}) in the script does not inherit {desired_class.__name__}.')

def validate_python_script(script_path: str, desired_class: Type):
    """
    Validate if the script at <script_path> can be loaded without any import
    or module not found error.
    Also checks to make sure if the first class of in the script is an implementation
    of the RenaScript class.
    This function ensures that of the script can be laoded. Then it will run under
    the scripting widget
    :param script_path: path to the script to be loaded
    """
    try:
        assert os.path.exists(script_path)
    except AssertionError:
        raise InvalidScriptPathError(script_path, 'File Not Found')
    try:
        assert script_path.endswith('.py')
    except AssertionError:
        raise InvalidScriptPathError(script_path, 'File name must end with .py')
    try:
        target_class = get_target_class(script_path)
        target_class_name = get_target_class_name(script_path, desired_class)
    except IndexError:
        raise InvalidScriptPathError(script_path, 'Script does not have class defined')
    except ModuleNotFoundError as e:
        raise ScriptMissingModuleError(script_path, e)
    except SyntaxError as e:
        raise ScriptSyntaxError(e)
    return target_class, target_class_name

def get_target_class(script_path):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(script_path)), script_path)
    script_module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(script_module)
    except ImportError as e:
        raise ScriptMissingModuleError(script_path, e)
    classes = [x for x in dir(script_module) if
               isclass(getattr(script_module, x))]  # all the classes defined in the module
    classes = [script_module.__getattribute__(x) for x in classes if x != 'RenaScript']  # exclude RenaScript itself
    classes = [x for x in classes if issubclass(x, RenaScript)]
    try:
        assert len(classes) == 1
    except AssertionError:
        raise InvalidScriptPathError(script_path, 'Script has more than one classes that extends RenaScript. There can be only one subclass of RenaScript in the script file.')
    return classes[0]


def get_target_class_name(script_path, desired_class: Type):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(script_path)), script_path)
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)
    classes = [x for x in dir(script_module) if
               isclass(getattr(script_module, x))]  # all the classes defined in the module
    classes = [x for x in classes if x != desired_class.__name__]  # exclude RenaScript itself
    if len(classes) == 0:
        raise InvalidScriptPathError(script_path, f'Script does not have desired class with name {desired_class.__name__} defined')
    return classes[0]


def start_rena_script(script_path, script_args):
    print('Script started')
    if not debugging:
        script_process = Process(target=start_script_server, args=(script_path, script_args))
        script_process.start()
        return script_process
    else:
        pickle.dump([script_path, script_args], open('start_script_args.p', 'wb'))


if __name__ == '__main__':
    """
    Running this script is for debugging
    """
    # script_args = {'inputs': None, 'input_shapes': None,
    #                'outputs': None, 'output_num_channels': None,
    #                'params': None, 'port': None, 'run_frequency': None,
    #                'time_window': None}
    # script_path = '../scripting/IndexPen.py'
    script_path, script_args = pickle.load(open('start_script_args.p', 'rb'))
    start_script_server(script_path, script_args)


def get_script_widgets_args():
    rtn = dict()
    config.settings.beginGroup('scripts')
    for script_id in config.settings.childGroups():
        config.settings.beginGroup(script_id)
        rtn[script_id] = dict([(k, config.settings.value(k)) for k in config.settings.childKeys()])
        rtn[script_id]['id'] = script_id
        config.settings.endGroup()
    config.settings.endGroup()
    return rtn


