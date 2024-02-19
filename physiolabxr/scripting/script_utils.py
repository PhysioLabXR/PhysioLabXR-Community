import importlib.util
import logging
import os.path
import os
import pickle
import sys
import threading
from concurrent import futures
from inspect import isclass
from typing import Type

import grpc

from physiolabxr.configs.shared import SCRIPT_INFO_PREFIX, SCRIPT_ERR_PREFIX, SCRIPT_WARNING_PREFIX
from physiolabxr.exceptions.exceptions import InvalidScriptPathError, ScriptSyntaxError, ScriptMissingModuleError

from multiprocessing import Process

from physiolabxr.configs import config
from physiolabxr.rpc.utils import run_rpc_server, create_rpc_server
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.fs_utils import load_file_classes
from physiolabxr.utils.networking_utils import send_string_router, recv_string_router

debugging = False


class RedirectStdout(object):
    def __init__(self, socket_interface, routing_id):
        self.terminal = sys.stdout
        self.routing_id = routing_id
        self.socket_interface = socket_interface

    def write(self, message):
        self.terminal.write(message)
        send_string_router(SCRIPT_INFO_PREFIX + message, self.routing_id, self.socket_interface)

    def flush(self):
        pass


class RedirectStderr(object):
    def __init__(self, socket_interface, routing_id):
        self.terminal = sys.stderr
        self.routing_id = routing_id
        self.socket_interface = socket_interface
        self.message_buffer = ""

    def write(self, message):
        self.terminal.write(message)
        self.message_buffer += message

    def flush(self):
        pass

    def send_buffered_messages(self):
        if self.message_buffer:
            send_string_router(SCRIPT_ERR_PREFIX + self.message_buffer, self.routing_id, self.socket_interface)
            self.message_buffer = ""

class SocketLoggingHandler(logging.Handler):
    def __init__(self, socket_interface, routing_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.socket_interface = socket_interface
        self.routing_id = routing_id

    def emit(self, record):
        try:
            msg = self.format(record)  # Format the log record to a string
            if record.levelno >= logging.ERROR:
                prefix = SCRIPT_ERR_PREFIX  # Use a different prefix for errors
            elif record.levelno >= logging.WARNING:
                prefix = SCRIPT_WARNING_PREFIX
            else:
                prefix = SCRIPT_INFO_PREFIX  # Default prefix for non-error messages
            # Send the message with the appropriate prefix
            send_string_router(prefix + msg, self.routing_id, self.socket_interface)
        except Exception:
            self.handleError(record)

def start_script_server(script_path, script_args):
    """
    This is the entry point of the script process.

    It starts the thread on which the RenaScrip lives, and also start the rpc server, whose wait functions
    spins on the main thread.
    """
    # redirect stdout
    stdout_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_STDOUT',
                                               port_id=script_args['port'],
                                               identity='server',
                                               pattern='router-dealer')
    logging.info('Waiting for stdout routing ID from main app for stdout socket')
    _, stdout_routing_id = recv_string_router(stdout_socket_interface, True)
    sys.stdout = redirect_stdout = RedirectStdout(socket_interface=stdout_socket_interface, routing_id=stdout_routing_id)
    sys.stderr = redirect_stderr = RedirectStderr(socket_interface=stdout_socket_interface, routing_id=stdout_routing_id)

    script_args['redirect_stdout'] = redirect_stdout
    script_args['redirect_stderr'] = redirect_stderr
    script_args['stdout_socket_interface'] = stdout_socket_interface

    # redirect logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    socket_handler = SocketLoggingHandler(socket_interface=stdout_socket_interface, routing_id=stdout_routing_id)
    socket_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(socket_handler)

    logging.info("Starting script thread")
    target_class = get_script_class(script_path)
    replay_client_thread = target_class(**script_args)
    replay_client_thread.start()

    # also start the rpc
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc_server = create_rpc_server(script_path, replay_client_thread, server, "50051")
    rpc_server.start()
    replay_client_thread.rpc_server = rpc_server

    rpc_server.wait_for_termination()

    logging.info('Script process terminated.')


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
        target_class = get_script_class(script_path)
        target_class_name = get_script_class_name(script_path, desired_class)
    except IndexError:
        raise InvalidScriptPathError(script_path, 'Script does not have class defined')
    except ModuleNotFoundError as e:
        raise ScriptMissingModuleError(script_path, e)
    except SyntaxError as e:
        raise ScriptSyntaxError(e)
    return target_class, target_class_name


# def get_servicer_class(script_path):
#     classes = load_file_classes(script_path)
#     # return the class that has the suffix Servicer
#     for c in classes:
#         if c.__name__.endswith('Servicer'):
#             return c
#     raise InvalidScriptPathError(script_path, 'Script does not have a class that ends with Servicer')


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


def get_script_class_name(script_path, desired_class: Type=RenaScript):
    spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(script_path)), script_path)
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)
    class_names = [x for x in dir(script_module) if
               isclass(getattr(script_module, x))]  # all the classes defined in the module
    class_names = [x for x in class_names if x != desired_class.__name__]  # exclude RenaScript itself
    class_names = [x for x in class_names if issubclass(script_module.__getattribute__(x), RenaScript)]

    if len(class_names) == 0:
        raise InvalidScriptPathError(script_path, f'Script does not have desired class with name {desired_class.__name__} defined')
    return class_names[0]


def get_script_class(script_path):
    classes = load_file_classes(script_path)
    classes = [x for x in classes if issubclass(x, RenaScript)]
    try:
        assert len(classes) == 1
    except AssertionError:
        raise InvalidScriptPathError(script_path,
                                     'Script has more than one classes that extends RenaScript. There can be only one subclass of RenaScript in the script file.')
    return classes[0]

