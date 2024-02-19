import logging
import pickle
import sys
from concurrent import futures
from multiprocessing import Process

import grpc

from physiolabxr.configs.shared import SCRIPT_ERR_PREFIX, SCRIPT_WARNING_PREFIX, SCRIPT_INFO_PREFIX, SCRIPT_FATAL_PREFIX
from physiolabxr.rpc.compiler import compile_rpc
from physiolabxr.rpc.utils import create_rpc_server
from physiolabxr.scripting.script_utils import get_script_class, debugging
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.networking_utils import recv_string_router, send_string_router


def start_script_server(script_path, script_args):
    """
    This is the entry point of the script process.

    It starts the thread on which the RenaScrip lives, and also start the rpc server, whose wait functions
    spins on the main thread.

    Note that any call to logging.fatal will terminate the script process.
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

    try:
        replay_client_thread = target_class(**script_args)
    except Exception as e:
        logging.fatal(f"Error creating script class: {e}")
        return

    # compile the rpc
    try:
        include_rpc = compile_rpc(script_path, target_class)
    except Exception as e:
        # notify the main app that the script has failed to start
        logging.fatal(f"Error compiling rpc, : {e}")
        return

    # also start the rpc
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpc_server = create_rpc_server(script_path, replay_client_thread, server, "50051")
    rpc_server.start()
    replay_client_thread.rpc_server = rpc_server

    replay_client_thread.start()
    rpc_server.wait_for_termination()

    logging.info('Script process terminated.')


class SocketLoggingHandler(logging.Handler):
    def __init__(self, socket_interface, routing_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.socket_interface = socket_interface
        self.routing_id = routing_id

    def emit(self, record):
        try:
            msg = self.format(record)  # Format the log record to a string
            # capture the fatal
            if record.levelno >= logging.FATAL:
                prefix = SCRIPT_FATAL_PREFIX
            elif record.levelno >= logging.ERROR:
                prefix = SCRIPT_ERR_PREFIX  # Use a different prefix for errors
            elif record.levelno >= logging.WARNING:
                prefix = SCRIPT_WARNING_PREFIX
            else:
                prefix = SCRIPT_INFO_PREFIX  # Default prefix for non-error messages
            # Send the message with the appropriate prefix
            send_string_router(prefix + msg, self.routing_id, self.socket_interface)
        except Exception:
            self.handleError(record)


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


def start_rena_script(script_path, script_args):
    print('Script started')
    if not debugging:
        script_process = Process(target=start_script_server, args=(script_path, script_args))
        script_process.start()
        return script_process
    else:
        pickle.dump([script_path, script_args], open('start_script_args.p', 'wb'))
