import asyncio
import logging
import pickle
import sys
from concurrent import futures
from multiprocessing import Process, Pipe

import grpc
import numpy as np

from physiolabxr.configs.shared import SCRIPT_ERR_PREFIX, SCRIPT_WARNING_PREFIX, SCRIPT_INFO_PREFIX, \
    SCRIPT_FATAL_PREFIX, SCRIPT_SETUP_FAILED, INCLUDE_RPC, EXCLUDE_RPC
from physiolabxr.rpc.compiler import compile_rpc
from physiolabxr.rpc.utils import create_rpc_server
from physiolabxr.scripting.script_utils import get_script_class, debugging
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.networking_utils import recv_string_router, send_string_router, send_router


def start_script_server(script_path, script_args, conn):
    """
    This is the entry point of the script process.

    It starts the thread on which the RenaScrip lives, and also start the rpc server, whose wait functions
    spins on the main thread.

    Note that any call to logging.fatal will terminate the script process.
    """
    # redirect stdout
    rpc_outputs = script_args['rpc_outputs']
    csharp_plugin_path = script_args['csharp_plugin_path']
    stdout_port = script_args['stdout_port']
    async_shutdown_event = script_args['async_shutdown_event']

    stdout_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_STDOUT',
                                               port_id=stdout_port,
                                               identity='client',
                                               pattern='router-dealer')
    info_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INFO',
                                             port_id=0,
                                             identity='server',
                                             pattern='router-dealer')

    input_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INPUT',
                                               port_id=0,
                                               identity='server',
                                               pattern='router-dealer')
    command_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_COMMAND',
                                                 port_id=0,
                                                 identity='server',
                                                 pattern='router-dealer')

    script_args['input_socket_interface'] = input_socket_interface
    script_args['command_socket_interface'] = command_socket_interface

    stdout_port = stdout_socket_interface.binded_port
    info_port = info_socket_interface.binded_port
    input_port = input_socket_interface.binded_port
    command_port = command_socket_interface.binded_port

    conn.send([stdout_port, info_port, input_port, command_port])
    conn.close()

    logging.info('Sending message to the main app to set up the routing ID')
    stdout_socket_interface.send_string('Go')  # send an empty message, this is for setting up the routing id
    # _, stdout_routing_id = recv_string_router(stdout_socket_interface, True)

    logging.info('Waiting for info routing ID from main app for info socket')
    _, info_routing_id = recv_string_router(info_socket_interface, True)

    sys.stdout = redirect_stdout = RedirectStdout(socket_interface=stdout_socket_interface)
    sys.stderr = redirect_stderr = RedirectStderr(socket_interface=stdout_socket_interface)

    script_args['redirect_stdout'] = redirect_stdout
    script_args['redirect_stderr'] = redirect_stderr

    script_args['stdout_socket_interface'] = stdout_socket_interface
    script_args['info_socket_interface'] = info_socket_interface
    script_args['info_routing_id'] = info_routing_id

    # redirect logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    socket_handler = SocketLoggingHandler(socket_interface=stdout_socket_interface)
    socket_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(socket_handler)

    logging.info("Starting script thread")
    target_class = get_script_class(script_path)

    # compile the rpc first
    try:
        compile_rtn = compile_rpc(script_path, script_class=target_class, rpc_outputs=rpc_outputs, csharp_plugin_path=csharp_plugin_path)
        if compile_rtn is not None:
            rpc_info, is_async = compile_rtn
        else: # no rpc methods to be exposed
            rpc_info = None
            is_async = False
    except Exception as e:
        # notify the main app that the script has failed to start
        logging.fatal(f"Error compiling rpc: {e}")
        send_router(np.array([SCRIPT_SETUP_FAILED]), info_routing_id, info_socket_interface)
        return

    try:
        rena_script_thread = target_class(**script_args)
    except Exception as e:
        logging.fatal(f"Error creating script class: {e}")
        send_router(np.array([SCRIPT_SETUP_FAILED]), info_routing_id, info_socket_interface)
        return

    if rpc_info is not None:  # there is rpc methods to be exposed
        # also start the rpc
        if is_async:
            server = grpc.aio.server()
            rpc_server, port = create_rpc_server(script_path, rena_script_thread, server)
            async def check_shutdown_event(server, shutdown_event):
                while not shutdown_event.is_set():
                    await asyncio.sleep(1)
                await server.stop(0)
            async def async_serve(shutdown_event):
                await rpc_server.start()
                await check_shutdown_event(server, shutdown_event)
        else:
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
            rpc_server, port = create_rpc_server(script_path, rena_script_thread, server)
            logging.info(f"Starting rpc server listening for calls on {port}")

        rena_script_thread.rpc_server = rpc_server
        send_router(np.array([INCLUDE_RPC]), rena_script_thread.info_routing_id, rena_script_thread.info_socket_interface)
        send_router(np.array([port]).astype(int), rena_script_thread.info_routing_id, rena_script_thread.info_socket_interface)
        send_router(np.array([rpc_info]).astype('<U61'), rena_script_thread.info_routing_id, rena_script_thread.info_socket_interface)
    else:
        rpc_server = None
        send_router(np.array([EXCLUDE_RPC]), rena_script_thread.info_routing_id, rena_script_thread.info_socket_interface)

    rena_script_thread.start()
    if rpc_info is not None:  # TODO async termination should use await
        if is_async:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(async_serve(async_shutdown_event))
            logging.info("Async server stopped")
        else:
            rpc_server.start()
            rpc_server.wait_for_termination()
    logging.info('Script process terminated.')


class SocketLoggingHandler(logging.Handler):
    def __init__(self, socket_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.socket_interface = socket_interface

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
            # send_string_router(prefix + msg, self.routing_id, self.socket_interface)
            self.socket_interface.send_string(prefix + msg)
        except Exception:
            self.handleError(record)


class RedirectStderr(object):
    def __init__(self, socket_interface):
        self.terminal = sys.stderr
        self.socket_interface = socket_interface
        self.message_buffer = ""

    def write(self, message):
        self.terminal.write(message)
        self.message_buffer += message

    def flush(self):
        pass

    def send_buffered_messages(self):
        if self.message_buffer:
            self.socket_interface.send_string(SCRIPT_ERR_PREFIX + self.message_buffer)
            # send_string_router(SCRIPT_ERR_PREFIX + self.message_buffer, self.routing_id, self.socket_interface)
            self.message_buffer = ""


class RedirectStdout(object):
    def __init__(self, socket_interface):
        self.terminal = sys.stdout
        self.socket_interface = socket_interface

    def write(self, message):
        self.terminal.write(message)
        # send_string_router(SCRIPT_INFO_PREFIX + message, self.routing_id, self.socket_interface)
        self.socket_interface.send_string(SCRIPT_INFO_PREFIX + message)

    def flush(self):
        pass


def start_rena_script(script_path, script_args):
    print('Script started')
    if not debugging:
        parent_conn, child_conn = Pipe()
        script_process = Process(target=start_script_server, args=(script_path, script_args, child_conn))
        script_process.start()
        stdout_port, info_port, input_port, command_port = parent_conn.recv()

        return script_process, stdout_port, info_port, input_port, command_port
    else:
        pickle.dump([script_path, script_args], open('start_script_args.p', 'wb'))
