import logging
import os
import sys
from concurrent import futures
from typing import List

import grpc

from physiolabxr.scripting.script_utils import get_script_class_name
from physiolabxr.utils.fs_utils import load_servicer_add_function, load_file_classes, import_file
from physiolabxr.utils.networking_utils import find_available_port_from_list


def create_rpc_server(script_path, script_instance, server):
    assert script_path.endswith('.py'), 'The server script path must end with .py'
    server_script_path = script_path[:-3] + 'Server.py'
    pb2_path = script_path[:-3] + '_pb2.py'
    pb2_grpc_path = script_path[:-3] + '_pb2_grpc.py'

    script_class_name = get_script_class_name(script_path)
    script_directory = os.path.dirname(script_path)  # this is also where the generated pb2 and pb2_grpc files are located

    # append the generated path to the sys path so that <script name>_pb2_grpc can find the module named <script name>_pb2
    sys.path.append(script_directory)

    # first import the pb2, it defines the protobuf messages that are needed in the server
    # import_file(pb2_path)
    server_class = load_file_classes(server_script_path)[0]
    add_server_func = load_servicer_add_function(script_class_name, pb2_grpc_path)

    server_instance = server_class()
    server_instance.script_instance = script_instance

    add_server_func(server_instance, server)
    try:
        port = server.add_insecure_port(f"[::]:0")
    except RuntimeError:
        logging.warning(f"RPC cannot bind to a port. RPC will not be available.")
        return

    # get which port did the server bind to

    return server, port

# def run_rpc_server(script_path, script_instance, server, port):
#     rpc_server = create_rpc_server(script_path, script_instance, server, port)
#     rpc_server.start()
#     try:
#         rpc_server.wait_for_termination()
#     except Exception as e:
#         print(f"Exception while waiting for server termination: {e}")
#     finally:
#         rpc_server.stop(0)