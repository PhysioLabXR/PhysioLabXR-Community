import inspect
import logging
import os
import shutil
import subprocess
import sys

from typing import get_type_hints, Union

from physiolabxr.configs.shared import grpc_deprecation_warning
from physiolabxr.exceptions.exceptions import CompileRPCError
from physiolabxr.presets.PresetEnums import RPCLanguage


def python_type_to_proto_type(python_type):
    """
    Convert a python type to a proto type
    """
    mapping = {
        int: "int32",
        float: "float",
        str: "string",
        bool: "bool"
    }
    # raise an error if the type is not in the mapping
    try:
        return mapping[python_type]
    except KeyError:
        raise CompileRPCError(f"Unsupported type '{python_type.__name__}' in RPC method")

def get_args_return_from_type_hints(name, method):
    type_hints = get_type_hints(method)

    # warn if the self argument is is typehinted
    if "self" in type_hints:
        logging.warning(f"RPC method {name} has a type hint for self, this is not necessary")

    # get rid of the self argument
    type_hints = {k: v for k, v in type_hints.items() if k != "self"}

    all_args = [arg for arg in inspect.signature(method).parameters if arg != "self"]
    # Check for missing type hints
    missing_type_hints = [arg for arg in all_args if arg not in type_hints]

    if missing_type_hints:
        message = f"RPC method {name} is missing type hints for argument(s): {', '.join(missing_type_hints)}"
        raise CompileRPCError(message)

    # get in input args that are not returns
    input_args = {k: v for k, v in type_hints.items() if k != "return"}
    if "return" in type_hints and type_hints["return"] is not None:
        returns = type_hints["return"]
        has_return = True
    else:
        returns = None
        has_return = False
    has_args = len(input_args) > 0  # Excluding self and potentially return

    return input_args, returns, has_return, has_args

def generate_proto_from_script_class(cls):

    proto_lines = ["syntax = \"proto3\";", 'import "google/protobuf/empty.proto";']
    service_methods = []
    messages = []
    cls_name = cls.__name__
    rpcs = []  # keep track of the rpcs that are found
    # get the name/methods that are rpc
    rpc_methods = [(name, method) for name, method in inspect.getmembers(cls, predicate=inspect.isfunction) if hasattr(method, "is_rpc_method") or hasattr(method, "is_async_rpc_method")]
    if len(rpc_methods) == 0:
        return None
    for name, method in rpc_methods:
        rpc_info = [name, [], []]  # name, input_args, returns
        # Generate RPC service method definition
        request_name = f"{name}Request"
        response_name = f"{name}Response"

        input_args, returns, has_return, has_args = get_args_return_from_type_hints(name, method)

        request_fields = []
        response_fields = []

        # Handle no input scenario
        request_type = request_name if has_args else "google.protobuf.Empty"
        response_type = response_name if has_return else "google.protobuf.Empty"
        service_methods.append(f"  rpc {name}({request_type}) returns ({response_type});")

        # Generate request and response messages if necessary
        if has_args:
            for i, (arg_name, arg_type) in enumerate(input_args.items(), start=1):
                try:
                    protobuf_type = python_type_to_proto_type(arg_type)
                except CompileRPCError as e:
                    logging.error(f"RPC method {name} has unsupported type hint for argument {arg_name}")
                    raise e
                request_fields.append(f"  {protobuf_type} {arg_name} = {i};")
                rpc_info[1].append(f"{arg_name}: {arg_type.__name__}")
            messages.append(f"message {request_name} {{\n" + "\n".join(request_fields) + "\n}")
            rpc_info[1] = f"{request_type} [" + ", ".join(rpc_info[1]) + "]"
        else:
            rpc_info[1] = f"Empty"

        if has_return:
            # if hasattr(type_hints["return"], '__iter__'):
            try:
                if isinstance(returns, tuple) or isinstance(returns, list):
                    for j, arg_type in enumerate(returns):
                        protobuf_type = python_type_to_proto_type(arg_type)
                        response_fields.append(f"  {protobuf_type} message{j} = {j+1};")
                        rpc_info[2].append((f"message{j}: {arg_type.__name__}"))
                    rpc_info[2] = f"{response_type} [" + ", ".join(rpc_info[2]) + "]"
                else:
                    protobuf_type = python_type_to_proto_type(returns)
                    response_fields.append(f"  {protobuf_type} message = 1;")
                    rpc_info[2] = f"{response_type} [message: {returns.__name__}]"

                messages.append(f"message {response_name} {{\n" + "\n".join(response_fields) + "\n}")
            except CompileRPCError as e:
                logging.error(f"RPC method {name} has unsupported type hint for its return")
                raise e
        else:
            rpc_info[2] = f"Empty"
            logging.info(f"No return type for RPC method {name}, if this is intentional, ignore this message.")
        logging.info(f"Generated RPC method {name} with request fields {request_fields} and response type {response_fields}")
        rpcs.append(rpc_info)

    proto_lines.append(f"service {cls_name} {{")
    proto_lines.extend(service_methods)
    proto_lines.append("}")
    proto_lines.extend(messages)

    return "\n".join(proto_lines), rpcs

def generate_server_code(script_path, script_class):
    """Generate a proto file from a class with rpc methods

    Returns:
        is_async: bool: True if the class has an async rpc method, False otherwise
    """
    script_directory_path = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)[:-3]
    server_file_path = os.path.join(script_directory_path, f"{script_name}Server.py")

    # Start generating server code
    server_code = ["from google.protobuf import empty_pb2",
                   "from google.protobuf.json_format import MessageToDict",
                   f"import {script_name}_pb2_grpc, {script_name}_pb2",
                   "",
                   f"class {script_class.__name__}Server({script_name}_pb2_grpc.{script_class.__name__}Servicer):",
                   "    script_instance = None"]

    # Iterate over RPC methods to add to server class
    is_async = False
    for name, method in inspect.getmembers(script_class, predicate=inspect.isfunction):
        if hasattr(method, "is_rpc_method") or hasattr(method, "is_async_rpc_method"):
            is_async = hasattr(method, "is_async_rpc_method")
            if is_async:
                server_code.append(f"    async def {name}(self, request, context):")
            else:
                server_code.append(f"    def {name}(self, request, context):")
            input_args, returns, has_return, has_args = get_args_return_from_type_hints(name, method)

            if has_args:
                server_code.append(f"        result = self.script_instance.{name}(**MessageToDict(request))")
            else:
                server_code.append(f"        result = self.script_instance.{name}()")

            if has_return:
                if isinstance(returns, tuple) or isinstance(returns, list):
                    return_arg_string = ", ".join([f"message{j}=result[{j}]" for j in range(len(returns))])
                    server_code.append(f"        return {script_name}_pb2.{name}Response({return_arg_string})")
                else:
                    server_code.append(f"        return {script_name}_pb2.{name}Response(message=result)")
            else:
                server_code.append("        return empty_pb2.Empty()")

            server_code.append("")

    # Write the server code to a file
    with open(server_file_path, 'w') as server_file:
        server_file.write("\n".join(server_code))

    logging.info(f"Server code generated at {server_file_path}")
    return is_async

def get_grpc_compile_command(proto_file_path, language, location, csharp_plugin_path):

    if language == RPCLanguage.PYTHON:
        return [
            sys.executable, '-m', 'grpc_tools.protoc',
            '-I.',  # Include the current directory in the search path.
            f'--{language.get_command_str()}_out={location}',  # Output directory for generated Python code.
            f'--grpc_{language.get_command_str()}_out={location}',  # Output directory for generated gRPC code.
            os.path.basename(proto_file_path)  # The .proto file to compile.
        ]
    elif language == RPCLanguage.CSHARP:  # TODO: Add support for C# compilation
        assert csharp_plugin_path is not None, "C# plugin path is required for C# compilation"
        return [
            "protoc", "-I.",
            f"--{language.get_command_str()}_out={location}",
            f"--grpc_{language.get_command_str()}_out={location}",
            f"--plugin=protoc-gen-grpc_csharp={csharp_plugin_path}",
            os.path.basename(proto_file_path)
        ]


def compile_rpc(script_path, csharp_plugin_path=None, script_class=None, rpc_outputs: list=None):
    """Compile the RPC methods in a script class

    Returns:
        rpc_info: list: List of RPC methods that were compiled
        is_async: bool: True if the class has an async rpc method, False otherwise
    """
    assert os.path.exists(script_path)
    assert script_path.endswith('.py'), "File name must end with .py"
    if script_class is None:
        from physiolabxr.scripting.script_utils import get_script_class
        script_class = get_script_class(script_path)

    script_directory_path = os.path.dirname(script_path)

    # process the rpc outputs
    if rpc_outputs is None:
        # default is to add a python output to the script directory
        rpc_outputs = [{"language": RPCLanguage.PYTHON, "location": '.'}]
    else:
        # python output is always needed for the server
        # add the python and . location to the outputs if it is not already there
        python_output = {"language": RPCLanguage.PYTHON, "location": '.'}
        if python_output not in rpc_outputs:
            rpc_outputs.append(python_output)

    logging.info(f"Compiling RPC for {script_class.__name__} in {script_path}")
    proto_rtn = generate_proto_from_script_class(script_class)
    if proto_rtn is None:
        logging.info(f"No RPC methods found in {script_class.__name__}, skipping compilation")
        return None
    proto_content, rpc_info = proto_rtn
    # save the proto content to the same directory as the script
    script_name = os.path.basename(script_path)[:-3]
    proto_file_path = os.path.join(os.path.dirname(script_path), f"{script_name}.proto")
    with open(proto_file_path, "w") as f:
        f.write(proto_content)
    logging.info(f"Successfully generated proto file {proto_file_path} from {script_class.__name__}, saved to {proto_file_path}")

    if len(rpc_outputs) == 0:
        logging.warning("")

    # call grpc compile on the proto content
    for rpc_output in rpc_outputs:
        language = rpc_output["language"]
        location = rpc_output["location"]

        command = get_grpc_compile_command(proto_file_path, language, location, csharp_plugin_path)
        logging.info("Compiling proto file with command: " + " ".join(command))
        logging.info("Current working directory is " + os.getcwd())
        logging.info("Compile working directory is " + script_directory_path)
        result = subprocess.run(command, cwd=script_directory_path, check=True, capture_output=True)
        # error = result.stderr.decode('utf-8').strip(grpc_deprecation_warning)
        if result.returncode != 0:
            message = f"Error compiling the proto file. \nstderr: \n{result.stderr.decode('utf-8')} \nstdout: \n{result.stdout.decode('utf-8')}"
            raise CompileRPCError(message)
        else:
            logging.info(f"{proto_file_path} file compiled successfully. Generated files are in {script_directory_path}")

    # generate the server code #############################################
    is_async = generate_server_code(script_path, script_class)

    # return the list of rpcs that were compiled
    return rpc_info, is_async