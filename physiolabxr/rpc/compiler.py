import inspect
import os
import shutil
import subprocess

from physiolabxr.scripting.script_utils import get_script_class
from typing import get_type_hints, Union


def python_type_to_proto_type(python_type):
    # Simplistic mapping, expand according to your needs
    mapping = {
        int: "int32",
        float: "float",
        str: "string",
        bool: "bool"
        # Add more mappings as necessary
    }
    return mapping.get(python_type, "string")  # Default to string if type not found

def generate_proto_from_script_class(cls):
    proto_lines = ["syntax = \"proto3\";", 'import "google/protobuf/empty.proto";']
    service_methods = []
    messages = []

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if hasattr(method, "is_rpc_method"):  # Assuming @rpc decorator sets this attribute
            # Generate RPC service method definition
            request_name = f"{name}Request"
            response_name = f"{name}Response"

            type_hints = get_type_hints(method)
            # get rid of the self argument
            type_hints = {k: v for k, v in type_hints.items() if k != "self"}

            # get in input args that are not returns
            input_args = {k: v for k, v in type_hints.items() if k != "return"}

            request_fields = []
            response_fields = []
            has_return = "return" in type_hints and type_hints["return"] is not None
            has_args = len(input_args) > 0  # Excluding self and potentially return

            # Handle no input scenario
            request_type = request_name if has_args else "google.protobuf.Empty"
            response_type = response_name if has_return else "google.protobuf.Empty"
            service_methods.append(f"  rpc {name}({request_type}) returns ({response_type});")

            # Generate request and response messages if necessary
            if has_args:
                for i, (arg_name, arg_type) in enumerate(input_args.items(), start=1):
                    protobuf_type = python_type_to_proto_type(arg_type)
                    request_fields.append(f"  {protobuf_type} {arg_name} = {i};")
                messages.append(f"message {request_name} {{\n" + "\n".join(request_fields) + "\n}")

            if has_return:
                if hasattr(type_hints["return"], "__origin__") and type_hints["return"].__origin__ is Union:
                    for j, union_arg_type in enumerate(type_hints["return"].__args__, start=1):
                        protobuf_type = python_type_to_proto_type(union_arg_type)
                        response_fields.append(f"  {protobuf_type} message{j-1} = {j};")
                else:
                    protobuf_type = python_type_to_proto_type(type_hints["return"])
                    response_fields.append(f"  {protobuf_type} message = 1;")
                messages.append(f"message {response_name} {{\n" + "\n".join(response_fields) + "\n}")

            # Generate request message
            # request_fields = []
            # response_fields = ["  string message = 1;"]  # Example response field, adjust as needed
            # type_hints = get_type_hints(method)
            # for i, (arg_name, arg_type) in enumerate(type_hints.items(), start=1):
            #     if arg_name == "return":
            #         continue  # Skip return type
            #     protobuf_type = python_type_to_proto_type(arg_type)
            #     request_fields.append(f"  {protobuf_type} {arg_name} = {i};")
            #
            # messages.append(f"message {request_name} {{\n" + "\n".join(request_fields) + "\n}")
            # messages.append(f"message {response_name} {{\n" + "\n".join(response_fields) + "\n}")

            # for i, (arg_name, arg_type) in enumerate(type_hints.items(), start=1):
            #     if arg_name == "return":
            #         # Handle multiple return types represented by Union
            #         if hasattr(arg_type, "__origin__") and arg_type.__origin__ is Union:
            #             for j, union_arg_type in enumerate(arg_type.__args__, start=1):
            #                 protobuf_type = python_type_to_proto_type(union_arg_type)
            #                 response_fields.append(f"  {protobuf_type} message{j - 1} = {j};")
            #         else:
            #             protobuf_type = python_type_to_proto_type(arg_type)
            #             response_fields.append(f"  {protobuf_type} message = 1;")
            #     else:
            #         protobuf_type = python_type_to_proto_type(arg_type)
            #         request_fields.append(f"  {protobuf_type} {arg_name} = {i};")
            #
            # messages.append(f"message {request_name} {{\n" + "\n".join(request_fields) + "\n}")
            # messages.append(f"message {response_name} {{\n" + "\n".join(response_fields) + "\n}")

    proto_lines.append("service MyService {")
    proto_lines.extend(service_methods)
    proto_lines.append("}")
    proto_lines.extend(messages)

    return "\n".join(proto_lines)


def compile_rpc(script_path, script_class=None):
    assert os.path.exists(script_path)
    assert script_path.endswith('.py'), "File name must end with .py"
    if script_class is None:
        script_class = get_script_class(script_path)
    script_directory_path = os.path.dirname(script_path)

    proto_content = generate_proto_from_script_class(script_class)
    # save the proto content to the same directory as the script
    script_name = os.path.basename(script_path)[:-3]
    proto_file_path = os.path.join(os.path.dirname(script_path), f"{script_name}.proto")
    with open(proto_file_path, "w") as f:
        f.write(proto_content)

    # call grpc compile on the proto content
    command = [
        'python', '-m', 'grpc_tools.protoc',
        '-I.',  # Include the current directory in the search path.
        f'--python_out=.',  # Output directory for generated Python code.
        f'--grpc_python_out=.',  # Output directory for generated gRPC code.
        os.path.basename(proto_file_path)  # The .proto file to compile.
    ]
    result = subprocess.run(command, cwd=script_directory_path, check=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("Error compiling .proto file:", result.stderr)
    else:
        print(f"{proto_file_path} file compiled successfully.")

    # generate the server code
