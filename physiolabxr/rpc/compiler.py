import inspect

def generate_proto_from_script(script_class):
    proto_definitions = []

    for name, method in inspect.getmembers(script_class, predicate=inspect.isfunction):
        if hasattr(method, "is_rpc_method"):  # Check if this method is marked as an RPC method
            sig = inspect.signature(method)
            parameters = sig.parameters
            return_type = sig.return_annotation
            proto_definitions.append((name, parameters, return_type))

    # Generate .proto file content based on proto_definitions
    # ...


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

def generate_proto_content(proto_definitions):
    lines = ["syntax = \"proto3\";", "service MyService {"]
    for name, parameters, return_type in proto_definitions:
        param_str = ", ".join([f"{python_type_to_proto_type(param.annotation)} {param.name}" for param in parameters.values()])
        return_str = python_type_to_proto_type(return_type)
        lines.append(f"  rpc {name}({param_str}) returns ({return_str});")
    lines.append("}")
    return "\n".join(lines)