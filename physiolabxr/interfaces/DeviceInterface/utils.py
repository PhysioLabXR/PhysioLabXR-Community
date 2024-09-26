import importlib


def create_custom_device_classes(device_name):
    """Dynamically imports and creates a custom device interface based on the stream name.

    The call stack should be
                DeviceWidget.__init__() -> DeviceWorker.__init() -> this function

    Assumes that:
    - Device plugins are located in physiolabxr/interfaces/DeviceInterface/<DeviceName>/
    - The plugin should include
        - interface class extending DeviceInterface class: <DeviceName>_Interface in the file physiolabxr/interfaces/DeviceInterface/<DeviceName>/<DeviceName>_Interface.py.
        - options class extending QWidget: <DeviceName>_Options in the file physiolabxr/interfaces/DeviceInterface/<DeviceName>/<DeviceName>_Options.py.

    Args:
        stream_widget (QWidget): The widget that will contain the device interface.
        stream_name (str): The name of the device stream.

    Returns:
        An instance of the relevant device interface class.
        An instance of the relevant device options class, if it is defined. This will be None if the options class is not defined.
    """
    module_name = f"physiolabxr.interfaces.DeviceInterface.{device_name}"
    try:
        # Define the path to the device's module based on the stream_name
        # Dynamically import the module containing the interface class
        module = importlib.import_module(module_name)
        # Dynamically get the class from the module
        device_interface_class = getattr(module, f"{device_name}_Interface")
    except ModuleNotFoundError as e:
        raise NotImplementedError(
            f"create_custom_device_classes: DeviceInterface class is "
            f"{device_name}_Interface is not implemented under {module_name}: {e}")
    # Instantiate and return the interface object
    try:
        interface_instance = device_interface_class()
    except Exception as e:
        raise Exception(f"Error creating device interface for {device_name}: {e}")

    try:
        options_class = getattr(module, f"{device_name}_Options")
    except ModuleNotFoundError as e:
        options_class = None
        print(f"Options class not found for {device_name}, will not have options UI for this device.")

    return interface_instance, options_class

