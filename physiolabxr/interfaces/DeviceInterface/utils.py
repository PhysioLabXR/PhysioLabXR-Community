from physiolabxr.exceptions.exceptions import FailToSetupDevice


def create_custom_device_interface(stream_name):
    if stream_name == 'UnicornHybridBlackBluetooth':
        try:
            import bluetooth
        except ImportError:
            raise FailToSetupDevice('Bluetooth module is not available.'
                            'Please install the bluetooth module by running "pip install pybluez"')
        from physiolabxr.interfaces.DeviceInterface.UnicornHybridBlackDeviceInterface import UnicornHybridBlackDeviceInterface
        interface = UnicornHybridBlackDeviceInterface()
        return interface
    elif stream_name == 'DSI24':
        from physiolabxr.interfaces.DeviceInterface.DSI24.DSI24Interface import DSI24Interface
        interface = DSI24Interface()
        return interface
    else:
        raise NotImplementedError(f"CustomDeviceInterface: {stream_name} is not implemented")

