from physiolabxr.utils.ConfigPresetUtils import DeviceType


class DeviceInterface:
    """

    Tips:
    * if you create zmq sockets to communicate with a device, make sure to close them in the stop_stream method.
    """

    def __init__(self, _device_name, _device_type, device_nominal_sampling_rate=100, device_available=True):
        self._device_name = _device_name
        self._device_type = _device_type
        self.device_nominal_sampling_rate = device_nominal_sampling_rate
        self.device_available = device_available

    def start_stream(self):
        pass

    def process_frames(self):
        """

        Returns
            Should return a tuple of frames and timestamps
            If there are no data available, return an empty list for both frames and timestamps
            If the device has faulted or disconnected, raise an exception
        """
        pass

    def stop_stream(self):
        pass

    def get_device_nominal_sampling_rate(self):
        return self.device_nominal_sampling_rate

    def is_stream_available(self):
        return self.device_available

    def __del__(self):
        """
        You don't need to call stop_stream in this method, this is already handled in CustomDeviceWorker.
        Make sure to close any zmq sockets in this method.
        """
        pass