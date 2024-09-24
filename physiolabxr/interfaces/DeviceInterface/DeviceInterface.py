from physiolabxr.utils.ConfigPresetUtils import DeviceType


class DeviceInterface:
    """
    The information
    Attributes:
        _device_name (str): The name of the device
        _device_type (DeviceType): The type of the device
        device_nominal_sampling_rate (int): The nominal sampling rate of the device
        is_supports_device_availability (bool): The availability of the device.

            If the device does not support availability, assume it is always available.
            The device availability is used to determine if the device is available to stream data.

            If you set this to true, you must also implement the is_stream_available method. Read more about it in
            the is_stream_available method's docstring.
    Tips:
    * if you create zmq sockets to communicate with a device, make sure to close them in the stop_stream method.
    """
    def __init__(self, _device_name, _device_type, device_nominal_sampling_rate=100, is_supports_device_availability=False):
        """Initialize the DeviceInterface.

        This function needs to be called in the __init__ function of the derived class.
        You will also need to pass in the correct parameters for your device when you call this function.

        Example:
        class MyAwesomeDevice(DeviceInterface):
            def __init__(self):
                super().__init__(_device_name='MyAwesomeDevice',
                                 _device_type=DeviceType.EEG,
                                 device_nominal_sampling_rate=300,
                                 is_supports_device_availability=False)
                # other initialization code here
        """
        self._device_name = _device_name
        self._device_type = _device_type
        self.device_nominal_sampling_rate = device_nominal_sampling_rate
        self.is_supports_device_availability = is_supports_device_availability
        if not is_supports_device_availability:
            self.device_available = True  # when the device does not support availability, assume it is always available
        else:
            self.device_available = False

    def start_stream(self):
        """Start the stream from the device."""
        pass

    def process_frames(self):
        """Process the frames, timestamps, and messages from the device.

        When you override this function in your derived class, you should return the frames, timestamps, and messages.
        See the following for detailed information.

        * If there is data available, return the frames, timestamps, and messages, where
            * frames is a list of frames, it should be either a list of list of numeric values or a 2D numpy array of shape (num_channels, num_timesteps)
            * timestamps is a list or a 1D numpy array of timestamps
            Note that the timestamps should be in the same order as the frames, and their length should be the same as the frames' num_timesteps.

            * messages is a list of messages. It should be a list of strings.
        * If there is no data available, return an empty list for both frames, timestamps. Messages can have values.
        * If the device has faulted or disconnected, raise an exception. This exception will be caught by
          DeviceWorker.process_on_tick() and will stop the stream. In this case, the DeviceWorker will call stop_stream().
          So make sure to clean up any resources in stop_stream() in case of an exception. The cleanup should make sure
          that the device is in a state where it can be started again.

        Returns
            Should return a tuple of frames, timestamps, and messages.
            If there are no data available, return an empty list for both frames, timestamps and messages.
            If the device has faulted or disconnected, raise an exception
        """
        pass

    def stop_stream(self):
        """Stop the stream from the device.

        This function should stop the stream from the device and clean up any resources. It should leave
        the device in a state where it can be started again by calling start_stream().
        """
        pass

    def get_device_nominal_sampling_rate(self):
        """Get the nominal sampling rate of the device."""
        return self.device_nominal_sampling_rate

    def is_stream_available(self):
        """Check if the device is available to stream data.

        If the device supports availability and you set self.is_supports_device_availability is set to true in your
        device interface, this function should return True if the device is available to stream data.
        This function is called every <physiolabxr.configs.config.stream_availability_wait_time> seconds.

        If self.is_supports_device_availability is set to False, this function is not called and the device is assumed to be always available.

        Notes:
            This function runs on the main thread and runs often. Make sure it is not blocking.
            If you are using zmq sockets, make sure to use the zmq.NOBLOCK flag when receiving data.

        """
        return self.device_available

    def __del__(self):
        """Perform any cleanup here that are not handled in stop_stream.

        This function is called when
        * the DeviceWidget is closed
        * the application is closed

        Note that you don't need to call stop_stream in this method, because the DeviceWorker would have already called
        stop_stream() when the device is disconnected.

        Tips:
        * if you open any zmq sockets for inter-process communication, make sure to close any zmq sockets&context in this method.
        * don't leave any process/thread un-joined or un-terminated.
        """
        pass