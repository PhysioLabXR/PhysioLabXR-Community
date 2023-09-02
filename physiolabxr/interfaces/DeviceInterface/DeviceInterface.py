from physiolabxr.utils.ConfigPresetUtils import DeviceType


class DeviceInterface:

    def __init__(self, _device_name, _device_type, device_nominal_sampling_rate=100):
        self._device_name = _device_name
        self._device_type = _device_type
        self.device_nominal_sampling_rate = device_nominal_sampling_rate

    def start_stream(self):
        pass

    def process_frames(self):
        pass
        # return np.array(frames), timestamps

    def stop_stream(self):
        pass

    def get_device_nominal_sampling_rate(self):
        return self.device_nominal_sampling_rate

    def is_stream_available(self):
        return True