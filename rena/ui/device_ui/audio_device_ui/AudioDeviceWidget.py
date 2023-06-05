from rena.interfaces.AudioInputInterface import RenaAudioInputInterface
from rena.presets.presets_utils import get_audio_device_channel_num, get_audio_device_index, get_audio_device_data_format, get_audio_device_sampling_rate, get_audio_device_frames_per_buffer
from rena.threadings.DeviceWorker import DeviceWorker
from rena.ui.device_ui.DeviceWidget import DeviceWidget


class AudioDeviceWidget(DeviceWidget):

    def __init__(self, parent_widget, parent_layout, stream_name, data_type, worker, networking_interface, port_number,
                 insert_position):
        super().__init__(parent_widget, parent_layout, stream_name, data_type, worker, networking_interface,
                         port_number, insert_position)


    def init_device_worker(self):
        if self.worker.is_streaming:
            self.worker.stop_stream()

        # audio_device_interface = RenaAudioInputInterface(stream_name=self.stream_name,
        #                                     audio_device_index=get_audio_device_index(self.stream_name),
        #                                     channels=get_audio_device_channel_num(self.stream_name),
        #                                     frames_per_buffer=get_audio_device_frames_per_buffer(self.stream_name),
        #                                     data_format=get_audio_device_data_format(self.stream_name),
        #                                     sampling_rate=get_audio_device_sampling_rate(self.stream_name))
        #
        # self.worker = DeviceWorker(device_interface=audio_device_interface)


