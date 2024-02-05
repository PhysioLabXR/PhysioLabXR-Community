import bluetooth
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import re
import time
import warnings

try:
    from pylsl import StreamInfo, StreamOutlet
except:
    warnings.warn("UnicornHybridBlackDeviceInterface: pylsl is not installed, LSL interface will not work.")

class UnicornHybridBlackDeviceInterface:
    def find_unicorn():
        bt_devices = bluetooth.discover_devices(duration = 1, lookup_names = True, lookup_class = True)
        unicorn = list(filter(lambda d: re.search(r'UN-\d{4}.\d{2}.\d{2}', d[1]), bt_devices))
        if len(unicorn) == 0:
            raise Exception('No Unicorns found!')
        if len(unicorn) > 1:
            raise Exception('Multiple Unicorns found!')
        return unicorn[0]
    
    def __init__(self, stream_name, stream_type='EEG', board_id="8",
                 log='store_true', streamer_params='',
                 ring_buffer_size=45000):  # default board_id 8 for UnicornHybridBlack
        unicorn = self.find_unicorn()
        self.params = BrainFlowInputParams()
        self.params.serial_number = unicorn[1]

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.board_id = int(board_id)
        self.streamer_params = streamer_params
        self.ring_buffer_size = ring_buffer_size

        if (log):
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        try:
            self._board = BoardShim(self.board_id, self.params)
            self.info_print()
        except brainflow.board_shim.BrainFlowError:
            print('Cannot connect to board')

    def start_sensor(self):
        # tell the sensor to start sending frames
        try:
            self._board.prepare_session()
        except brainflow.board_shim.BrainFlowError:
            raise AssertionError('Unable to connect to unicorn')
        print('UnicornHybridBlackDeviceInterface: connected to sensor')

        try:
            self._board.start_stream(self.ring_buffer_size, self.streamer_params)
        except brainflow.board_shim.BrainFlowError:
            raise AssertionError('Unable to connect to device: please check the sensor connection or bluetooth stability.')
        print('UnicornHybridBlackDeviceInterface: connected to sensor')

    def process_frames(self):
        # return one or more frames of the sensor
        frames = self._board.get_board_data()

        return frames

    def stop_sensor(self):
        try:
            self._board.stop_stream()
            print('UnicornHybridBlackDeviceInterface: stopped streaming.')
            self._board.release_session()
            print('UnicornHybridBlackDeviceInterface: released session.')
        except brainflow.board_shim.BrainFlowError as e:
            print(e)


    def info_print(self):
        print("Board Information:")
        print("Sampling Rate:", self._board.get_sampling_rate(self.board_id))
        print("Board Id:", self._board.get_board_id())
        print("EEG names:", self._board.get_eeg_names(self.board_id))
        print("Package Num Channel: ", self._board.get_package_num_channel(self.board_id))
        print("EEG Channels:", self._board.get_eeg_channels(self.board_id))
        print("Accel Channels: ", self._board.get_accel_channels(self.board_id))
        print("Other Channels:", self._board.get_other_channels(self.board_id))
        print("Analog Channels: ", self._board.get_analog_channels(self.board_id))
        print("TimeStamp: ", self._board.get_timestamp_channel(self.board_id))
        print("Marker Channel: ", self._board.get_marker_channel(self.board_id))

    def get_sampling_rate(self):
        return self._board.get_sampling_rate(self.board_id)