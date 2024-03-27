try:
    import bluetooth
except ImportError:
    print('Bluetooth module is not available')
    print('Please install the bluetooth module by running "pip install pybluez"')

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import re
import time
import warnings
from physiolabxr.configs.GlobalSignals import GlobalSignals

from physiolabxr.exceptions.exceptions import CustomDeviceNotFoundError, CustomDeviceStartStreamError, \
    CustomDeviceStreamInterruptedError
from physiolabxr.interfaces.DeviceInterface.DeviceInterface import DeviceInterface
import pylsl

try:
    from pylsl import StreamInfo, StreamOutlet
except:
    warnings.warn("UnicornHybridBlackDeviceInterface: pylsl is not installed, LSL interface will not work.")


class UnicornHybridBlackDeviceInterface(DeviceInterface):

    def __init__(self,
                 _device_name='UnicornHybridBlackBluetooth',
                 _device_type='UnicornHybridBlack',
                 _device_nominal_sampling_rate=250,
                 board_id="8",
                 log='store_true',
                 streamer_params='',
                 ring_buffer_size=45000):  # default board_id 8 for UnicornHybridBlack
        super(UnicornHybridBlackDeviceInterface, self).__init__(_device_name=_device_name,
                                                                _device_type=_device_type,
                                                                device_nominal_sampling_rate=_device_nominal_sampling_rate,
                                                                device_available=False)

        self.stream_name = _device_name
        self.stream_type = _device_type
        self.board_id = int(board_id)
        self.streamer_params = streamer_params
        self.ring_buffer_size = ring_buffer_size

        if (log):
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

    def start_stream(self):

        # check if bluetooth module has been imported
        if 'bluetooth' not in globals():
            raise CustomDeviceStartStreamError('Bluetooth module is not available. \n'
                                               'Note: the Pybluez module with g.tec Unicorn only works on Windows.')

        bt_devices = bluetooth.discover_devices(duration=1, lookup_names=True, lookup_class=True)
        unicorns = list(filter(lambda d: re.search(r'UN-\d{4}.\d{2}.\d{2}', d[1]), bt_devices))
        if len(unicorns) == 0:
            raise CustomDeviceStartStreamError('No Unicorns found!')
        elif len(unicorns) > 1:
            raise CustomDeviceStartStreamError('Multiple Unicorns found! Please ensure only one is connected!')
        elif len(unicorns) == 1:
            unicorn = unicorns[0]  # Unpack list into list[3]

        self.params = BrainFlowInputParams()
        self.params.serial_number = unicorn[1]
        try:
            self._board = BoardShim(self.board_id, self.params)
            # self.info_print()
        except brainflow.board_shim.BrainFlowError:
            raise CustomDeviceStartStreamError('BoardShim constructor error. Check connection.')

        # tell the sensor to start sending frames
        if not self._board.is_prepared():
            try:
                self._board.prepare_session()
            except brainflow.board_shim.BrainFlowError:
                raise CustomDeviceStartStreamError('prepare_session error. Check connection.')
            print('UnicornHybridBlackDeviceInterface: connected to sensor')
        else:
            print("Unicorn BrainFlow previously prepared, skipping prepare_session()")

        try:
            self._board.start_stream(self.ring_buffer_size, self.streamer_params)
        except brainflow.board_shim.BrainFlowError:
            raise CustomDeviceStartStreamError(
                'Unable to connect start stream: please check the sensor connection or bluetooth stability.')

        # Successful connection
        self.device_available = True
        self.data_started = False
        print('UnicornHybridBlackDeviceInterface: connected to sensor')

    def process_frames(self):
        # return one or more frames of the sensor
        frames = self._board.get_board_data()
        # print(frames)
        if not self.data_started and frames.size != 0:
            print('UnicornHybridBlackDeviceInterface: data started')
            self.data_started = True

        # If connection interrupted, get_board_data() returns []
        if self.data_started and frames.size == 0:
            print('Unicorn connection interrupted, stopping stream automatically!')
            self.data_started = False
            self.device_available = False
            self.device_worker.device_widget.start_stop_stream_btn_clicked()
            GlobalSignals().show_notification_signal.emit({
                'title': 'Unicorn Connection Lost',
                'body': 'Lost connection to {0}'
                .format(self.params.serial_number)
            })

        timestamp_channel = self._board.get_timestamp_channel(self.board_id)
        timestamps = frames[timestamp_channel]

        absolute_time_to_lsl_time_offset = time.time() - pylsl.local_clock()
        timestamps = timestamps - absolute_time_to_lsl_time_offset

        return frames, timestamps

    def stop_stream(self):
        try:
            if self.device_available:
                self._board.stop_stream()
                print('UnicornHybridBlackDeviceInterface: stopped streaming.')
            self._board.release_session()
            print('UnicornHybridBlackDeviceInterface: released session.')
            self.data_started = False
        except brainflow.board_shim.BrainFlowError as e:
            # print(e)
            pass

    def is_stream_available(self):
        # unicorn = self.find_unicorn()
        return self.device_available

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


def create_custom_device_interface(stream_name):
    if stream_name == 'UnicornHybridBlackBluetooth':
        interface = UnicornHybridBlackDeviceInterface()
        return interface

    # interface = UnicornHybridBlackDeviceInterface()
    # return interface

# def create_unicorn_hybrid_black_interface():
#     interface = UnicornHybridBlackDeviceInterface()
#     return interface
