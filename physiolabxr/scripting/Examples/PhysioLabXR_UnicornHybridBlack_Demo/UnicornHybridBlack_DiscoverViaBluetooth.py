# PhysioLabXR Unicorn Hybrid Black Script Example
# Feb 2024, Jace Li & Haowen 'John' Wei, Columbia University (yl4862@columbia.edu, hw2892@columbia.edu)

# This is an example script for PhysiolabXR for the g.tec Unicorn Hybrid Black.
# First, it searches for connected Unicorns via Bluetooth.
#   Note: if no Unicorns are found, or more than one is found, an exception is raised.
# Then, it initializes the Unicorn using Brainflow's built-in interface.
# Then, it streams all channels to PhysiolabXR via LSL.
#   Note: the output stream name is "Unicorn Hybrid Black".
#         remember to add this to the scripting output section in PhysioLabXR!

import re
import time

try:
    import bluetooth
except ImportError:
    print('Bluetooth module is not available')
    print('Please install the bluetooth module by running "pip install pybluez"')
    raise


from brainflow import BrainFlowError
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import pylsl
import warnings
from physiolabxr.scripting.RenaScript import RenaScript


# def find_unicorn():
#     while (True):
#         bt_devices = bluetooth.discover_devices(
#             duration=1, lookup_names=True, lookup_class=True)
#         unicorn = list(filter(lambda d: re.search(
#             r'UN-\d{4}.\d{2}.\d{2}', d[1]), bt_devices))
#         if len(unicorn) == 0:
#             print('No Unicorns found!')
#         elif len(unicorn) > 1:
#             print('Multiple Unicorns found!')
#         elif len(unicorn) == 1:
#             return unicorn[0]


class UnicornHybridBlackBluetoothDataStreamScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):


        print("init() called, searching for Unicorn...")
        while (True):
            bt_devices = bluetooth.discover_devices(
                duration=1, lookup_names=True, lookup_class=True)
            unicorns = list(filter(lambda d: re.search(
                r'UN-\d{4}.\d{2}.\d{2}', d[1]), bt_devices))
            if len(unicorns) == 0:
                print('No Unicorns found!')
                continue
            elif len(unicorns) > 1:
                print('Multiple Unicorns found!')
                continue
            elif len(unicorns) == 1:
                unicorn = unicorns[0]
                print('Connected to Unicorn! SN: ' + unicorn[1])
                print('Initializing BrainFlow board...')
                self.brainflow_input_params = BrainFlowInputParams()
                self.brainflow_input_params.serial_number = unicorn[1]
                self.board_id = BoardIds.UNICORN_BOARD.value
                self.board = BoardShim(self.board_id, self.brainflow_input_params)
                try:
                    self.board.prepare_session()
                    break
                except BrainFlowError as e:
                    # print error message
                    print(e)
                    continue
        self.board.start_stream(45000, '')
        print('BrainFlow board initialized! Stream started!')

    # loop is called <Run Frequency> times per second

    def loop(self):
        timestamp_channel = self.board.get_timestamp_channel(self.board_id)
        accelerometer_channels = self.board.get_accel_channels(self.board_id)
        gyroscope_channels = self.board.get_gyro_channels(self.board_id)
        battery_channel = self.board.get_battery_channel(self.board_id)
        eeg_channels = self.board.get_eeg_channels(self.board_id)

        # print(f"Timestamp channel: {timestamp_channel}")
        # print(f"EEG channels: {eeg_channels}")
        # print(f"Accelerometer channels: {accelerometer_channels}")
        # print(f"Gyroscope channels: {gyroscope_channels}")
        # print(f"Battery channel: {battery_channel}")

        data = self.board.get_board_data()
        timestamps = data[timestamp_channel]
        eeg_data = data[eeg_channels]
        accelerometer_data = data[accelerometer_channels]
        gyroscope_data = data[gyroscope_channels]
        battery_data = data[battery_channel]

        absolute_time_to_lsl_time_offset = time.time() - pylsl.local_clock()
        timestamps = timestamps - absolute_time_to_lsl_time_offset

        self.set_output(stream_name="UnicornHybridBlackLSL",
                        data=data, timestamp=timestamps)
        # print('Data sent to LSL!')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
