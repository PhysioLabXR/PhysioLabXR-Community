import os
import subprocess
import sys
import warnings

import numpy as np

from exceptions.exceptions import ChannelMismatchError
from rena.interfaces import LSLInletInterface
from rena.interfaces.OpenBCILSLInterface import OpenBCILSLInterface
from rena.interfaces.MmWaveSensorLSLInterface import MmWaveSensorLSLInterface


def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def get_fps(queue):
    try:
        return len(queue) / (queue[-1] - queue[0])
    except ZeroDivisionError:
        return 0


def create_lsl_interface(lsl_name, channel_names):
    # try:
    #     interface = LSLInletInterface.LSLInletInterface(lsl_name, len(channel_names))
    # except AttributeError:
    #     raise AssertionError('Unable to find LSL Stream in LAN.')
    interface = LSLInletInterface.LSLInletInterface(lsl_name, len(channel_names))
    return interface


def process_preset_create_openBCI_interface_startsensor(device_name, serial_port, board_id):
    try:
        interface = OpenBCILSLInterface(stream_name=device_name,
                                        serial_port=serial_port,
                                        board_id=board_id,
                                        log='store_false', )
    except AssertionError as e:
        raise AssertionError(e)

    return interface


def process_preset_create_TImmWave_interface_startsensor(num_range_bin, Dport, Uport, config_path):
    # create interface
    interface = MmWaveSensorLSLInterface(num_range_bin=num_range_bin)
    # connect Uport Dport

    try:
        interface.connect(uport_name=Uport, dport_name=Dport)
    except AssertionError as e:
        raise AssertionError(e)

    # send config
    try:
        if not os.path.exists(config_path):
            raise AssertionError('The config file Does not exist: ', str(config_path))

        interface.send_config(config_path)

    except AssertionError as e:
        raise AssertionError(e)

    # start mmWave 6843 sensor
    try:
        interface.start_sensor()
    except AssertionError as e:
        raise AssertionError(e)

    return interface


# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class DataBuffer():
    def __init__(self, stream_buffer_sizes: dict = None):
        self.buffer = dict()
        self.data_type_buffer_sizes = stream_buffer_sizes if stream_buffer_sizes else dict()

    def update_buffer(self, data_dict: dict):
        if len(data_dict) > 0:
            self._update_buffer(data_dict['stream_name'], data_dict['frames'], data_dict['timestamps'])
            # data_type = data_dict['stream_name']  # get the name of the newly-come data
            #
            # if data_type not in self.buffer.keys():
            #     self.buffer[data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
            #                               np.empty(shape=(0,))]  # data first, timestamps second
            # buffered_data = self.buffer[data_dict['stream_name']][0]
            # buffered_timestamps = self.buffer[data_dict['stream_name']][1]
            #
            # self.buffer[data_type][0] = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
            # self.buffer[data_type][1] = np.concatenate([buffered_timestamps, data_dict['timestamps']])
            #
            # if data_type in self.data_type_buffer_sizes.keys():  # keep only the latest data according to the buffer size
            #     buffer_time_points = self.buffer[data_type][0].shape[-1]
            #     cut_to = -np.min([buffer_time_points, self.data_type_buffer_sizes[data_type]])
            #     self.buffer[data_type][0] = self.buffer[data_type][0][:, cut_to:]
            #     self.buffer[data_type][1] = self.buffer[data_type][1][cut_to:]

    def update_buffers(self, data_buffer):
        for stream_name, (frames, timestamps) in data_buffer.items():
            self._update_buffer(stream_name, frames, timestamps)

    def _update_buffer(self, stream_name, frames, timestamps):

        if stream_name not in self.buffer.keys():
            self.buffer[stream_name] = [np.empty(shape=(frames.shape[0], 0)),
                                      np.empty(shape=(0,))]  # data first, timestamps second
        buffered_data = self.buffer[stream_name][0]
        buffered_timestamps = self.buffer[stream_name][1]

        self.buffer[stream_name][0] = np.concatenate([buffered_data, frames], axis=-1)
        self.buffer[stream_name][1] = np.concatenate([buffered_timestamps, timestamps])

        if stream_name in self.data_type_buffer_sizes.keys():  # keep only the latest data according to the buffer size
            buffer_time_points = self.buffer[stream_name][0].shape[-1]
            cut_to = -np.min([buffer_time_points, self.data_type_buffer_sizes[stream_name]])
            self.buffer[stream_name][0] = self.buffer[stream_name][0][:, cut_to:]
            self.buffer[stream_name][1] = self.buffer[stream_name][1][cut_to:]

    def clear_buffer(self) -> None:
        self.buffer = dict()

    def clear_stream_buffer(self, stream_name: str) -> None:
        try:
            self.buffer.pop(stream_name)
        except KeyError:
            warnings.warn(f'Unable to clear the buffer for stream name {stream_name}, key not found')

    def clear_stream_buffer_data(self, stream_name: str) -> None:
        """
        Remove the buffered data for a stream without removing the existing keys.
        The data and timestamps array will instead become empty arraries
        """
        try:
            self.buffer[stream_name][0] = np.empty([self.buffer[stream_name][0].shape[0], 0]) # event marker
            self.buffer[stream_name][1] = np.array([]) # timestamp
        except KeyError:
            warnings.warn(f'Unable to clear the buffer for stream name {stream_name}, key not found')

    def clear_buffer_data(self) -> None:
        """
        Remove buffered data for all streams without removing the existing keys.
        The data and timestamps array will instead become empty arraries
        """
        for stream_name in self.buffer.keys():
            self.clear_stream_buffer_data(stream_name)
    def clear_up_to(self, timestamp, ignores=()):
        """
        The resulting timestamp is guaranteed to be greater than the given cut-to timestamp
        :param timestamp:
        :return:
        """
        skip_count = 0
        for stream_name in self.buffer.keys():
            if stream_name in ignores:
                continue
            if len(self.buffer[stream_name][1]) == 0:
                skip_count += 1
                continue
            if timestamp < np.min(self.buffer[stream_name][1]):
                skip_count += 1
            elif timestamp >= np.max(self.buffer[stream_name][1]):
                self.clear_stream_buffer_data(stream_name)
            else:
                cut_to_index = np.argmax([self.buffer[stream_name][1] > timestamp])
                self.buffer[stream_name][1] = self.buffer[stream_name][1][cut_to_index:]
                self.buffer[stream_name][0] = self.buffer[stream_name][0][:, cut_to_index:]
        if skip_count == len(self.buffer):
            warnings.warn('DataBuffer: nothing is cleared, given cut-to time is smaller than smallest stream timestamp')

    def __getitem__(self, key):
        return self.buffer[key]

    def get_stream(self, stream):
        return self.buffer[stream]

    def get_data(self, stream_name):
        return self.buffer[stream_name][0]

    def get_timestamps(self, stream_name):
        return self.buffer[stream_name][1]

    def keys(self):
        return self.buffer.keys()

class DataBufferSingleStream():
    def __init__(self, num_channels=None, buffer_sizes: int = None, append_zeros=False):
        self.buffer_size = buffer_sizes
        self.buffer = []
        self.append_zeros = append_zeros
        self.samples_received = 0
        self.num_channels = num_channels
        self.reset_buffer()

    def update_buffer(self, data_dict: dict):
        '''

        :param data_dict: two keys: frame and timestamp, note this is different from DataBuffer defined above
        where the keys are the lsl stream names
        :return:
        '''
        if len(self.buffer) == 0:  # init the data buffer
            self.init_buffer(data_dict['frames'].shape[0])

        # self.bu(self.buffer, data_dict)
        try:
            # pass
            self.buffer[0] = np.concatenate([self.buffer[0], data_dict['frames']], axis=-1)
        except ValueError:
            raise ChannelMismatchError(data_dict['frames'].shape[0])
        self.buffer[1] = np.concatenate([self.buffer[1], data_dict['timestamps']])

        buffer_time_points = self.buffer[0].shape[-1]
        cut_to = -np.min([buffer_time_points, self.buffer_size])
        self.buffer[0] = self.buffer[0][:, cut_to:]
        self.buffer[1] = self.buffer[1][cut_to:]
        self.samples_received += data_dict['frames'].shape[1]

    def init_buffer(self, num_channels):
        time_dim = self.buffer_size if self.append_zeros else 0
        self.buffer.append(np.zeros(shape=(num_channels, time_dim)))
        self.buffer.append(np.zeros(shape=(time_dim,)))  # data first, timestamps second
        self.samples_received = 0

    def reset_buffer(self):
        if self.num_channels is not None:
            self.init_buffer(self.num_channels)

    def has_data(self):
        return self.samples_received > 0


def flatten(l):
    return [item for sublist in l for item in sublist]


def click_on_file(filename):
    '''Open document with default application in Python.'''
    try:
        os.startfile(filename)
    except AttributeError:
        subprocess.call(['open', filename])


def check_buffer_timestamps_monotonic(data_buffer: DataBuffer):
    for stream_name, (_, timestamps) in data_buffer.buffer.items():
        if not np.all(np.diff(timestamps) >= 0):
            raise Exception("timestamps for stream {0} is not monotonic.".format(stream_name))
