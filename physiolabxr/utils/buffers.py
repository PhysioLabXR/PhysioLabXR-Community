import os
import subprocess
import sys
import warnings

import numpy as np

from physiolabxr.exceptions.exceptions import ChannelMismatchError
from physiolabxr.interfaces.OpenBCIDeviceInterface import OpenBCIDeviceInterface
from physiolabxr.interfaces.MmWaveSensorLSLInterface import MmWaveSensorLSLInterface


def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def get_fps(queue):
    try:
        return len(queue) / (queue[-1] - queue[0])
    except ZeroDivisionError:
        return 0


def process_preset_create_openBCI_interface_startsensor(device_name, serial_port, board_id):
    try:
        interface = OpenBCIDeviceInterface(stream_name=device_name,
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
        self.stream_name_buffer_sizes = stream_buffer_sizes if stream_buffer_sizes else dict()

    def update_buffer(self, data_dict: dict):
        if len(data_dict) > 0:
            self._update_buffer(data_dict['stream_name'], data_dict['frames'], data_dict['timestamps'])

    def update_buffers(self, data_buffer):
        for stream_name, (frames, timestamps) in data_buffer.items():
            self._update_buffer(stream_name, frames, timestamps)

    def update_buffer_size(self, stream_name, size):
        self.stream_name_buffer_sizes[stream_name] = size

    def _update_buffer(self, stream_name, frames, timestamps):
        """
        frames: channels x time
        """
        if stream_name not in self.buffer.keys():
            self.buffer[stream_name] = [np.empty(shape=(*frames.shape[:-1], 0), dtype=frames.dtype),
                                      np.empty(shape=(0,))]  # data first, timestamps second

        # check if the new frame's number of channels matches that of the buffered data
        if frames.shape[0] != self.buffer[stream_name][0].shape[0]:
            # reset the buffer with the new frame channels
            self.buffer[stream_name] = [np.empty(shape=(frames.shape[0], 0), dtype=frames.dtype),
                                        np.empty(shape=(0,))]

        buffered_data = self.buffer[stream_name][0]
        buffered_timestamps = self.buffer[stream_name][1]

        self.buffer[stream_name][0] = np.concatenate([buffered_data, frames], axis=-1)
        self.buffer[stream_name][1] = np.concatenate([buffered_timestamps, timestamps])

        if stream_name in self.stream_name_buffer_sizes.keys():  # keep only the latest data according to the buffer size
            buffer_time_points = self.buffer[stream_name][0].shape[-1]
            cut_to = int(-np.min([buffer_time_points, self.stream_name_buffer_sizes[stream_name]]))
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

    def clear_stream_up_to(self, stream_name, timestamp):
        """
        The resulting timestamp is guaranteed to be greater than the given cut-to timestamp
        :param timestamp:
        :return:
        """
        if stream_name not in self.buffer.keys():
            return
        if len(self.buffer[stream_name][1]) == 0:
            return
        if timestamp < np.min(self.buffer[stream_name][1]):
            return
        elif timestamp >= np.max(self.buffer[stream_name][1]):
            self.clear_stream_buffer_data(stream_name)
        else:
            cut_to_index = np.argmax([self.buffer[stream_name][1] > timestamp])
            self.buffer[stream_name][1] = self.buffer[stream_name][1][cut_to_index:]
            self.buffer[stream_name][0] = self.buffer[stream_name][0][:, cut_to_index:]

    def clear_stream_up_to_index(self, stream_name, cut_to_index):
        if stream_name not in self.buffer.keys():
            return
        if len(self.buffer[stream_name][1]) == 0:
            return
        self.buffer[stream_name][1] = self.buffer[stream_name][1][cut_to_index:]
        self.buffer[stream_name][0] = self.buffer[stream_name][0][:, cut_to_index:]

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

    def get_stream_in_time_range(self, stream_name, start_time, end_time):

        # start time must be smaller than end time
        if start_time > end_time:
            raise ValueError('start_time must be smaller than end_time')

        stream_data = self.buffer[stream_name][0]
        stream_timestamps = self.buffer[stream_name][1]

        start_index = np.searchsorted(stream_timestamps, [start_time], side='left')[0]
        end_index = np.searchsorted(stream_timestamps, [end_time], side='right')[0]

        return [stream_data[:, start_index:end_index], stream_timestamps[start_index:end_index]]

    def get_all_streams_in_time_range(self, start_time, end_time):
        """
        Returns all streams in the buffer in the given time range
        :param start_time:
        :param end_time:
        :return:
        """
        # start time must be smaller than end time
        if start_time > end_time:
            raise ValueError('start_time must be smaller than end_time')

        all_streams = dict()
        for stream_name in self.buffer.keys():
            # use get_stream_in_time_range to get the data
            all_streams[stream_name] = self.get_stream_in_time_range(stream_name, start_time, end_time)

        data_buffer = DataBuffer()
        data_buffer.buffer = all_streams
        return data_buffer

    def get_stream_in_index_range(self, stream_name, start_index, end_index):

        if start_index < 0 or end_index < 0:
            raise ValueError('start_index and end_index must be positive')

        if start_index > end_index:
            raise ValueError('start_index must be smaller than end_index')

        stream_data = self.buffer[stream_name][0]
        stream_timestamps = self.buffer[stream_name][1]

        return [stream_data[:, start_index:end_index], stream_timestamps[start_index:end_index]]



    def get_timestamps(self, stream_name):
        return self.buffer[stream_name][1]

    def keys(self):
        return self.buffer.keys()

class DataBufferSingleStream():
    def __init__(self, num_channels: int, buffer_sizes: int, append_zeros=False):
        self.buffer_size = buffer_sizes
        self.buffer = []
        self.append_zeros = append_zeros
        self.samples_received = 0
        self.num_channels = num_channels
        self.reset_buffer()

    def update_buffer(self, data_dict: dict):
        '''

        Args
        data_dict: two keys: 'frames' and 'timestamp', note this is different from DataBuffer defined above
            where the keys are the lsl stream names.
            Shape of the 'frames' should be (num_channels, num_samples)
        :return:
        '''
        frames = data_dict['frames']
        timestamps = data_dict['timestamps']
        if len(self.buffer) == 0:  # init the data buffer
            self.init_buffer(frames.shape[0])
        if frames.shape[0] != self.num_channels:
            print(f"DataBufferSingleStream: Channel mismatch. Expected {self.num_channels}, got {frames.shape[0]}. Maybe you need to reshape the data.")
            raise ChannelMismatchError(frames.shape[0])

        self.buffer[0] = np.roll(self.buffer[0], -frames.shape[-1], axis=-1)
        self.buffer[1] = np.roll(self.buffer[1], -frames.shape[-1])

        # self.buffer[0] = np.concatenate([self.buffer[0], data_dict['frames']], axis=-1)
        # self.buffer[1] = np.concatenate([self.buffer[1], data_dict['timestamps']])
        self.buffer[0][:, -frames.shape[-1]:] = frames[:, -self.buffer_size:]
        self.buffer[1][-frames.shape[-1]:] = timestamps[-self.buffer_size:]

        # buffer_time_points = self.buffer[0].shape[-1]
        # cut_to = -np.min([buffer_time_points, self.buffer_size])
        # self.buffer[0] = self.buffer[0][:, cut_to:]
        # self.buffer[1] = self.buffer[1][cut_to:]

        self.samples_received += frames.shape[1]

    def init_buffer(self, num_channels):
        self.buffer.append(np.zeros(shape=(num_channels, self.buffer_size)))
        self.buffer.append(np.zeros(shape=(self.buffer_size,)))  # data first, timestamps second
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
