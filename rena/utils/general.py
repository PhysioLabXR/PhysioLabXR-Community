import os
import subprocess
import sys

import numpy as np

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
        interface.start_sensor()
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

    def clear_buffer(self):
        self.buffer = dict()

    def clear_stream_buffer(self, stream_name):
        self.buffer.pop(stream_name)

    def __getitem__(self, key):
        return self.buffer[key]  # TODO does this work?

    def get_data(self, stream_name):
        return self.buffer[stream_name][0]

    def get_timestamps(self, stream_name):
        return self.buffer[stream_name][1]




class DataBufferSingleStream():
    def __init__(self, num_channels=None, buffer_sizes: int = None, append_zeros=False):
        self.buffer_size = buffer_sizes
        self.buffer = []
        self.append_zeros = append_zeros
        if num_channels is not None:
            self.init_buffer(num_channels)

    def update_buffer(self, data_dict: dict):
        '''

        :param data_dict: two keys: frame and timestamp, note this is different from DataBuffer defined above
        where the keys are the lsl stream names
        :return:
        '''
        if len(self.buffer) == 0:  # init the data buffer
            self.init_buffer(data_dict['frames'].shape[0])
        self.buffer[0] = np.concatenate([self.buffer[0], data_dict['frames']], axis=-1)
        self.buffer[1] = np.concatenate([self.buffer[1], data_dict['timestamps']])

        buffer_time_points = self.buffer[0].shape[-1]
        cut_to = -np.min([buffer_time_points, self.buffer_size])
        self.buffer[0] = self.buffer[0][:, cut_to:]
        self.buffer[1] = self.buffer[1][cut_to:]

    def init_buffer(self, num_channels):
        time_dim = self.buffer_size if self.append_zeros else 0
        self.buffer.append(np.empty(shape=(num_channels, time_dim)))
        self.buffer.append(np.empty(shape=(time_dim,)))  # data first, timestamps second

    def clear_buffer(self):
        self.buffer = []

    def has_data(self):
        return len(self.buffer) > 0


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
