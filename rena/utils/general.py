import os
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
    def __init__(self, data_type_buffer_sizes:dict=None):
        self.buffer = dict()
        self.data_type_buffer_sizes = data_type_buffer_sizes if data_type_buffer_sizes else dict()

    def update_buffers(self, data_dict: dict):
        lsl_data_type = data_dict['lsl_data_type']  # get the type of the newly-come data

        if lsl_data_type not in self.buffer.keys():
            self.buffer[lsl_data_type] = [np.empty(shape=(data_dict['frames'].shape[0], 0)),
                                          np.empty(shape=(0,))]  # data first, timestamps second
        buffered_data = self.buffer[data_dict['lsl_data_type']][0]
        buffered_timestamps = self.buffer[data_dict['lsl_data_type']][1]

        self.buffer[lsl_data_type][0] = np.concatenate([buffered_data, data_dict['frames']], axis=-1)
        self.buffer[lsl_data_type][1] = np.concatenate([buffered_timestamps, data_dict['timestamps']])

        if lsl_data_type in self.data_type_buffer_sizes.keys(): # keep only the latest data according to the buffer size
            buffer_time_points = self.buffer[lsl_data_type][0].shape[-1]
            cut_to = -np.min([buffer_time_points, self.data_type_buffer_sizes[lsl_data_type]])
            self.buffer[lsl_data_type][0] = self.buffer[lsl_data_type][0][:, cut_to:]
            self.buffer[lsl_data_type][1] = self.buffer[lsl_data_type][1][cut_to:]

    def clear_buffer(self):
        self.buffer = dict()

def flatten(l):
    return [item for sublist in l for item in sublist]
