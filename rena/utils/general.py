import json
import os
import sys

import cv2
import numpy as np

from rena import config
from rena.config import DEFAULT_CHANNEL_DISPLAY_NUM
from rena.interfaces import LSLInletInterface
from rena.interfaces.OpenBCILSLInterface import OpenBCILSLInterface
from rena.interfaces.MmWaveSensorLSLInterface import MmWaveSensorLSLInterface


def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def load_all_lslStream_presets(lsl_preset_roots='../Presets/LSLPresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = load_LSL_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_all_Device_presets(device_preset_roots='../Presets/DevicePresets'):
    preset_file_names = os.listdir(device_preset_roots)
    preset_file_paths = [os.path.join(device_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = load_LSL_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_all_experiment_presets(exp_preset_roots='../Presets/ExperimentPresets'):
    preset_file_names = os.listdir(exp_preset_roots)
    preset_file_paths = [os.path.join(exp_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        presets[loaded_preset_dict['ExperimentName']] = loaded_preset_dict['PresetStreamNames']
    return presets


def load_LSL_preset(preset_dict):
    if 'GroupChannelsInPlot' in preset_dict.keys():
        try:
            assert 'ChannelNames' in preset_dict.keys() or 'NumChannels' in preset_dict.keys()
        except AssertionError:
            raise ValueError('Preset with stream name {0} has GroupChnanlesInPlot field. In this case, this preset must also have either ChannelNmaes field or NumChannels field'
                             '. This is likely a problem with the default presets or bug in preset creation'.format(preset_dict['StreamName']))
    if 'ChannelNames' in preset_dict.keys() and 'NumChannels' not in preset_dict.keys():
        preset_dict['NumChannels'] = len(preset_dict['ChannelNames'])
    if 'ChannelNames' not in preset_dict.keys():
        preset_dict['ChannelNames'] = None
    if 'GroupChannelsInPlot' not in preset_dict.keys():
        preset_dict['GroupChannelsInPlot'] = None
        preset_dict['GroupFormat'] = None
    if 'GroupFormat' not in preset_dict.keys():
        preset_dict['GroupFormat'] = None
    if 'NominalSamplingRate' not in preset_dict.keys():
        preset_dict['NominalSamplingRate'] = None
    return preset_dict


def create_LSL_preset(stream_name, channel_names=None, plot_group_slices=None, plot_group_formats=None):
    preset_dict = {'StreamName': stream_name,
                   'ChannelNames': channel_names,
                   'GroupChannelsInPlot': plot_group_slices,
                   'GroupFormat': plot_group_formats}
    preset_dict = load_LSL_preset(preset_dict)
    return preset_dict


def process_plot_group(preset_dict):
    channel_num = preset_dict['NumChannels']
    if preset_dict['GroupChannelsInPlot'] is None or 'GroupChannelsInPlot' not in preset_dict:
        # create GroupChannelsInPlot from 0 to x
        # if channel num is greater than 100, we hide the rest
        if channel_num <= DEFAULT_CHANNEL_DISPLAY_NUM:
            is_channels_shown = [1 for c in range(0, channel_num)]
        else:
            is_channels_shown = [1 for c in range(0, DEFAULT_CHANNEL_DISPLAY_NUM)]
            is_channels_shown.extend([0 for c in range(DEFAULT_CHANNEL_DISPLAY_NUM, channel_num)])

        preset_dict['GroupChannelsInPlot'] = {
            "Group1": {
                "group_index": 1,
                "plot_format": "time_series",
                "channels": [channel_index for channel_index in range(0, channel_num)],
                "is_channels_shown": is_channels_shown,
                "is_group_shown": 1,
                "group_description": ""
            }
        }
    else:
        plot_group_slice = []
        head = 0
        for x in preset_dict['GroupChannelsInPlot']:
            plot_group_slice.append((head, x))
            head = x
        if head != channel_num:
            plot_group_slice.append(
                (head, channel_num))  # append the last group
            # create GroupChannelsInPlot from 0 to x
            # preset_dict['GroupChannelsInPlot'] = [[channel_index for channel_index in range(0, len(preset_dict['ChannelNames']))]]

        if preset_dict['GroupFormat'] is None or 'GroupFormat' not in preset_dict:
            preset_dict['GroupFormat'] = ['time_series'] * (len(preset_dict['GroupChannelsInPlot']))

        preset_dict['GroupChannelsInPlot'] = dict()
        num_shown_channel = 0
        for i, group in enumerate(plot_group_slice):
            channel_indices = list(range(*group))
            num_available_ch_shown = DEFAULT_CHANNEL_DISPLAY_NUM - num_shown_channel
            if num_available_ch_shown <= 0:
                is_channels_shown = [0 for c in range(len(channel_indices))]
            else:
                is_channels_shown = [1 for c in range(min(len(channel_indices), DEFAULT_CHANNEL_DISPLAY_NUM))]
                is_channels_shown += [0] * (len(channel_indices) - len(is_channels_shown))  # won't pad if len(channel_indices) - len(is_channels_shown) is negative
                num_shown_channel += min(len(channel_indices), DEFAULT_CHANNEL_DISPLAY_NUM)

            preset_dict['GroupChannelsInPlot']["Group{0}".format(i)] = \
                {
                    "group_index": i,
                    "plot_format": "time_series",
                    "channels": channel_indices,
                    "is_channels_shown": is_channels_shown,
                    "is_group_shown": 1,
                    "group_description": ""
                }

    return preset_dict


def create_lsl_interface(lsl_name, channel_names):
    try:
        interface = LSLInletInterface.LSLInletInterface(lsl_name)
    except AttributeError:
        raise AssertionError('Unable to find LSL Stream in LAN.')
    lsl_num_chan = interface.get_num_chan()

    try:
        assert lsl_num_chan == len(channel_names)
    except AssertionError:
        raise ValueError('The preset has {0} channel names, but the \n stream in LAN has {1} channels'.format(len(channel_names), lsl_num_chan))

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


def process_preset_create_TImmWave_interface_startsensor(device_preset_dict):
    # create interface
    num_range_bin = device_preset_dict['NumRangeBin']
    interface = MmWaveSensorLSLInterface(num_range_bin=num_range_bin)
    # connect Uport Dport

    try:
        Dport = device_preset_dict['Dport(Standard)']
        Uport = device_preset_dict['Uport(Enhanced)']

        interface.connect(uport_name=Uport, dport_name=Dport)
    except AssertionError as e:
        raise AssertionError(e)

    # send config
    try:
        config_path = device_preset_dict['ConfigPath']
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

    return device_preset_dict, interface


# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_working_camera_id():
    """
    deprecated, not in use. Use the more optimized version as in general.get_working_camera_ports()
    :return:
    """
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def get_working_camera_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Video device port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Video device port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Video device port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
        camera.release()
    return available_ports, working_ports, non_working_ports

