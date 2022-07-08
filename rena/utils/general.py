import json
import os
import sys

import numpy as np

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
    if preset_dict['GroupChannelsInPlot'] is None or 'GroupChannelsInPlot' not in preset_dict:
        # create GroupChannelsInPlot from 0 to x
        # if channel num is greater than 100, we hide the rest
        channel_num = len(preset_dict['ChannelNames'])
        if channel_num <= DEFAULT_CHANNEL_DISPLAY_NUM:
            channels_display = [1 for channel in range(0, channel_num)]
        else:
            channels_display = [1 for channel in range(0, DEFAULT_CHANNEL_DISPLAY_NUM)]
            channels_display.extend([0 for channel in range(DEFAULT_CHANNEL_DISPLAY_NUM, channel_num)])

        preset_dict['GroupChannelsInPlot'] = {
            "Group1": {
                "group_index": 1,
                "plot_format": "time_series",
                "channels": [channel_index for channel_index in range(0, len(preset_dict['ChannelNames']))],
                "channels_display": channels_display,
                "group_display": 1,
                "group_description": ""
            }
        }
    else:
        plot_group_slice = []
        head = 0
        for x in preset_dict['GroupChannelsInPlot']:
            plot_group_slice.append((head, x))
            head = x
        if head != preset_dict['NumChannels']:
            plot_group_slice.append(
                (head, preset_dict['NumChannels']))  # append the last group
            # create GroupChannelsInPlot from 0 to x
            # preset_dict['GroupChannelsInPlot'] = [[channel_index for channel_index in range(0, len(preset_dict['ChannelNames']))]]

        if preset_dict['GroupFormat'] is None or 'GroupFormat' not in preset_dict:
            preset_dict['GroupFormat'] = ['time_series'] * (len(preset_dict['GroupChannelsInPlot']))

        preset_dict['GroupChannelsInPlot'] = dict()
        for i, group in enumerate(plot_group_slice):
            preset_dict['GroupChannelsInPlot']["Group{0}".format(i)] = \
                {
                    "group_index": list(range(*group)),
                    "plot_format": "time_series",
                    "channels": [channel_index for channel_index in range(0, len(preset_dict['ChannelNames']))]
                }

    return preset_dict


def process_preset_create_lsl_interface(preset):
    lsl_stream_name, lsl_chan_names, group_chan_in_plot = preset['StreamName'], preset['ChannelNames'], \
                                                          preset['GroupChannelsInPlot']
    try:
        interface = LSLInletInterface.LSLInletInterface(lsl_stream_name)
    except AttributeError:
        raise AssertionError('Unable to find LSL Stream with given type {0}.'.format(lsl_stream_name))
    lsl_num_chan = interface.get_num_chan()
    preset['NumChannels'] = lsl_num_chan

    # process srate
    if not preset['NominalSamplingRate']:  # try to find the nominal srate from lsl stream info if not provided
        preset['NominalSamplingRate'] = interface.get_nominal_srate()
        if not preset['NominalSamplingRate']:
            raise AssertionError(
                'Unable to load preset with name {0}, it does not have a nominal srate. RN requires all its streams to provide nominal srate for visualization purpose. You may manually define the NominalSamplingRate in presets.'.format(
                    lsl_stream_name))

    # process channel names ###########################
    if lsl_chan_names:
        if lsl_num_chan != len(lsl_chan_names):
            raise AssertionError(
                'Unable to load preset with name {0}, number of channels mismatch the number of channel names.'.format(
                    lsl_stream_name))
    else:
        preset['ChannelNames'] = ['channel_' + str(i) for i in
                                  list(range(0, preset['NumChannels']))]  # ['Unknown'] * preset['NumChannels']
    # process lsl presets ###########################
    # if group_chan_in_plot and len(group_chan_in_plot) > 0:
    #     # if np.max(preset_dict['GroupChannelsInPlot']) > preset_dict['NumChannels']:
    #     #     raise AssertionError(
    #     #         'Unable to load preset with name {0}, GroupChannelsInPlot max must be less than the number of channels.'.format(
    #     #             lsl_stream_name))
    #     preset_dict = process_LSL_plot_group(preset_dict)

    # now always create time series data format and set group channels from 0 to x if no group channels
    preset_dict = process_plot_group(preset)
    return preset_dict, interface


def process_preset_create_openBCI_interface_startsensor(devise_preset_dict):
    try:
        interface = OpenBCILSLInterface(stream_name=devise_preset_dict['StreamName'],
                                        stream_type=devise_preset_dict['StreamType'],
                                        serial_port=devise_preset_dict["SerialPort"],
                                        board_id=devise_preset_dict["Board_id"],
                                        log='store_true', )
        interface.start_sensor()
    except AssertionError as e:
        raise AssertionError(e)

    lsl_preset_dict = devise_preset_dict

    return lsl_preset_dict, interface


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