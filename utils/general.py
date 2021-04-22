import json
import os

import brainflow
import numpy as np

from interfaces.LSLInletInterface import LSLInletInterface
from interfaces.OpenBCILSLInterface import OpenBCILSLInterface


def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def load_all_LSL_presets(lsl_preset_roots='Presets/LSLPresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = load_LSL_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_all_Device_presets(lsl_preset_roots='Presets/DevicePresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = load_LSL_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_LSL_preset(preset_dict):
    if 'ChannelNames' not in preset_dict.keys():
        preset_dict['ChannelNames'] = None
    if 'GroupChannelsInPlot' not in preset_dict.keys():
        preset_dict['GroupChannelsInPlot'] = None
        preset_dict['PlotGroupSlices'] = None
    if 'NominalSamplingRate' not in preset_dict.keys():
        preset_dict['NominalSamplingRate'] = None
    return preset_dict


def create_LSL_preset(stream_name, channel_names=None, plot_group_slices=None):
    preset_dict = {'StreamName': stream_name, 'ChannelNames': channel_names, 'PlotGroupSlices': plot_group_slices}
    preset_dict = load_LSL_preset(preset_dict)
    return preset_dict


def process_LSL_plot_group(preset_dict):
    preset_dict["PlotGroupSlices"] = []
    head = 0
    for x in preset_dict['GroupChannelsInPlot']:
        preset_dict["PlotGroupSlices"].append((head, x))
        head = x
    if head != preset_dict['NumChannels']:
        preset_dict["PlotGroupSlices"].append(
            (head, preset_dict['NumChannels']))  # append the last group
    return preset_dict


def process_preset_create_lsl_interface(preset_dict):
    lsl_stream_name, lsl_chan_names, group_chan_in_plot = preset_dict['StreamName'], preset_dict['ChannelNames'], \
                                                          preset_dict['GroupChannelsInPlot']
    try:
        interface = LSLInletInterface(lsl_stream_name)
    except AttributeError:
        raise AssertionError('Unable to find LSL Stream with given type {0}.'.format(lsl_stream_name))
    lsl_num_chan = interface.get_num_chan()
    preset_dict['NumChannels'] = lsl_num_chan

    # process srate
    if not preset_dict['NominalSamplingRate']:  # try to find the nominal srate from lsl stream info if not provided
        preset_dict['NominalSamplingRate'] = interface.get_nominal_srate()
        if not preset_dict['NominalSamplingRate']:
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
        preset_dict['ChannelNames'] = ['Unknown'] * preset_dict['NumChannels']
    # process lsl presets ###########################
    if group_chan_in_plot and len(group_chan_in_plot) > 0:
        if np.max(preset_dict['GroupChannelsInPlot']) > preset_dict['NumChannels']:
            raise AssertionError(
                'Unable to load preset with name {0}, GroupChannelsInPlot max must be less than the number of channels.'.format(
                    lsl_stream_name))
        preset_dict = process_LSL_plot_group(preset_dict)
    return preset_dict, interface


def process_preset_create_openBCI_interface(devise_preset_dict):
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
