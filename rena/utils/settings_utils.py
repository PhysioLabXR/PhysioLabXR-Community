import json
import os

from exceptions.exceptions import InvalidPresetError
from rena import config
from rena.config import DEFAULT_CHANNEL_DISPLAY_NUM


def get_presets_by_category(setting_category):
    assert setting_category == 'streampresets' or setting_category == 'experimentpresets'
    group = 'presets/{0}'.format(setting_category)
    config.settings.beginGroup(group)
    presets = list(config.settings.childGroups())
    config.settings.endGroup()
    return presets

def get_all_preset_names():
    config.settings.beginGroup('presets/streampresets')
    stream_preset_names = list(config.settings.childGroups())
    config.settings.endGroup()
    config.settings.beginGroup('presets/experimentpresets')
    experiment_preset_names = list(config.settings.childGroups())
    config.settings.endGroup()
    return stream_preset_names + experiment_preset_names

def get_stream_preset_names():
    config.settings.beginGroup('presets/streampresets')
    stream_preset_names = list(config.settings.childGroups())
    config.settings.endGroup()
    return stream_preset_names

def get_stream_preset_info(stream_name, key):
    return config.settings.value('presets/streampresets/{0}/{1}'.format(stream_name, key))

def collect_stream_group_info(stream_name):
    rtn = dict()
    config.settings.beginGroup('presets/streampresets/{0}/GroupInfo'.format(stream_name))
    for group_name in config.settings.childGroups():
        config.settings.beginGroup(group_name)
        rtn[group_name] = dict([(k, config.settings.value(k)) for k in config.settings.childKeys()])
        rtn[group_name]['is_group_shown'] = bool(int(rtn[group_name]['is_group_shown']))
        rtn[group_name]['is_channels_shown'] = [bool(int(x)) for x in rtn[group_name]['is_channels_shown']]
        config.settings.endGroup()
    config.settings.endGroup()
    return rtn

def get_childKeys_for_group(group):
    config.settings.beginGroup(group)
    rtn = config.settings.childKeys()
    config.settings.endGroup()
    return rtn

def get_childGroups_for_group(group):
    config.settings.beginGroup(group)
    rtn = config.settings.childGroups()
    config.settings.endGroup()
    return rtn

def get_all_lsl_device_preset_names():
    return get_childGroups_for_group('presets/streampresets')


def export_preset_to_settings(preset, setting_category):
    assert setting_category == 'streampresets' or setting_category == 'experimentpresets'
    if setting_category == 'experimentpresets':
        config.settings.setValue('presets/experimentpresets/{0}/PresetStreamNames'.format(preset[0]), preset[1])
    else:
        config.settings.beginGroup('presets/{0}'.format(setting_category))

        for preset_key, value in preset.items():
            if preset_key != 'GroupInfo':
                config.settings.setValue('{0}/{1}'.format(preset['StreamName'], preset_key), value)

        for group_name, group_info_dict in preset['GroupInfo'].items():
            for group_info_key, group_info_value in group_info_dict.items():
                config.settings.setValue('{0}/GroupInfo/GroupName{1}/{2}'.format(preset['StreamName'], group_info_dict['group_index'], group_info_key), group_info_value)
        config.settings.endGroup()


def load_all_lslStream_presets(lsl_preset_roots='../Presets/LSLPresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = add_keys_to_preset(loaded_preset_dict)
        stream_name = preset_dict['StreamName']
        presets[stream_name] = preset_dict
    return presets


def load_all_Device_presets(device_preset_roots='../Presets/DevicePresets'):
    preset_file_names = os.listdir(device_preset_roots)
    preset_file_paths = [os.path.join(device_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))
        preset_dict = add_keys_to_preset(loaded_preset_dict)
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


def add_keys_to_preset(preset_dict):
    if 'GroupInfo' in preset_dict.keys():
        try:
            assert 'ChannelNames' in preset_dict.keys() or 'NumChannels' in preset_dict.keys()
        except AssertionError:
            raise ValueError('Preset with stream name {0} has GroupChnanlesInPlot field. In this case, this preset must also have either ChannelNmaes field or NumChannels field'
                             '. This is likely a problem with the default presets or bug in preset creation'.format(preset_dict['StreamName']))
    if 'ChannelNames' in preset_dict.keys() and 'NumChannels' not in preset_dict.keys():
        preset_dict['NumChannels'] = len(preset_dict['ChannelNames'])
    elif 'NumChannels' in preset_dict.keys() and 'ChannelNames' not in preset_dict.keys():
        preset_dict['ChannelNames'] = ['Channel'] + list(range(int(preset_dict['NumChannels'])))
    else:
        raise InvalidPresetError(preset_dict['stream_name'])
    if 'GroupInfo' not in preset_dict.keys():
        preset_dict['GroupInfo'] = None
        preset_dict['GroupFormat'] = None
    if 'GroupFormat' not in preset_dict.keys():
        preset_dict['GroupFormat'] = None
    if 'NominalSamplingRate' not in preset_dict.keys():
        preset_dict['NominalSamplingRate'] = 1
    return preset_dict


def create_default_preset(stream_name):
    preset_dict = {'StreamName': stream_name,
                   'ChannelNames': ['channel1']}
    preset_dict = add_keys_to_preset(preset_dict)
    preset_dict = process_plot_group(preset_dict)
    export_preset_to_settings(preset_dict, setting_category='streampresets')
    return preset_dict


def process_plot_group(preset_dict):
    channel_num = preset_dict['NumChannels']
    if preset_dict['GroupInfo'] is None or 'GroupInfo' not in preset_dict:
        # create groupinfo from 0 to x
        # if channel num is greater than 100, we hide the rest
        if channel_num <= DEFAULT_CHANNEL_DISPLAY_NUM:
            is_channels_shown = [1 for c in range(0, channel_num)]
        else:
            is_channels_shown = [1 for c in range(0, DEFAULT_CHANNEL_DISPLAY_NUM)]
            is_channels_shown.extend([0 for c in range(DEFAULT_CHANNEL_DISPLAY_NUM, channel_num)])

        preset_dict['GroupInfo'] = {
            "Group1": {
                "group_index": 1,
                "plot_format": "time_series",
                "channel_indices": [channel_index for channel_index in range(0, channel_num)],
                "is_channels_shown": is_channels_shown,
                "is_group_shown": 1,
                "group_description": ""
            }
        }
    else:
        plot_group_slice = []
        head = 0
        for x in preset_dict['GroupInfo']:
            plot_group_slice.append((head, x))
            head = x
        if head != channel_num:
            plot_group_slice.append(
                (head, channel_num))  # append the last group
            # create GroupInfo from 0 to x
            # preset_dict['GroupInfo'] = [[channel_index for channel_index in range(0, len(preset_dict['ChannelNames']))]]

        if preset_dict['GroupFormat'] is None or 'GroupFormat' not in preset_dict:
            preset_dict['GroupFormat'] = ['time_series'] * (len(preset_dict['GroupInfo']))

        preset_dict['GroupInfo'] = dict()
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

            preset_dict['GroupInfo']["Group{0}".format(i)] = \
                {
                    "group_index": i,
                    "plot_format": "time_series",
                    "channel_indices": channel_indices,
                    "is_channels_shown": is_channels_shown,
                    "is_group_shown": 1,
                    "group_description": ""
                }

    return preset_dict