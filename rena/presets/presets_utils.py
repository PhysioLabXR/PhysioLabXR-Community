from typing import Union, List

from exceptions.exceptions import InvalidPresetErrorChannelNameOrNumChannel
from rena import config
from rena.presets.GroupEntry import GroupEntry, PlotFormat
from rena.presets.Presets import Presets, PresetType, preprocess_stream_preset
from rena.utils.data_utils import convert_dict_keys_to_snake_case


def get_preset_category(preset_name):
    preset = Presets()
    if preset_name in preset.experiment_presets.keys():
        return PresetType.EXPERIMENT
    else:
        return preset[preset_name].preset_type


def get_all_preset_names():
    return Presets().keys()


def get_stream_preset_names():
    return Presets().stream_presets.keys()


def get_experiment_preset_names():
    return Presets().experiment_presets.keys()


def get_experiment_preset_streams(exp_name):
    return Presets().experiment_presets[exp_name]


def get_stream_preset_info(stream_name, key):
    return Presets().stream_presets[stream_name].__getattribute__(key)


def get_stream_preset_custom_info(stream_name) -> dict:
    return Presets().stream_presets[stream_name].device_info


def set_stream_preset_info(stream_name, key, value):
    setattr(Presets().stream_presets[stream_name], key, value)


def check_preset_exists(stream_name):
    return stream_name in Presets().stream_presets.keys()

def get_stream_group_info(stream_name) -> dict[str, GroupEntry]:
    return Presets().stream_presets[stream_name].group_info

def get_stream_a_group_info(stream_name, group_name) -> GroupEntry:
    return Presets().stream_presets[stream_name].group_info[group_name]

def get_group_image_config(stream_name, group_name):
    return Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config

def get_is_group_shown(stream_name, group_name) -> List[bool]:
    return Presets().stream_presets[stream_name].group_info[group_name].is_channels_shown

def is_group_image_only(stream_name, group_name):
    return Presets().stream_presets[stream_name].group_info[group_name].is_image_only

def set_stream_a_group_selected_plot_format(stream_name, group_name, plot_format: Union[str, int, PlotFormat]) -> PlotFormat:
    if isinstance(plot_format, str):
        plot_format = PlotFormat[plot_format.upper()]
    elif isinstance(plot_format, int):
        plot_format = PlotFormat(plot_format)

    Presets().stream_presets[stream_name].group_info[group_name].selected_plot_format = plot_format
    return plot_format

def set_stream_a_group_selected_img_config(stream_name, group_name, height, width, scaling):
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.height = height
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.width = width
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.scaling = scaling

def set_bar_chart_max_min_range(stream_name, group_name, max_range, min_range):
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.barchart_config.y_max = max_range
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.barchart_config.y_min = min_range

def get_bar_chart_max_min_range(stream_name, group_name) -> tuple[float, float]:
    return Presets().stream_presets[stream_name].group_info[group_name].plot_configs.barchart_config.y_max, \
           Presets().stream_presets[stream_name].group_info[group_name].plot_configs.barchart_config.y_min

def set_group_image_format(stream_name, group_name, image_format):
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.image_format = image_format

def set_group_image_channel_format(stream_name, group_name, channel_format):
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.channel_format = channel_format

def set_group_image_valid(stream_name, group_name, is_valid):
    Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.is_valid = is_valid

def get_group_image_valid(stream_name, group_name):
    return Presets().stream_presets[stream_name].group_info[group_name].plot_configs.image_config.is_valid
def get_selected_plot_format(stream_name, group_name) -> PlotFormat:
    return Presets().stream_presets[stream_name].group_info[group_name].selected_plot_format
def get_selected_plot_format_index(stream_name, group_name) -> int:
    return Presets().stream_presets[stream_name].group_info[group_name].selected_plot_format.value

def get_group_channel_indices(stream_name, group_name) -> list[int]:
    return Presets().stream_presets[stream_name].group_info[group_name].channel_indices

def save_preset(is_async=True):
    Presets().save(is_async=is_async)


def create_default_preset(stream_name, data_type, port, preset_type_str, num_channels, nominal_sample_rate: int=None):
    if check_preset_exists(stream_name):
        raise ValueError(f'Stream preset with stream name {stream_name} already exists.')
    preset_dict = {'StreamName': stream_name,
                   'ChannelNames': ['channel{0}'.format(i) for i in range(num_channels)],
                   'DataType': data_type,
                   'PortNumber': port}
    if nominal_sample_rate:
        preset_dict['NominalSamplingRate'] = nominal_sample_rate

    stream_preset_dict = preprocess_stream_preset(preset_dict, preset_type_str)
    Presets().add_stream_preset(stream_preset_dict)
    return preset_dict

def pop_group_from_stream_preset(stream_name, group_name) -> GroupEntry:
    return Presets().stream_presets[stream_name].group_info.pop(group_name)

def add_group_dict_to_stream(stream_name, new_group_info):
    if set(Presets().stream_presets.keys()) & set(new_group_info.keys()):
        raise ValueError('Group name already exists in stream preset')
    Presets().stream_presets[stream_name].group_info.update(new_group_info)

def set_group_channel_indices(stream_name, group_name, channel_indices):
    Presets().stream_presets[stream_name].group_info[group_name].channel_indices = channel_indices

def set_group_channel_is_shown(stream_name, group_name, is_shown):
    Presets().stream_presets[stream_name].group_info[group_name].is_channels_shown = is_shown

def change_stream_group_order(stream_name, group_order):
    new_group_info = dict[str: GroupEntry]
    for group_name in group_order:
        new_group_info[group_name] = Presets.stream_presets[stream_name].group_info.pop(group_name)
    Presets.stream_presets[stream_name].group_info = new_group_info

def change_stream_group_name(stream_name, new_group_name, old_group_name):
    assert new_group_name not in Presets.stream_presets[stream_name].group_info.keys(), f'New group name {new_group_name} already exists for stream {stream_name}'
    Presets.stream_presets[stream_name].group_info[new_group_name] = Presets.stream_presets[stream_name].group_info.pop(old_group_name)


def validate_preset_json_preset(preset_dict):
    if 'GroupInfo' in preset_dict.keys():
        try:
            assert 'ChannelNames' in preset_dict.keys() or 'NumChannels' in preset_dict.keys()
        except AssertionError:
            raise ValueError('Preset with stream name {0} has GroupChnanelsInPlot field. In this case, this preset must also have either ChannelNames field or NumChannels field'
                             '. This is likely a problem with the default presets or bug in preset creation'.format(preset_dict['StreamName']))
    else:
        preset_dict['GroupInfo'] = None
    if 'ChannelNames' in preset_dict.keys() and 'NumChannels' not in preset_dict.keys():
        preset_dict['NumChannels'] = len(preset_dict['ChannelNames'])
    elif 'NumChannels' in preset_dict.keys() and 'ChannelNames' not in preset_dict.keys():
        preset_dict['ChannelNames'] = ['Channel{0}'.format(x) for x in list(range(int(preset_dict['NumChannels'])))]
    else:
        raise InvalidPresetErrorChannelNameOrNumChannel(preset_dict['StreamName'])
    # if 'GroupInfo' not in preset_dict.keys():
    #     preset_dict['GroupInfo'] = None
    #     preset_dict['GroupFormat'] = None
    # if 'GroupFormat' not in preset_dict.keys():
    #     preset_dict['GroupFormat'] = None
    # if 'NominalSamplingRate' not in preset_dict.keys():
    #     preset_dict['NominalSamplingRate'] = 1
    # if 'DisplayDuration' not in preset_dict.keys():
    #     preset_dict['DisplayDuration'] = config.settings.value('viz_display_duration')

    # if 'NetworkingInterface' not in preset_dict.keys():
    #     preset_dict['NetworkingInterface'] = 'LSL'  # default is LSL
    # if 'PortNumber' not in preset_dict.keys():
    #     preset_dict['PortNumber'] = None
    # if 'DataType' not in preset_dict.keys():
    #     preset_dict['DataType'] = 'float32'
    preset_dict = convert_dict_keys_to_snake_case(preset_dict)
    return preset_dict


def pop_stream_preset_from_settings(stream_name):
    return Presets().stream_presets.pop(stream_name)