from typing import Union, List

from rena.presets.GroupEntry import GroupEntry, PlotFormat
from rena.presets.Presets import Presets, PresetType, preprocess_stream_preset


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
    return Presets().stream_presets[stream_name].group_info[group_name].is_image_only()


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


def get_group_image_valid(stream_name, group_name):
    return Presets().stream_presets[stream_name].group_info[group_name].is_image_valid()


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

def add_group_entry_to_stream(stream_name, group_entry):
    Presets().stream_presets[stream_name].add_group_entry(group_entry)

def set_group_channel_indices(stream_name, group_name, channel_indices):
    Presets().stream_presets[stream_name].group_info[group_name].channel_indices = channel_indices

def set_group_channel_is_shown(stream_name, group_name, is_shown):
    Presets().stream_presets[stream_name].group_info[group_name].is_channels_shown = is_shown

def change_stream_group_order(stream_name, group_order):
    new_group_info = dict()
    for group_name in group_order:
        new_group_info[group_name] = Presets().stream_presets[stream_name].group_info.pop(group_name)
    Presets().stream_presets[stream_name].group_info = new_group_info

def change_stream_group_name(stream_name, new_group_name, old_group_name):
    try:
        assert new_group_name not in Presets().stream_presets[stream_name].group_info.keys()
    except AssertionError as e:
        raise ValueError(f'New group name {new_group_name} already exists for stream {stream_name}')
    Presets().stream_presets[stream_name].group_info[new_group_name] = Presets().stream_presets[stream_name].group_info.pop(old_group_name)


def pop_stream_preset_from_settings(stream_name):
    return Presets().stream_presets.pop(stream_name)