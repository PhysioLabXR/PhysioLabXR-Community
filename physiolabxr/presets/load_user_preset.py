from physiolabxr.exceptions.exceptions import InvalidPresetErrorChannelNameOrNumChannel
from physiolabxr.configs import config
from physiolabxr.configs.config import DEFAULT_CHANNEL_DISPLAY_NUM, default_group_name
from physiolabxr.presets.GroupEntry import GroupEntry
from physiolabxr.utils.data_utils import convert_dict_keys_to_snake_case



def create_default_group_info(channel_num: int, group_name: str = config.default_group_name,
                              channel_indices=None, is_channels_shown=None):
    """
    create default group info from channel num
    @param channel_num:
    @param group_name: default is the default group name defined in config.py. This is used when calling process_plot_group_json_preset.
    This is also used in StreamWidget to create the default group info.
    @return:
    """
    group_entry = create_default_group_entry(channel_num, group_name, channel_indices, is_channels_shown)
    return {group_name: group_entry}

def create_default_group_entry(channel_num: int, group_name: str = config.default_group_name,
                               channel_indices=None, is_channels_shown=None):
    channel_indices = [channel_index for channel_index in range(0, channel_num)] if channel_indices is None else channel_indices
    is_channels_shown = [True] * channel_num if is_channels_shown is None else is_channels_shown
    group_entry = GroupEntry(group_name=group_name, channel_indices=channel_indices, is_channels_shown=is_channels_shown)
    return group_entry


def process_plot_group_json_preset(preset_dict):
    """
    create group info from the json format
    Note on the json format:
    the group info is a list of integers, where each integer is the indices at which a new group starts

    Example:
        1. for a stream that has eight channels, a group info defined as follows
        [2, 4, 6]
        will create four groups, where the first group has two channels, the second group has two channels, the third group has two channels, and the fourth group has two channels

        2. for a stream with 65 channels, a group info defined as follows
        [1]
        will create two groups, where the first group has one channel, and the second group has 63 channels
        The first channel could be the time series, and the rest are EEG. The time series obviously should be plotted in a separate plot
        than EEG because they have very different numeric ranges.
    """
    channel_num = preset_dict['num_channels']
    if preset_dict['group_info'] is None:  # group_info will be none if the json file does not contain group_info. In that case, a none group_info will added when calling validate_preset_json_preset
        preset_dict['group_info'] = create_default_group_info(channel_num)
    else:  # the only information the preset conveys is how to divide the channels into groups, only consecutive channels can be grouped together
        plot_group_slice = []
        head = 0
        for x in preset_dict['group_info']:
            plot_group_slice.append((head, x))
            head = x
        if head != channel_num:
            plot_group_slice.append(
                (head, channel_num))  # append the last group
            # create GroupInfo from 0 to x
            # preset_dict['GroupInfo'] = [[channel_index for channel_index in range(0, len(preset_dict['ChannelNames']))]]

        # if preset_dict['GroupFormat'] is None or 'GroupFormat' not in preset_dict:  # default is always time series
        #     preset_dict['GroupFormat'] = ['time_series'] * (len(preset_dict['GroupInfo']))

        preset_dict['group_info'] = dict()
        num_shown_channel = 0
        for i, group in enumerate(plot_group_slice):
            channel_indices = list(range(*group))
            num_available_ch_shown = DEFAULT_CHANNEL_DISPLAY_NUM - num_shown_channel
            if num_available_ch_shown <= 0:
                is_channels_shown = [True] * len(channel_indices)
                # is_channels_shown = [0 for c in range(len(channel_indices))]
            else:
                # is_channels_shown = [1 for c in range(min(len(channel_indices), DEFAULT_CHANNEL_DISPLAY_NUM))]
                # is_channels_shown += [0] * (len(channel_indices) - len(is_channels_shown))  # won't pad if len(channel_indices) - len(is_channels_shown) is negative

                is_channels_shown = [True] * min(len(channel_indices), DEFAULT_CHANNEL_DISPLAY_NUM)
                is_channels_shown += [False] * (len(channel_indices) - len(is_channels_shown))
                num_shown_channel += min(len(channel_indices), DEFAULT_CHANNEL_DISPLAY_NUM)

            # preset_dict['GroupInfo'][f"{default_group_name}{i}"] = \
            #     {
            #         'selected_plot_format': 0,
            #         "plot_format": default_plot_format,
            #         "channel_indices": channel_indices,
            #         "is_channels_shown": is_channels_shown,
            #         "group_description": ""
            #     }
            preset_dict['group_info'][f"{default_group_name}{i}"] = GroupEntry(
                group_name=f"{default_group_name}{i}",
                channel_indices=channel_indices,
                is_channels_shown=is_channels_shown)  # nothing is loaded for plot config

            # preset_dict['GroupInfo'][f"{default_group_name}{i}"] = \
            #     {
            #         "plot_format": default_plot_format,
            #         "channel_indices": channel_indices,
            #         "is_channels_shown": is_channels_shown,
            #         "group_description": ""
            #     }

    return preset_dict

#####################################################################
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
