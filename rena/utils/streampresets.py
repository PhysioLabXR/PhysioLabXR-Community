from rena.utils.settings_utils import collect_stream_all_groups_info


class StreamPresets:
    def __init__(self, stream_name):
        pass
        self.stream_name = stream_name
        # group_info = {}
        self.group_info = None
        group_info = collect_stream_all_groups_info(self.stream_name)

    '''
        group_info saved in memory for rapid plotting
    '''
    def get_group_channel_indices(self, group_name):
        return self.group_info[group_name]['group_channel_indices']

    def get_group_index(self, group_name):
        return self.group_info[group_name]['group_index']

    def get_group_is_channels_shown(self, group_name):
        return self.group_info[group_name]['group_is_channels_shown']

    def get_group_selected_plot_format(self, group_name):
        return self.group_info[group_name]['group_selected_plot_format']

    