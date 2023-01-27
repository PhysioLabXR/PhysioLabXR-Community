from rena.config_ui import plot_format_index_dict
from rena.ui.FilterComponentButterworthBandPass import FilterComponentButterworthBandPass
from rena.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from rena.utils.settings_utils import collect_stream_all_groups_info


class StreamPresets:
    def __init__(self, stream_name):
        pass
        self.stream_name = stream_name
        # group_info = {}
        self.group_info = None
        # init all stream group info
        self.collect_stream_all_groups_info()
        self.group_info_widgets = {}

        ## self.filter_widget
        ## self.plot_format_widget


    ## while the group been selected, we update the info widget

    '''
        group_info saved in memory for rapid plotting
    '''
    def get_group_channel_indices(self, group_name):
        return self.group_info[group_name]['group_channel_indices']

    def get_group_index(self, group_name):
        return self.group_info[group_name]['group_index']

    def get_group_is_channels_shown(self, group_name):
        return self.group_info[group_name]['group_is_channels_shown']

    def get_selected_plot_format(self, group_name):
        return self.group_info[group_name]['group_selected_plot_format']

    def get_group_format(self, group_name):
        return self.group_info[group_name]['plot_format']

    ##############################
    def get_plot_format_time_series_display(self, group_name):
        return self.group_info[group_name][plot_format_index_dict[0]]['display']

    def get_plot_format_time_series_is_valid(self, group_name):
        return self.group_info[group_name][plot_format_index_dict[0]]['is_valid']

    def get_plot_format_image_display(self, group_name):
        return self.group_info[group_name][plot_format_index_dict[0]]['display']

    def get_plot_format_image_is_valid(self, group_name):
        return self.group_info[group_name][plot_format_index_dict[0]]['is_valid']

    def collect_stream_all_groups_info(self):
        self.group_info = collect_stream_all_groups_info(self.stream_name)

    def init_group_filter_widgets(self):
        for group_name in self.group_info:
            self.group_info_widgets[group_name] = OptionsWindowPlotFormatWidget(stream_name=self.stream_name)

    def init_butter_worth_band_pass_filter_widget(self, lowcut, highcut, fs, order, channel_num):
        butter_worth_band_pass_filter_widget = FilterComponentButterworthBandPass(args=None)
        butter_worth_band_pass_filter_widget.init_filter(lowcut=lowcut, highcut=highcut, fs=fs, order=order, channel_num=channel_num)
        return butter_worth_band_pass_filter_widget

    # def export_filter_info_to_settings(self, stream_name, group_name, args):


