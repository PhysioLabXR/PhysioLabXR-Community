import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QLabel

from rena import config
from rena.presets.GroupEntry import PlotFormat
from rena.presets.PlotConfig import ImageFormat, ChannelFormat
from rena.presets.presets_utils import get_stream_preset_info, get_is_group_shown, \
    set_stream_a_group_selected_plot_format, \
    is_group_image_only, get_bar_chart_max_min_range, get_selected_plot_format, get_selected_plot_format_index, \
    get_group_channel_indices, get_group_image_valid, get_group_image_config
from rena.utils.ui_utils import get_distinct_colors, \
    convert_rgb_to_qt_image, convert_array_to_qt_heatmap


class GroupPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent, stream_name, group_name, channel_names, sampling_rate, plot_format_changed_signal):
        """
        :param channel_names: channel names for all the channels in this group
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi("ui/GroupPlotWidget.ui", self)
        self.update_group_name(group_name)

        self.stream_name = stream_name
        self.group_name = group_name
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.channel_index_channel_dict = dict()
        self.channel_plot_item_dict = dict()

        self.linechart_widget = None
        self.image_label = None
        self.barchart_widget = None
        self.legends = None

        if not is_group_image_only(self.stream_name, group_name):  # a stream will become image only when it has too many channels
            self.init_line_chart()
            self.init_image()
            self.init_bar_chart()
            pass
        else:
            self.init_image()
            self.plot_tabs.setTabEnabled(0, False)
            self.plot_tabs.setTabEnabled(2, False)
            assert get_selected_plot_format(self.stream_name, group_name) == PlotFormat.IMAGE
        self.viz_time_vector = self.get_viz_time_vector()

        self.update_group_shown()  # show or hide this group
        self.plot_tabs.setCurrentIndex(get_selected_plot_format_index(self.stream_name, group_name))

        self.plot_format_changed_signal = plot_format_changed_signal
        self.plot_format_changed_signal.connect(self.plot_format_on_change)
        self.plot_tabs.currentChanged.connect(self.plot_tab_changed)

    # def update_image_info(self, new_image_info):
    #     self.this_group_info['image'] = new_image_info

    @QtCore.pyqtSlot(dict)
    def plot_format_on_change(self, info_dict):
        if self.group_name == info_dict['group_name']:
            self.plot_tabs.currentChanged.disconnect(self.plot_tab_changed)
            self.plot_tabs.setCurrentIndex(info_dict['new_format'].value)  # get the value of the PlotFormat enum
            self.plot_tabs.currentChanged.connect(self.plot_tab_changed)

    def plot_tab_changed(self, index):
        new_plot_format = set_stream_a_group_selected_plot_format(self.stream_name, self.group_name, index)
        info_dict = {
            'stream_name': self.stream_name,
            'group_name': self.group_name,
            'new_format': new_plot_format
        }
        self.plot_format_changed_signal.emit(info_dict)

    def init_line_chart(self):
        self.linechart_widget = pg.PlotWidget()
        self.linechart_layout.addWidget(self.linechart_widget)

        channel_indices = get_group_channel_indices(self.stream_name, self.group_name)
        is_channels_shown = get_is_group_shown(self.stream_name, self.group_name)

        distinct_colors = get_distinct_colors(len(channel_indices))
        self.legends = self.linechart_widget.addLegend()
        # self.linechart_widget.enableAutoRange(enable=False)
        for channel_index_in_group, (channel_index, channel_name) in enumerate(
                zip(channel_indices, self.channel_names)):
            is_channel_shown = is_channels_shown[channel_index_in_group]
            channel_plot_item = self.linechart_widget.plot([], [], pen=pg.mkPen(color=distinct_colors[channel_index_in_group]), name=channel_name)
            self.channel_index_channel_dict[int(channel_index)] = channel_plot_item
            if not is_channel_shown:
                channel_plot_item.hide()  # TODO does disable do what it should do: uncheck from the plots
            downsample_method = 'mean' if self.sampling_rate > config.settings.value('downsample_method_mean_sr_threshold') else 'subsample'
            channel_plot_item.setDownsampling(auto=True, method=downsample_method)
            channel_plot_item.setClipToView(True)
            channel_plot_item.setSkipFiniteCheck(True)
            self.channel_plot_item_dict[channel_name] = channel_plot_item

    def init_image(self):
        self.image_label = QLabel('Image_Label')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_layout.addWidget(self.image_label)

    def init_bar_chart(self):
        self.barchart_widget = pg.PlotWidget()
        self.barchart_layout.addWidget(self.barchart_widget)
        barchar_min, barchar_max = get_bar_chart_max_min_range(self.stream_name, self.group_name)

        self.barchart_widget.setYRange(barchar_min, barchar_max)
        label_x_axis = self.barchart_widget.getAxis('bottom')
        label_dict = dict(enumerate(self.channel_names)).items()
        label_x_axis.setTicks([label_dict])
        x = np.arange(len(self.channel_names))
        y = np.array([0] * len(self.channel_names))
        bars = pg.BarGraphItem(x=x, height=y, width=1, brush='r')
        self.barchart_widget.addItem(bars)

    def update_group_shown(self):
        # assuming group info is update to date with in the persistent settings
        # check if there's active channels in this group
        if get_is_group_shown(self.stream_name, self.group_name):  # get is group_shown from preset
            self.plot_tabs.show()
        else:
            self.plot_tabs.hide()

    def update_channel_shown(self):
        # TODO when the channels are checking and unchecked from stream options
        pass

    def update_group_name(self, group_name):
        self.group_name_label.setText(group_name)
        self.group_name = group_name

    def get_selected_format(self):
        return self.plot_tabs.currentIndex()

    def get_viz_time_vector(self):
        display_duration = get_stream_preset_info(self.stream_name, 'display_duration')
        num_points_to_plot = int(display_duration * get_stream_preset_info(self.stream_name, 'nominal_sampling_rate'))
        return np.linspace(0., get_stream_preset_info(self.stream_name, 'display_duration'), num_points_to_plot)

    def plot_data(self, data):
        channel_indices = get_group_channel_indices(self.stream_name, self.group_name)

        if data.shape[1] != len(self.viz_time_vector):  # num_points_to_plot has been updated
            self.viz_time_vector = self.get_viz_time_vector()
        if self.get_selected_format() == 0:
            for index_in_group, channel_index in enumerate(channel_indices):
                plot_data_item = self.linechart_widget.plotItem.curves[index_in_group]
                if plot_data_item.isVisible():
                    plot_data_item.setData(self.viz_time_vector, data[int(channel_index), :])
        elif self.get_selected_format() == 1 and get_group_image_valid(self.stream_name, self.group_name):
            image_config = get_group_image_config(self.stream_name, self.group_name)
            width, height, image_format, channel_format, scaling = image_config.width, image_config.height, image_config.image_format, image_config.channel_format, image_config.scaling
            depth = image_format.depth_dim()
            image_plot_data = data[channel_indices, -1]  # only visualize the last frame
            if image_format == ImageFormat.rgb:
                if channel_format == ChannelFormat.channel_last:
                    image_plot_data = np.reshape(image_plot_data, (depth, height, width))
                    image_plot_data = np.moveaxis(image_plot_data, 0, -1)
                elif channel_format == ChannelFormat.channel_first:
                    image_plot_data = np.reshape(image_plot_data, (height, width, depth))
                # image_plot_data = convert_numpy_to_uint8(image_plot_data)
                image_plot_data = image_plot_data.astype(np.uint8)
                image_plot_data = convert_rgb_to_qt_image(image_plot_data, scaling_factor=scaling)
                self.image_label.setPixmap(image_plot_data)

            # if we chose PixelMap
            if image_format == ImageFormat.pixelmap:
                # pixel map return value
                image_plot_data = np.reshape(image_plot_data, (height, width))  # matrix : (height, width)
                image_plot_data = convert_array_to_qt_heatmap(image_plot_data, scaling_factor=scaling)
                self.image_label.setPixmap(image_plot_data)
        elif self.get_selected_format() == 2:
            bar_chart_plot_data = data[channel_indices, -1]  # only visualize the last frame
            self.barchart_widget.plotItem.curves[0].setOpts(x=np.arange(len(bar_chart_plot_data)), height=bar_chart_plot_data, width=1, brush='r')

    def update_bar_chart_range(self):
        if not is_group_image_only(self.stream_name, self.group_name):  # if barplot exists for this group
            barchart_min, barchat_max = get_bar_chart_max_min_range(self.stream_name, self.group_name)
            self.barchart_widget.setYRange(min=barchart_min, max=barchat_max)

    # def on_plot_format_change(self):
    #     '''
    #     emit selected group and changed to the stream widget, and to the stream options
    #     '''
    #     pass

    def change_group_name(self, new_group_name):
        self.update_group_name(new_group_name)

    def change_channel_name(self, new_ch_name, old_ch_name, lsl_index):
        # change_plot_label(self.linechart_widget, self.channel_plot_item_dict[old_ch_name], new_ch_name)
        self.channel_plot_item_dict[old_ch_name].setData(name=new_ch_name)
        self.channel_plot_item_dict[new_ch_name] = self.channel_plot_item_dict.pop(old_ch_name)

        # self.channel_plot_item_dict[old_ch_name].legend.setText(new_ch_name)
        channel_indices = get_group_channel_indices(self.stream_name, self.group_name)
        index_in_group = channel_indices.index(lsl_index)
        self.legends.items[index_in_group][1].setText(new_ch_name)

