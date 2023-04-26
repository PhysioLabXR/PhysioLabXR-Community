import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QLabel

from rena import config
from rena.config_ui import image_depth_dict
from rena.utils.settings_utils import get_stream_preset_info, is_group_shown, update_selected_plot_format
from rena.utils.ui_utils import get_distinct_colors, \
    convert_rgb_to_qt_image, convert_array_to_qt_heatmap


class GroupPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent, stream_name, group_name, this_group_info, channel_names, sampling_rate, plot_format_changed_signal):
        """
        :param channel_names: channel names for all the channels in this group
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi("ui/GroupPlotWidget.ui", self)
        self.update_group_name(group_name)

        self.stream_name = stream_name
        self.this_group_info = this_group_info
        self.channel_indices = self.this_group_info['channel_indices']
        self.channel_names =  channel_names
        self.sampling_rate = sampling_rate
        self.channel_index_channel_dict = dict()
        self.channel_plot_item_dict = dict()

        self.linechart_widget = None
        self.image_label = None
        self.barchart_widget = None
        self.legends = None

        if not (is_only_image_enabled := self.this_group_info['is_image_only']):  # a stream will become image only when it has too many channels
            self.init_line_chart()
            self.init_image()
            self.init_bar_chart()
            pass
        else:
            self.init_image()
            # TODO only show the image tab
            self.plot_tabs.setTabEnabled(0, False)
            self.plot_tabs.setTabEnabled(2, False)
            assert self.this_group_info['selected_plot_format'] == 1
        self.viz_time_vector = self.get_viz_time_vector()

        # TODO select the right tab base on the selected_plot_format in the group_info
        self.update_group_shown()  # show or hide this group
        self.plot_tabs.setCurrentIndex(self.this_group_info['selected_plot_format'])

        self.plot_format_changed_signal = plot_format_changed_signal
        self.plot_format_changed_signal.connect(self.plot_format_on_change)
        self.plot_tabs.currentChanged.connect(self.plot_tab_changed)


    def update_image_info(self, new_image_info):
        self.this_group_info['image'] = new_image_info

    @QtCore.pyqtSlot(dict)
    def plot_format_on_change(self, info_dict):
        if self.get_group_name() == info_dict['group_name']:
            self.plot_tabs.currentChanged.disconnect(self.plot_tab_changed)
            self.plot_tabs.setCurrentIndex(info_dict['new_format'])
            self.plot_tabs.currentChanged.connect(self.plot_tab_changed)

    def plot_tab_changed(self, index):
        update_selected_plot_format(self.stream_name, self.get_group_name(), index)
        info_dict = {
            'stream_name': self.stream_name,
            'group_name': self.get_group_name(),
            'new_format': index
        }
        self.this_group_info['selected_plot_format'] = index
        self.plot_format_changed_signal.emit(info_dict)

    def init_line_chart(self):
        self.linechart_widget = pg.PlotWidget()
        self.linechart_layout.addWidget(self.linechart_widget)
        distinct_colors = get_distinct_colors(len(self.this_group_info['channel_indices']))
        self.legends = self.linechart_widget.addLegend()
        # self.linechart_widget.enableAutoRange(enable=False)
        for channel_index_in_group, (channel_index, channel_name) in enumerate(
                zip(self.channel_indices, self.channel_names)):
            is_channel_shown = self.this_group_info['is_channels_shown'][channel_index_in_group]
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
        self.barchart_widget.setYRange(self.this_group_info['plot_format']['bar_chart']['y_min'], self.this_group_info['plot_format']['bar_chart']['y_max'])
        # barchart_widget.sigRangeChanged.connect(self.bar_chart_range_changed)
        # barchart_widget.setLimits(xMin=-0.5, xMax=len(self.group_info[group_name]['channel_indices']), yMin=plot_format['bar_chart']['y_min'], yMax=plot_format['bar_chart']['y_max'])
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
        if is_group_shown(self.get_group_name(), self.stream_name):  # get is group_shown from preset
            self.plot_tabs.show()
        else:
            self.plot_tabs.hide()

    def update_channel_shown(self):
        # TODO when the channels are checking and unchecked from stream options
        pass
    def update_group_name(self, group_name):
        self.group_name_label.setText(group_name)

    def get_group_name(self):
        return self.group_name_label.text()

    def get_selected_format(self):
        return self.plot_tabs.currentIndex()

    def get_viz_time_vector(self):
        display_duration = get_stream_preset_info(self.stream_name, 'DisplayDuration')
        num_points_to_plot = int(display_duration * get_stream_preset_info(self.stream_name, 'NominalSamplingRate'))
        return np.linspace(0., get_stream_preset_info(self.stream_name, 'DisplayDuration'), num_points_to_plot)

    def plot_data(self, data):
        if data.shape[1] != len(self.viz_time_vector):  # num_points_to_plot has been updated
            self.viz_time_vector = self.get_viz_time_vector()
        if self.get_selected_format() == 0 and self.this_group_info["plot_format"]['time_series']['is_valid']:
            for index_in_group, channel_index in enumerate(self.this_group_info['channel_indices']):
                plot_data_item = self.linechart_widget.plotItem.curves[index_in_group]
                if plot_data_item.isVisible():
                    plot_data_item.setData(self.viz_time_vector, data[int(channel_index), :])
        elif self.get_selected_format() == 1 and self.this_group_info["plot_format"]['image']['is_valid']:
            width, height, depth, image_format, channel_format, scaling_factor = self.get_image_format_and_shape(self.get_group_name())
            image_plot_data = data[self.this_group_info['channel_indices'], -1]  # only visualize the last frame
            if image_format == 'RGB':
                if channel_format == 'Channel First':
                    image_plot_data = np.reshape(image_plot_data, (depth, height, width))
                    image_plot_data = np.moveaxis(image_plot_data, 0, -1)
                elif channel_format == 'Channel Last':
                    image_plot_data = np.reshape(image_plot_data, (height, width, depth))
                # image_plot_data = convert_numpy_to_uint8(image_plot_data)
                image_plot_data = image_plot_data.astype(np.uint8)
                image_plot_data = convert_rgb_to_qt_image(image_plot_data, scaling_factor=scaling_factor)
                self.image_label.setPixmap(image_plot_data)

            # if we chose PixelMap
            if image_format == 'PixelMap':
                # pixel map return value
                image_plot_data = np.reshape(image_plot_data, (height, width))  # matrix : (height, width)
                image_plot_data = convert_array_to_qt_heatmap(image_plot_data, scaling_factor=scaling_factor)
                self.image_label.setPixmap(image_plot_data)
        elif self.get_selected_format() == 2 and self.this_group_info["plot_format"]['bar_chart']['is_valid']:
            bar_chart_plot_data = data[self.this_group_info['channel_indices'], -1]  # only visualize the last frame
            self.barchart_widget.plotItem.curves[0].setOpts(x=np.arange(len(bar_chart_plot_data)), height=bar_chart_plot_data, width=1, brush='r')

    def get_image_format_and_shape(self, group_name):
        width = self.this_group_info['plot_format']['image']['width']
        height = self.this_group_info['plot_format']['image']['height']
        image_format = self.this_group_info['plot_format']['image']['image_format']
        depth = image_depth_dict[image_format]
        channel_format = self.this_group_info['plot_format']['image']['channel_format']
        scaling_factor = self.this_group_info['plot_format']['image']['scaling_factor']

        return width, height, depth, image_format, channel_format, scaling_factor

    def update_bar_chart_range(self, new_group_info):
        self.this_group_info = new_group_info
        if not self.this_group_info['is_image_only']:  # if barplot exists for this group
            self.barchart_widget.setYRange(min=self.this_group_info['plot_format']['bar_chart']['y_min'], max=self.this_group_info['plot_format']['bar_chart']['y_max'])

    def on_plot_format_change(self):
        '''
        emit selected group and changed to the stream widget, and to the stream options
        '''
        pass

    def change_group_name(self, new_group_name):
        self.update_group_name(new_group_name)

    def change_channel_name(self, new_ch_name, old_ch_name, lsl_index):
        # change_plot_label(self.linechart_widget, self.channel_plot_item_dict[old_ch_name], new_ch_name)
        self.channel_plot_item_dict[old_ch_name].setData(name=new_ch_name)
        self.channel_plot_item_dict[new_ch_name] = self.channel_plot_item_dict.pop(old_ch_name)

        # self.channel_plot_item_dict[old_ch_name].legend.setText(new_ch_name)
        index_in_group = self.this_group_info['channel_indices'].index(lsl_index)
        self.legends.items[index_in_group][1].setText(new_ch_name)

