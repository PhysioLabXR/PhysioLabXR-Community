# This Python file uses the following encoding: utf-8

from PyQt6 import QtCore, QtWidgets
from PyQt6 import uic
from PyQt6.QtGui import QIntValidator, QDoubleValidator

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.Cmap import Cmap
from physiolabxr.presets.PlotConfig import ImageFormat, ChannelFormat
from physiolabxr.presets.presets_utils import get_stream_a_group_info, \
    set_stream_a_group_selected_plot_format, set_stream_a_group_selected_img_h_w, \
    set_bar_chart_max_min_range, set_group_image_format, set_group_image_channel_format, \
    get_group_image_config, set_spectrogram_time_per_segment, set_spectrogram_time_overlap, \
    get_spectrogram_time_per_segment, get_spectrogram_time_overlap, set_spectrogram_cmap, \
    set_spectrogram_percentile_level_min, set_spectrogram_percentile_level_max, set_image_levels_max, \
    set_image_levels_min, get_valid_image_levels, set_image_cmap, \
    set_image_levels_rmin, set_image_levels_bmax, set_image_levels_bmin, set_image_levels_gmax, set_image_levels_gmin, \
    set_image_levels_rmax, set_image_scaling_percentile, get_image_levels
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel
from physiolabxr.utils.Validators import NoCommaIntValidator


class OptionsWindowPlotFormatWidget(QtWidgets.QWidget):
    image_change_signal = QtCore.pyqtSignal(dict)

    def __init__(self, parent, stream_widget, stream_name, plot_format_changed_signal):
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        # self.setWindowTitle('Options')
        self.ui = uic.loadUi(AppConfigs()._ui_OptionsWindowPlotFormatWidget, self)
        self.stream_name = stream_name
        self.group_name = None
        self.parent = parent
        self.stream_widget = stream_widget
        self.plot_format_changed_signal = plot_format_changed_signal
        self.plot_format_changed_signal.connect(self.plot_format_tab_changed)

        self.plotFormatTabWidget.currentChanged.connect(self.plot_format_tab_selection_changed)

        # image ###############################################################
        self.imageWidthLineEdit.setValidator(NoCommaIntValidator())
        self.imageHeightLineEdit.setValidator(NoCommaIntValidator())
        self.imageFormatComboBox.addItems([format.name for format in ImageFormat])
        self.channelFormatCombobox.addItems([format.name for format in ChannelFormat])

        self.image_level_min_line_edit.setValidator(QDoubleValidator())
        self.image_level_max_line_edit.setValidator(QDoubleValidator())
        self.combobox_image_cmap.addItems([name for name, member in Cmap.__members__.items()])
        self.image_levels_invalid = False
        self.image_display_scaling_percentile_slider = SliderWithValueLabel(minimum=1, maximum=100, value=100)
        self.display_scaling_widget.layout().addWidget(self.image_display_scaling_percentile_slider)

        # barplot ###############################################################
        self.barPlotYMaxLineEdit.setValidator(QDoubleValidator())
        self.barPlotYMinLineEdit.setValidator(QDoubleValidator())

        # spectrogram ###############################################################
        self.line_edit_time_per_segments.setValidator(QDoubleValidator())
        self.line_edit_overlap_between_segments.setValidator(QDoubleValidator())
        self.label_invalid_spectrogram_param.setStyleSheet("color: red")

        self.slider_spectrogram_percentile_max = SliderWithValueLabel(minimum=1, maximum=100, value=100)
        self.slider_spectrogram_percentile_min = SliderWithValueLabel(minimum=0, maximum=99, value=1)
        self.spectrogram_gridLayout.addWidget(self.slider_spectrogram_percentile_max, 6, 1)
        self.spectrogram_gridLayout.addWidget(self.slider_spectrogram_percentile_min, 5, 1)
        self.comboBox_spectrogram_cmap.addItems([name for name, member in Cmap.__members__.items()])

        self.last_time_per_segment = None
        self.last_time_overlap = None

    def set_plot_format_widget_info(self, group_name):
        self._set_to_group(group_name)

    def _set_to_group(self, group_name):
        this_group_entry = get_stream_a_group_info(self.stream_name, group_name)
        # disconnect while switching selected group
        self.plotFormatTabWidget.currentChanged.disconnect()
        self.plotFormatTabWidget.setCurrentIndex(this_group_entry.selected_plot_format.value)
        if this_group_entry.is_image_only():
            self.enable_only_image_tab()
        self.plotFormatTabWidget.currentChanged.connect(self.plot_format_tab_selection_changed)
        # self.plot_format_changed_signal.connect(self.plot_format_changed)

        if self.group_name is not None:
            # barplot ###############################################################
            self.barPlotYMaxLineEdit.textChanged.disconnect()
            self.barPlotYMinLineEdit.textChanged.disconnect()

            # image ###############################################################
            self.imageWidthLineEdit.textChanged.disconnect()
            self.imageHeightLineEdit.textChanged.disconnect()
            self.image_display_scaling_percentile_slider.valueChanged.disconnect()
            self.imageFormatComboBox.currentTextChanged.disconnect()
            self.channelFormatCombobox.currentTextChanged.disconnect()
            self.combobox_image_cmap.currentTextChanged.disconnect()

            self.image_level_min_line_edit.textChanged.disconnect()
            self.image_level_max_line_edit.textChanged.disconnect()

            self.image_level_rmin_line_edit.textChanged.disconnect()
            self.image_level_rmax_line_edit.textChanged.disconnect()
            self.image_level_gmin_line_edit.textChanged.disconnect()
            self.image_level_gmax_line_edit.textChanged.disconnect()
            self.image_level_bmin_line_edit.textChanged.disconnect()
            self.image_level_bmax_line_edit.textChanged.disconnect()

            # spectrogram ###############################################################
            self.line_edit_time_per_segments.textChanged.disconnect()
            self.line_edit_overlap_between_segments.textChanged.disconnect()

            self.slider_spectrogram_percentile_min.valueChanged.disconnect(self.spectrogram_percentile_min_changed)  # only disconnect this one, as there are other signals connected to the same slot
            self.slider_spectrogram_percentile_max.valueChanged.disconnect(self.spectrogram_percentile_max_changed)

        # image ###############################################################
        self.imageWidthLineEdit.setText(str(this_group_entry.plot_configs.image_config.width))
        self.imageHeightLineEdit.setText(str(this_group_entry.plot_configs.image_config.height))
        self.image_display_scaling_percentile_slider.setValue(this_group_entry.plot_configs.image_config.scaling_percentage)

        self.imageFormatComboBox.setCurrentText(this_group_entry.plot_configs.image_config.image_format.name)
        self.channelFormatCombobox.setCurrentText(this_group_entry.plot_configs.image_config.channel_format.name)

        valid_levels = get_valid_image_levels(self.stream_name, group_name)
        self.update_image_level_valid(valid_levels)
        image_format = this_group_entry.plot_configs.image_config.image_format
        if image_format == ImageFormat.rgb or image_format == ImageFormat.bgr:
            self.pixelmap_display_options.setVisible(False)
            self.rgb_display_options.setVisible(True)
        elif image_format == ImageFormat.pixelmap:
            self.rgb_display_options.setVisible(False)
            self.pixelmap_display_options.setVisible(True)

        levels_for_update = get_image_levels(self.stream_name, group_name)
        if image_format == ImageFormat.rgb:
            self.image_level_rmin_line_edit.setText(str(levels_for_update[0][0]))
            self.image_level_rmax_line_edit.setText(str(levels_for_update[0][1]))
            self.image_level_gmin_line_edit.setText(str(levels_for_update[1][0]))
            self.image_level_gmax_line_edit.setText(str(levels_for_update[1][1]))
            self.image_level_bmin_line_edit.setText(str(levels_for_update[2][0]))
            self.image_level_bmax_line_edit.setText(str(levels_for_update[2][1]))
        elif image_format == ImageFormat.pixelmap:
            self.image_level_min_line_edit.setText(str(levels_for_update[0]))
            self.image_level_max_line_edit.setText(str(levels_for_update[1]))

        self.combobox_image_cmap.setCurrentIndex(this_group_entry.plot_configs.image_config.cmap.value)

        self.image_level_min_line_edit.textChanged.connect(self.image_level_min_line_edit_changed)
        self.image_level_max_line_edit.textChanged.connect(self.image_level_max_line_edit_changed)
        self.image_level_rmin_line_edit.textChanged.connect(self.image_level_rmin_line_edit_changed)
        self.image_level_rmax_line_edit.textChanged.connect(self.image_level_rmax_line_edit_changed)
        self.image_level_gmin_line_edit.textChanged.connect(self.image_level_gmin_line_edit_changed)
        self.image_level_gmax_line_edit.textChanged.connect(self.image_level_gmax_line_edit_changed)
        self.image_level_bmin_line_edit.textChanged.connect(self.image_level_bmin_line_edit_changed)
        self.image_level_bmax_line_edit.textChanged.connect(self.image_level_bmax_line_edit_changed)

        self.combobox_image_cmap.currentTextChanged.connect(self.image_cmap_changed)

        self.imageWidthLineEdit.textChanged.connect(self.image_w_h_scaling_percentile_on_change)
        self.imageHeightLineEdit.textChanged.connect(self.image_w_h_scaling_percentile_on_change)
        self.image_display_scaling_percentile_slider.valueChanged.connect(self.image_scaling_percentile_on_change)
        self.imageFormatComboBox.currentTextChanged.connect(self.image_format_change)
        self.channelFormatCombobox.currentTextChanged.connect(self.image_channel_format_change)

        # barplot ###############################################################
        self.barPlotYMaxLineEdit.setText(str(this_group_entry.plot_configs.barchart_config.y_max))
        self.barPlotYMinLineEdit.setText(str(this_group_entry.plot_configs.barchart_config.y_min))

        self.barPlotYMaxLineEdit.textChanged.connect(self.bar_chart_range_on_change)
        self.barPlotYMinLineEdit.textChanged.connect(self.bar_chart_range_on_change)

        # spectrogram ###############################################################
        self.line_edit_time_per_segments.setText(str(this_group_entry.plot_configs.spectrogram_config.time_per_segment_second))
        self.line_edit_overlap_between_segments.setText(str(this_group_entry.plot_configs.spectrogram_config.time_overlap_second))

        self.line_edit_time_per_segments.textChanged.connect(self.time_per_segment_changed)
        self.line_edit_overlap_between_segments.textChanged.connect(self.time_overlap_changed)

        self.last_time_per_segment = this_group_entry.plot_configs.spectrogram_config.time_per_segment_second
        self.last_time_overlap = this_group_entry.plot_configs.spectrogram_config.time_overlap_second

        self.slider_spectrogram_percentile_min.setValue(this_group_entry.plot_configs.spectrogram_config.percentile_level_min)
        self.slider_spectrogram_percentile_max.setValue(this_group_entry.plot_configs.spectrogram_config.percentile_level_max)
        self.slider_spectrogram_percentile_min.valueChanged.connect(self.spectrogram_percentile_min_changed)
        self.slider_spectrogram_percentile_max.valueChanged.connect(self.spectrogram_percentile_max_changed)

        self.label_invalid_spectrogram_param.setVisible(False)
        self.comboBox_spectrogram_cmap.setCurrentIndex(this_group_entry.plot_configs.spectrogram_config.cmap.value)
        self.parent.set_spectrogram_cmap(group_name)  # Call stack: StreamOptionsWindow -> StreamWindow -> VizComponents -> GroupPlotWidget
        self.comboBox_spectrogram_cmap.currentTextChanged.connect(self.spectrogram_cmap_changed)

        self.group_name = group_name
        self.image_valid_update()

    def spectrogram_percentile_min_changed(self, value):
        if value >= self.slider_spectrogram_percentile_max.value():
            self.slider_spectrogram_percentile_max.setValue(value + 1)
        set_spectrogram_percentile_level_min(self.stream_name, self.group_name, value)

    def spectrogram_percentile_max_changed(self, value):
        if value <= self.slider_spectrogram_percentile_min.value():
            self.slider_spectrogram_percentile_min.setValue(value - 1)
        set_spectrogram_percentile_level_max(self.stream_name, self.group_name, value)

    def plot_format_tab_selection_changed(self, index):
        # create value
        new_plot_format = set_stream_a_group_selected_plot_format(self.stream_name, self.group_name, index)

        # new format, old format
        info_dict = {
            'stream_name': self.stream_name,
            'group_name': self.group_name,
            'new_format': new_plot_format
        }
        self.plot_format_changed_signal.disconnect(self.plot_format_tab_changed)
        self.plot_format_changed_signal.emit(info_dict)
        self.plot_format_changed_signal.connect(self.plot_format_tab_changed)

    @QtCore.pyqtSlot(dict)
    def plot_format_tab_changed(self, info_dict):
        if self.group_name == info_dict['group_name']:  # if current selected group is the plot-format-changed group
            self._set_to_group(self.group_name)

    def image_w_h_scaling_percentile_on_change(self):
        width = self.get_image_width()
        height = self.get_image_height()
        set_stream_a_group_selected_img_h_w(self.stream_name, self.group_name, height=height, width=width)
        self.image_changed()

    def image_scaling_percentile_on_change(self):
        scaling_percentile = self.image_display_scaling_percentile_slider.value()
        set_image_scaling_percentile(self.stream_name, self.group_name, scaling_percentile=scaling_percentile)
        self.image_changed()

    def image_format_change(self):
        image_format = self.get_image_format()
        set_group_image_format(self.stream_name, self.group_name, image_format=image_format)

        if image_format == ImageFormat.rgb or image_format == ImageFormat.bgr:
            self.pixelmap_display_options.setVisible(False)
            self.rgb_display_options.setVisible(True)
        elif image_format == ImageFormat.pixelmap:
            self.rgb_display_options.setVisible(False)
            self.pixelmap_display_options.setVisible(True)
        self.image_changed()

    def image_channel_format_change(self):
        image_channel_format = self.get_image_channel_format()
        set_group_image_channel_format(self.stream_name, self.group_name, channel_format=image_channel_format)
        if image_channel_format == ImageFormat.rgb or image_channel_format == ImageFormat.bgr:
            self.pixelmap_display_options.setVisible(False)
            self.rgb_display_options.setVisible(True)
        elif image_channel_format == ImageFormat.pixelmap:
            self.rgb_display_options.setVisible(False)
            self.pixelmap_display_options.setVisible(True)
        self.image_changed()

    def image_valid_update(self):
        if self.group_name is not None:
            image_config = get_group_image_config(self.stream_name, self.group_name)
            group_info = get_stream_a_group_info(self.stream_name, self.group_name)
            channel_num = group_info.get_num_channels()
            width, height, image_format, channel_format = image_config.width, image_config.height, image_config.image_format, image_config.channel_format

            self.imageFormatInfoLabel.setText('Width x Height x Depth = {0} \n Group Channel Number = {1}'.format(
                str(width * height * image_format.depth_dim()), str(channel_num)
            ))

            if get_stream_a_group_info(self.stream_name, self.group_name).is_image_valid():
                self.imageFormatInfoLabel.setStyleSheet('color: green')
                # print('Valid Image Format')
            else:
                self.imageFormatInfoLabel.setStyleSheet('color: red')
                # print('Invalid Image Format')

    def spectrogram_valid_update(self, is_valid):
        self.label_invalid_spectrogram_param.setVisible(not is_valid)

    def get_image_width(self):
        try:
            new_image_width = abs(int(self.imageWidthLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_image_width

    def get_image_height(self):
        try:
            new_image_height = abs(int(self.imageHeightLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_image_height

    def get_bar_chart_max_range(self):
        try:
            new_bar_chart_max_range = float(self.barPlotYMaxLineEdit.text())
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_bar_chart_max_range

    def get_bar_chart_min_range(self):
        try:
            new_bar_chart_min_range = float(self.barPlotYMinLineEdit.text())
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_bar_chart_min_range

    def get_image_format(self):
        current_format = self.imageFormatComboBox.currentText()
        # image_channel_num = image_depth_dict(current_format)
        return ImageFormat.__members__[current_format]

    def get_image_channel_format(self):
        current_format = self.channelFormatCombobox.currentText()
        # image_channel_num = image_depth_dict(current_format)
        return ChannelFormat.__members__[current_format]

    def image_changed(self):
        self.image_valid_update()
        self.image_change_signal.emit({'group_name': self.group_name, 'this_group_info_image': get_group_image_config(self.stream_name, self.group_name)})

    def bar_chart_range_on_change(self):
        bar_chart_max = self.get_bar_chart_max_range()
        bar_chart_min = self.get_bar_chart_min_range()

        set_bar_chart_max_min_range(self.stream_name, self.group_name, max_range=bar_chart_max,  min_range=bar_chart_min)  # change in the settings
        self.stream_widget.bar_chart_range_on_change(self.group_name)

    def enable_only_image_tab(self):
        self.plotFormatTabWidget.setTabEnabled(0, False)
        self.plotFormatTabWidget.setTabEnabled(2, False)
        self.plotFormatTabWidget.setTabEnabled(3, False)

    def change_group_name(self, new_name):
        self.group_name = new_name

    def time_per_segment_changed(self):
        """
        the invalid check ensures that invalid values are never saved to the preset
        """
        try:
            time_per_segment = float(self.line_edit_time_per_segments.text())
        except ValueError:
            time_per_segment = 0

        if time_per_segment < get_spectrogram_time_overlap(self.stream_name, self.group_name):
            time_per_segment = 0

        if time_per_segment == 0:
            time_per_segment = self.last_time_per_segment
            self.spectrogram_valid_update(False)
        else:
            self.last_time_per_segment = time_per_segment
            self.spectrogram_valid_update(True)

        set_spectrogram_time_per_segment(self.stream_name, self.group_name, time_per_segment)

    def time_overlap_changed(self):
        """
        the invalid check ensures that invalid values are never saved to the preset
        """
        try:
            overlap = float(self.line_edit_overlap_between_segments.text())
        except ValueError:
            overlap = 0
        if overlap > get_spectrogram_time_per_segment(self.stream_name, self.group_name):
            overlap = 0

        if overlap == 0:
            overlap = self.last_time_overlap
            self.spectrogram_valid_update(False)
        else:
            self.spectrogram_valid_update(True)
            self.last_time_overlap = overlap
        set_spectrogram_time_overlap(self.stream_name, self.group_name, overlap)

    def spectrogram_cmap_changed(self):
        selected_cmap = getattr(Cmap, self.comboBox_spectrogram_cmap.currentText())
        set_spectrogram_cmap(self.stream_name, self.group_name, selected_cmap)
        self.parent.set_spectrogram_cmap(self.group_name)

    def image_cmap_changed(self):
        selected_cmap = getattr(Cmap, self.combobox_image_cmap.currentText())
        set_image_cmap(self.stream_name, self.group_name, selected_cmap)
        self.parent.get_viz_components().group_plots[self.group_name].set_image_cmap()  # Call stack: StreamOptionsWindow -> StreamWindow -> VizComponents -> GroupPlotWidget

    def image_level_max_line_edit_changed(self):
        try:
            level_max = float(self.image_level_max_line_edit.text())
        except ValueError:
            return
        set_image_levels_max(self.stream_name, self.group_name, level_max)
        self.image_level_changed()

    def image_level_min_line_edit_changed(self):
        try:
            level_min = float(self.image_level_min_line_edit.text())
        except ValueError:
            return
        set_image_levels_min(self.stream_name, self.group_name, level_min)
        self.image_level_changed()

    def image_level_rmin_line_edit_changed(self):
        try:
            level_rmin = float(self.image_level_rmin_line_edit.text())
        except ValueError:
            return
        set_image_levels_rmin(self.stream_name, self.group_name, level_rmin)
        self.image_level_changed()

    def image_level_rmax_line_edit_changed(self):
        try:
            level_rmax = float(self.image_level_rmax_line_edit.text())
        except ValueError:
            return
        set_image_levels_rmax(self.stream_name, self.group_name, level_rmax)
        self.image_level_changed()

    def image_level_gmin_line_edit_changed(self):
        try:
            level_gmin = float(self.image_level_gmin_line_edit.text())
        except ValueError:
            return
        set_image_levels_gmin(self.stream_name, self.group_name, level_gmin)
        self.image_level_changed()

    def image_level_gmax_line_edit_changed(self):
        try:
            level_gmax = float(self.image_level_gmax_line_edit.text())
        except ValueError:
            return
        set_image_levels_gmax(self.stream_name, self.group_name, level_gmax)
        self.image_level_changed()

    def image_level_bmin_line_edit_changed(self):
        try:
            level_bmin = float(self.image_level_bmin_line_edit.text())
        except ValueError:
            return
        set_image_levels_bmin(self.stream_name, self.group_name, level_bmin)
        self.image_level_changed()

    def image_level_bmax_line_edit_changed(self):
        try:
            level_bmax = float(self.image_level_bmax_line_edit.text())
        except ValueError:
            return
        set_image_levels_bmax(self.stream_name, self.group_name, level_bmax)
        self.image_level_changed()

    def image_level_changed(self):
        levels = get_valid_image_levels(self.stream_name, self.group_name)
        self.update_image_level_valid(levels)
        self.parent.get_viz_components().group_plots[self.group_name].set_image_levels(levels)

    def update_image_level_valid(self, levels):
        if levels is None and not self.image_levels_invalid:
            self.image_levels_invalid = True
            self.display_options_valid_label.setStyleSheet("color: red")
            self.display_options_valid_label.setVisible(True)
            self.display_options_valid_label.setText("Invalid image levels: min levels must be less than max levels")
        else:
            self.display_options_valid_label.setVisible(False)
            self.image_levels_invalid = False