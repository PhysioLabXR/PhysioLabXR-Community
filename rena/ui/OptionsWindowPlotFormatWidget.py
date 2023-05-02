# This Python file uses the following encoding: utf-8

from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from rena.presets.PlotConfig import ImageFormat, ChannelFormat
from rena.presets.presets_utils import get_stream_group_info, get_stream_a_group_info, \
    set_stream_a_group_selected_plot_format, set_stream_a_group_selected_img_config, \
    set_bar_chart_max_min_range, set_group_image_format, set_group_image_channel_format, set_group_image_valid, \
    get_group_image_config


class OptionsWindowPlotFormatWidget(QtWidgets.QWidget):
    image_change_signal = QtCore.pyqtSignal(dict)

    def __init__(self, parent, stream_widget, stream_name, plot_format_changed_signal):
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        # self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindowPlotFormatWidget.ui", self)
        self.stream_name = stream_name
        self.group_name = None
        self.parent = parent
        self.stream_widget = stream_widget
        # self.stream_name = stream_name
        # self.grou_name = group_name
        self.plotFormatTabWidget.currentChanged.connect(self.plot_format_tab_selection_changed)
        self.imageWidthLineEdit.setValidator(QIntValidator())
        self.imageHeightLineEdit.setValidator(QIntValidator())
        self.imageScalingFactorLineEdit.setValidator(QIntValidator())

        self.imageWidthLineEdit.textChanged.connect(self.image_W_H_on_change)
        self.imageHeightLineEdit.textChanged.connect(self.image_W_H_on_change)
        self.imageScalingFactorLineEdit.textChanged.connect(self.image_W_H_on_change)

        self.imageFormatComboBox.addItems([format.name for format in ImageFormat])
        self.channelFormatCombobox.addItems([format.name for format in ChannelFormat])

        self.imageFormatComboBox.currentTextChanged.connect(self.image_format_change)
        self.imageFormatComboBox.currentTextChanged.connect(self.image_channel_format_change)

        self.barPlotYMaxLineEdit.setValidator(QDoubleValidator())
        self.barPlotYMinLineEdit.setValidator(QDoubleValidator())

        self.barPlotYMaxLineEdit.textChanged.connect(self.bar_chart_range_on_change)
        self.barPlotYMinLineEdit.textChanged.connect(self.bar_chart_range_on_change)

        # self.image_format_on_change_signal.connect(self.image_valid_update)
        # image format change
        self.plot_format_changed_signal = plot_format_changed_signal

    def set_plot_format_widget_info(self, group_name):
        self.group_name = group_name
        # which one to select
        self.update_display()

    def update_display(self):
        this_group_entry = get_stream_a_group_info(self.stream_name, self.group_name)
        # disconnect while switching selected group
        self.plotFormatTabWidget.currentChanged.disconnect()
        self.plotFormatTabWidget.setCurrentIndex(this_group_entry.selected_plot_format.value)
        if this_group_entry.is_image_only():
            self.enable_only_image_tab()
        self.plotFormatTabWidget.currentChanged.connect(self.plot_format_tab_selection_changed)
        self.plot_format_changed_signal.connect(self.plot_format_changed)

        # image format information
        self.imageWidthLineEdit.setText(str(this_group_entry.plot_configs.image_config.width))
        self.imageHeightLineEdit.setText(str(this_group_entry.plot_configs.image_config.height))
        self.imageScalingFactorLineEdit.setText(str(this_group_entry.plot_configs.image_config.scaling))
        self.imageFormatComboBox.setCurrentText(this_group_entry.plot_configs.image_config.image_format.name)
        self.channelFormatCombobox.setCurrentText(this_group_entry.plot_configs.image_config.channel_format.name)

        # bar chart format information
        self.barPlotYMaxLineEdit.setText(str(this_group_entry.plot_configs.barchart_config.y_max))
        self.barPlotYMinLineEdit.setText(str(this_group_entry.plot_configs.barchart_config.y_min))

    def plot_format_tab_selection_changed(self, index):
        # create value
        # update the index in display
        # get current selected
        # update_selected_plot_format
        # if index==2:
        # update_selected_plot_format(self.stream_name, self.group_name, index)
        new_plot_format = set_stream_a_group_selected_plot_format(self.stream_name, self.group_name, index)

        # self.this_group_info['selected_plot_format'] = index

        # new format, old format
        info_dict = {
            'stream_name': self.stream_name,
            'group_name': self.group_name,
            'new_format': new_plot_format
        }
        self.plot_format_changed_signal.emit(info_dict)

    @QtCore.pyqtSlot(dict)
    def plot_format_changed(self, info_dict):
        if self.group_name == info_dict['group_name']:  # if current selected group is the plot-format-changed group
            self.update_display()

    def image_W_H_on_change(self):
        # check if W * H * D = Channel Num
        # W * H * D
        # update the value to settings
        width = self.get_image_width()
        height = self.get_image_height()
        scaling_factor = self.get_image_scaling_factor()
        set_stream_a_group_selected_img_config(self.stream_name, self.group_name, height=height, width=width, scaling=scaling_factor)

        self.image_changed()

    def image_format_change(self):
        image_format = self.get_image_format()
        set_group_image_format(self.stream_name, self.group_name, image_format=image_format)
        self.image_changed()

    def image_channel_format_change(self):
        image_channel_format = self.get_image_channel_format()
        set_group_image_channel_format(self.stream_name, self.group_name, channel_format=image_channel_format)
        self.image_changed()

    def image_valid_update(self):
        image_format_valid = self.image_format_valid()
        set_group_image_valid(self.stream_name, self.group_name, image_format_valid)
        image_config = get_group_image_config(self.stream_name, self.group_name)
        channel_num = len(get_stream_a_group_info(self.stream_name, self.group_name).channel_indices)
        width, height, image_format, channel_format = image_config.width, image_config.height, image_config.image_format, image_config.channel_format

        self.imageFormatInfoLabel.setText('Width x Height x Depth = {0} \n LSL Channel Number = {1}'.format(
            str(width * height * image_format.depth_dim()), str(channel_num)
        ))

        if image_format_valid:
            self.imageFormatInfoLabel.setStyleSheet('color: green')
            print('Valid Image Format XD')
        else:
            self.imageFormatInfoLabel.setStyleSheet('color: red')
            print('Invalid Image Format')


    def image_format_valid(self):
        image_config = get_group_image_config(self.stream_name, self.group_name)
        channel_num = len(get_stream_a_group_info(self.stream_name, self.group_name).channel_indices)
        width, height, image_format, channel_format = image_config.width, image_config.height, image_config.image_format, image_config.channel_format
        if channel_num != width * height * image_format.depth_dim():
            return 0
        else:
            return 1

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

    def get_image_scaling_factor(self):
        try:
            new_image_scaling_factor = abs(int(self.imageScalingFactorLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_image_scaling_factor

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

    def change_group_name(self, new_name):
        self.group_name = new_name