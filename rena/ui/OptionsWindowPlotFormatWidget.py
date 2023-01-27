# This Python file uses the following encoding: utf-8

from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from rena import config
from rena.config_ui import plot_format_index_dict, image_depth_dict, color_green, color_red
from rena.ui.FilterComponentButterworthBandPass import FilterComponentButterworthBandPass
from rena.ui.FilterComponentButterworthHighPass import FilterComponentButterworthHighPass
from rena.ui.FilterComponentButterworthLowPass import FilterComponentButterworthLowPass
from rena.utils.settings_utils import collect_stream_group_info, update_selected_plot_format, set_plot_image_w_h, \
    set_plot_image_format, set_plot_image_channel_format, set_plot_image_valid, set_bar_chart_max_min_range


class OptionsWindowPlotFormatWidget(QtWidgets.QWidget):
    plot_format_on_change_signal = QtCore.pyqtSignal(dict)
    preset_on_change_signal = QtCore.pyqtSignal()
    bar_chart_range_on_change_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, stream_name):
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        # self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindowPlotFormatWidget.ui", self)
        self.stream_name = stream_name
        self.group_name = None
        # self.stream_name = stream_name
        # self.grou_name = group_name
        self.plotFormatTabWidget.currentChanged.connect(self.plot_format_tab_current_changed)
        self.imageWidthLineEdit.setValidator(QIntValidator())
        self.imageHeightLineEdit.setValidator(QIntValidator())
        self.imageScalingFactorLineEdit.setValidator(QIntValidator())

        self.imageWidthLineEdit.textChanged.connect(self.image_W_H_on_change)
        self.imageHeightLineEdit.textChanged.connect(self.image_W_H_on_change)
        self.imageScalingFactorLineEdit.textChanged.connect(self.image_W_H_on_change)
        self.imageFormatComboBox.currentTextChanged.connect(self.image_format_change)
        self.imageFormatComboBox.currentTextChanged.connect(self.image_channel_format_change)

        self.barPlotYMaxLineEdit.setValidator(QDoubleValidator())
        self.barPlotYMinLineEdit.setValidator(QDoubleValidator())

        self.barPlotYMaxLineEdit.textChanged.connect(self.bar_chart_range_on_change)
        self.barPlotYMinLineEdit.textChanged.connect(self.bar_chart_range_on_change)

        # self.image_format_on_change_signal.connect(self.image_valid_update)
        # image format change

    def set_plot_format_widget_info(self, stream_name, group_name):

        self.group_name = group_name
        # which one to select
        group_info = collect_stream_group_info(stream_name, group_name)
        # change selected tab

        # disconnect while switching selected group
        self.plotFormatTabWidget.currentChanged.disconnect()
        self.plotFormatTabWidget.setCurrentIndex(group_info['selected_plot_format'])
        if collect_stream_group_info(stream_name, group_name)['is_image_only']:
            self.enable_only_image_tab()
        self.plotFormatTabWidget.currentChanged.connect(self.plot_format_tab_current_changed)

        # image format information
        self.imageWidthLineEdit.setText(str(group_info['plot_format']['image']['width']))
        self.imageHeightLineEdit.setText(str(group_info['plot_format']['image']['height']))
        self.imageScalingFactorLineEdit.setText(str(group_info['plot_format']['image']['scaling_factor']))
        self.imageFormatComboBox.setCurrentText(group_info['plot_format']['image']['image_format'])
        self.channelFormatCombobox.setCurrentText(group_info['plot_format']['image']['channel_format'])

        # bar chart format information
        self.barPlotYMaxLineEdit.setText(str(group_info['plot_format']['bar_chart']['y_max']))
        self.barPlotYMinLineEdit.setText(str(group_info['plot_format']['bar_chart']['y_min']))

    def plot_format_tab_current_changed(self, index):
        # create value
        # update the index in display
        # get current selected
        # update_selected_plot_format
        update_selected_plot_format(self.stream_name, self.group_name, index)
        # if index==2:

        # new format, old format
        info_dict = {
            'stream_name': self.stream_name,
            'group_name': self.group_name,
            'new_format': index
        }

        self.plot_format_changed(info_dict)

    def image_W_H_on_change(self):
        # check if W * H * D = Channel Num
        # W * H * D
        # update the value to settings
        width = self.get_image_width()
        height = self.get_image_height()
        scaling_factor = self.get_image_scaling_factor()
        set_plot_image_w_h(self.stream_name, self.group_name, height=height, width=width, scaling_factor=scaling_factor)

        self.image_changed()

    def image_format_change(self):
        image_format = self.get_image_format()
        set_plot_image_format(self.stream_name, self.group_name, image_format=image_format)

        self.image_changed()

    def image_channel_format_change(self):
        image_channel_format = self.get_image_channel_format()
        set_plot_image_channel_format(self.stream_name, self.group_name, channel_format=image_channel_format)

        self.image_changed()

    def image_valid_update(self):
        image_format_valid = self.image_format_valid()
        set_plot_image_valid(self.stream_name, self.group_name, image_format_valid)
        width, height, image_format, channel_format, channel_num = self.get_image_info()

        self.imageFormatInfoLabel.setText('Width x Height x Depth = {0} \n LSL Channel Number = {1}'.format(
            str(width * height * image_depth_dict[image_format]), str(channel_num)
        ))

        if image_format_valid:
            self.imageFormatInfoLabel.setStyleSheet('color: green')
            print('Valid Image Format XD')
        else:
            self.imageFormatInfoLabel.setStyleSheet('color: red')
            print('Invalid Image Format')

    def get_image_info(self):
        group_info = collect_stream_group_info(self.stream_name, self.group_name)
        width = group_info['plot_format']['image']['width']
        height = group_info['plot_format']['image']['height']
        image_format = group_info['plot_format']['image']['image_format']
        channel_format = group_info['plot_format']['image']['channel_format']
        channel_num = len(group_info['channel_indices'])
        return width, height, image_format, channel_format, channel_num

    def image_format_valid(self):
        # group_info =
        # height = self.get_image_height()
        # width = self.get_image_width()
        # image_channel_num = self.get_image_channel_num()
        width, height, image_format, channel_format, channel_num = self.get_image_info()
        if channel_num != width * height * image_depth_dict[image_format]:
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
        return current_format

    def get_image_channel_format(self):
        current_format = self.channelFormatCombobox.currentText()
        # image_channel_num = image_depth_dict(current_format)
        return current_format

    def image_changed(self):
        self.image_valid_update()
        self.preset_on_change_signal.emit()

    def plot_format_changed(self, info_dict):
        self.plot_format_on_change_signal.emit(info_dict)

    def bar_chart_range_on_change(self):
        bar_chart_max_range = self.get_bar_chart_max_range()
        bar_chart_min_range = self.get_bar_chart_min_range()

        set_bar_chart_max_min_range(self.stream_name,
                                    self.group_name,
                                    max_range=bar_chart_max_range,
                                    min_range=bar_chart_min_range)

        self.bar_chart_range_on_change_signal.emit(self.stream_name, self.group_name)

    def enable_only_image_tab(self):
        self.plotFormatTabWidget.setTabEnabled(0, False)
        self.plotFormatTabWidget.setTabEnabled(2, False)


    def add_filter_btn_clicked(self):
        """
        add inactive filer widget and RenaFilter
        """
        # group_info = collect_stream_group_info(self.stream_name, self.group_name)
        # channel_num =
        filter_type = self.filterSelectionCombobox.currentText()

        # create filter
        if filter_type == "ButterWorthBandPass":
            pass

        elif filter_type == "ButterWorthBandPass":
            filter_widget = FilterComponentButterworthBandPass()
            # self.filter_widgets.append(filter_widget)
            # the current FilterScrollAreaWidgetLayout is attached to the current group and group name
            self.FilterScrollAreaWidgetLayout.addWidget(filter_widget)

        elif filter_type == "ButterWorthHighPass":
            filter_widget = FilterComponentButterworthHighPass()
            # self.filter_widgets.append(filter_widget)
            # the current FilterScrollAreaWidgetLayout is attached to the current group and group name
            self.FilterScrollAreaWidgetLayout.addWidget(filter_widget)

        elif filter_type == "ButterWorthLowPass":
            filter_widget = FilterComponentButterworthLowPass()
            # self.filter_widgets.append(filter_widget)
            # the current FilterScrollAreaWidgetLayout is attached to the current group and group name
            self.FilterScrollAreaWidgetLayout.addWidget(filter_widget)

