# This Python file uses the following encoding: utf-8
from PyQt6 import QtCore
from PyQt6 import uic
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget

from physiolabxr.configs.config_ui import *
from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.GroupEntry import PlotFormat
from physiolabxr.presets.presets_utils import get_stream_preset_info, set_stream_preset_info, is_group_image_only
from physiolabxr.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from physiolabxr.ui.StreamGroupView import StreamGroupView
from physiolabxr.ui.dsp_ui.OptionsWindowDataProcessingWidget import OptionsWindowDataProcessingWidget
from physiolabxr.ui.ui_shared import num_points_shown_text
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.ui_utils import dialog_popup


class StreamOptionsWindow(QWidget):
    # plot_format_on_change_signal = QtCore.pyqtSignal(dict)
    bar_chart_range_on_change_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, parent_stream_widget, stream_name, plot_format_changed_signal):
        """
        note that this class does not keep a copy of the group_info
        @param parent_stream_widget:
        @param stream_name:
        @param group_info:
        @param plot_format_changed_signal:
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()

        self.ui = uic.loadUi(AppConfigs()._ui_StreamOptionsWindow, self)
        self.parent = parent_stream_widget
        self.has_reported_invalid_num_points = False

        self.stream_name = stream_name
        self.setWindowTitle('Options for {}'.format(self.stream_name))
        window_icon = QIcon(AppConfigs()._app_logo)
        self.setWindowIcon(window_icon)

        # plot format
        self.plot_format_widget = OptionsWindowPlotFormatWidget(self, self.parent, stream_name, plot_format_changed_signal)
        plot_format_changed_signal.connect(self.plot_format_changed)
        self.image_change_signal = self.plot_format_widget.image_change_signal
        self.plot_format_widget.hide()
        self.signalActionsSplitter.addWidget(self.plot_format_widget)
        # signalActionsSplitter
        # data processor
        self.data_processing_widget = OptionsWindowDataProcessingWidget(self, self.parent, stream_name)
        self.data_processing_widget.hide()
        self.signalActionsSplitter.addWidget(self.data_processing_widget)

        # barplot
        self.bar_chart_range_on_change_signal.connect(self.parent.bar_chart_range_on_change)

        # stream group tree view
        self.stream_group_view = StreamGroupView(parent_stream_options=self,
                                                 stream_widget=parent_stream_widget,
                                                 format_widget=self.plot_format_widget,
                                                 data_processing_widget=self.data_processing_widget,
                                                 stream_name=stream_name)

        self.SignalTreeViewLayout.addWidget(self.stream_group_view)
        self.stream_group_view.selection_changed_signal.connect(self.update_info_box)
        self.stream_group_view.update_info_box_signal.connect(self.update_info_box)
        self.stream_group_view.channel_is_display_changed_signal.connect(self.channel_is_display_changed)

        # nominal sampling rate UI elements
        self.nominalSamplingRateIineEdit.setValidator(NoCommaIntValidator())
        self.dataDisplayDurationLineEdit.setValidator(NoCommaIntValidator())
        self.load_sr_and_display_duration_from_settings_to_ui()
        self.nominalSamplingRateIineEdit.textChanged.connect(self.nominal_sampling_rate_changed)
        self.nominalSamplingRateIineEdit.textChanged.connect(self.update_num_points_to_display)
        self.dataDisplayDurationLineEdit.textChanged.connect(self.update_num_points_to_display)

        # self.add_group_btn = QPushButton()
        # self.add_group_btn.setText('Create New Group')
        self.add_group_btn.hide()
        self.add_group_btn.clicked.connect(self.add_group_clicked)
        # self.actionsWidgetLayout.addWidget(self.add_group_btn)

        self.update_num_points_to_display()

        self.lineedit_style_sheet = self.nominalSamplingRateIineEdit.styleSheet()
        self.label_style_sheet = self.nominalSamplingRateIineEdit.styleSheet()

        self.error_lineedit_style_sheet = self.lineedit_style_sheet + "border: 1px solid red;"
        self.error_label_style_sheet = self.label_style_sheet + "color: red;"

    def add_group_clicked(self):
        change_dict = self.stream_group_view.add_group()
        self.parent.channel_group_changed(change_dict)

    def update_num_points_to_display(self):
        num_points_to_plot, new_sampling_rate, new_display_duration = self.get_num_points_to_plot_info()
        self.numPointsShownLabel.setText(num_points_shown_text.format(int(num_points_to_plot)))

        if num_points_to_plot > AppConfigs().viz_buffer_max_size or num_points_to_plot == 0:
            if not self.has_reported_invalid_num_points:  # will only report once
                self.show_valid_num_points_to_plot(False)
                dialog_popup(f'The number of points to display is too large. Max number of points to point is {AppConfigs().viz_buffer_max_size}' if num_points_to_plot > AppConfigs().viz_buffer_max_size else 'The number of points to display must be greater than 0.'
                             'Please change the sampling rate or display duration.', mode='modal')
                self.has_reported_invalid_num_points = True
            return
        else:
            if self.has_reported_invalid_num_points:
                self.show_valid_num_points_to_plot(True)
                self.has_reported_invalid_num_points = False

        num_points_to_plot = int(num_points_to_plot)
        assert num_points_to_plot <= AppConfigs().viz_buffer_max_size
        self.update_sr_and_display_duration_in_settings(new_sampling_rate, new_display_duration)
        self.parent.on_num_points_to_display_change()

    def show_valid_num_points_to_plot(self, is_valid):
        if is_valid:
            self.numPointsShownLabel.setStyleSheet(self.label_style_sheet)
            self.nominalSamplingRateIineEdit.setStyleSheet(self.lineedit_style_sheet)
            self.dataDisplayDurationLineEdit.setStyleSheet(self.lineedit_style_sheet)
        else:
            self.numPointsShownLabel.setStyleSheet(self.error_label_style_sheet)
            self.nominalSamplingRateIineEdit.setStyleSheet(self.error_lineedit_style_sheet)
            self.dataDisplayDurationLineEdit.setStyleSheet(self.error_lineedit_style_sheet)

    def update_sr_and_display_duration_in_settings(self, new_sampling_rate, new_display_duration):
        '''
        this function is called by StreamWidget when on_num_points_to_display_change is called
        :param new_sampling_rate:
        :param new_display_duration:
        :return:
        '''
        set_stream_preset_info(self.stream_name, 'nominal_sampling_rate', new_sampling_rate)
        set_stream_preset_info(self.stream_name, 'display_duration', new_display_duration)

    def get_display_duration(self):
        try:
            new_display_duration = abs(float(self.dataDisplayDurationLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_display_duration

    def get_nominal_sampling_rate(self):
        try:
            new_sampling_rate = abs(float(self.nominalSamplingRateIineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0

        return new_sampling_rate

    def get_num_points_to_plot_info(self):
        new_sampling_rate = self.get_nominal_sampling_rate()
        new_display_duration = self.get_display_duration()
        num_points_to_plot = new_sampling_rate * new_display_duration
        return num_points_to_plot, new_sampling_rate, new_display_duration

    @QtCore.pyqtSlot(str)
    def update_info_box(self, info):
        # self.infoWidgetLayout.addStretch()
        selection_state, selected_groups, selected_channels = \
            self.stream_group_view.selection_state, self.stream_group_view.selected_groups, self.stream_group_view.selected_channels
        # self.clearLayout(self.infoWidgetLayout)
        # self.clearLayout(self.actionsWidgetLayout)
        if selection_state != group_selected:
            self.plot_format_widget.hide()
            self.data_processing_widget.hide()
        else:
            group_name = selected_groups[0].data(0, 0)
            self.plot_format_widget.show()
            self.data_processing_widget.show()

            self.plot_format_widget.set_plot_format_widget_info(group_name=group_name)
            self.data_processing_widget.set_data_processing_widget_info(group_name=group_name)

        if selection_state == channels_selected or selection_state == channel_selected:
            self.add_group_btn.show()
        else:
            self.add_group_btn.hide()


        ################################################################################
        if selection_state == nothing_selected:  # nothing selected
            pass


        ################################################################################
        elif selection_state == channel_selected:  # only one channel selected
            print('A channel are selected')


        ################################################################################
        elif selection_state == mix_selected:  # both groups and channels are selected
            pass


        ################################################################################
        elif selection_state == channels_selected:  # channels selected
            print('Channels are selected')

        ################################################################################
        elif selection_state == group_selected:  # one group selected
            pass

        elif selection_state == groups_selected:  # multiple groups selected
            pass

            # merge_groups_btn = init_button(parent=self.actionsWidgetLayout, label='Merge Selected Groups',
            #                                function=self.merge_groups_btn_clicked)

    #         self.infoWidgetLayout.addStretch()

    def reload_preset_to_UI(self):
        """
        reload the preset info to the UI
        @return:
        """
        self.reload_group_info_in_treeview()
        self.load_sr_and_display_duration_from_settings_to_ui()

    def reload_group_info_in_treeview(self):
        '''
        this function is called when the group info in the persistent settings
        is changed externally
        :return:
        '''
        self.stream_group_view.clear_tree_view()
        self.stream_group_view.create_tree_view()

    def init_plot_format_widget(self, selected_group_name):
        pass
        # self.OptionsWindowPlotFormatWidget = OptionsWindowPlotFormatWidget(self.stream_name, selected_group_name)
        # self.infoWidgetLayout.addWidget(self.OptionsWindowPlotFormatWidget)

    def load_sr_and_display_duration_from_settings_to_ui(self):
        self.nominalSamplingRateIineEdit.setText(str(get_stream_preset_info(self.stream_name, 'nominal_sampling_rate')))
        self.dataDisplayDurationLineEdit.setText(str(get_stream_preset_info(self.stream_name, 'display_duration')))

    def nominal_sampling_rate_changed(self):
        # save changed nominal sampling rate to preset if it is valid
        new_sampling_rate = self.get_nominal_sampling_rate()
        if new_sampling_rate > 0:
            set_stream_preset_info(self.stream_name, 'nominal_sampling_rate', new_sampling_rate)
            GlobalSignals().stream_preset_nominal_srate_changed.emit((self.stream_name, new_sampling_rate))
    @QtCore.pyqtSlot(tuple)
    def channel_is_display_changed(self, change: tuple):
        pass
        # channel_index, parent_group, checked = change
        # # check if changed from previous value
        # if checked != is_channel_displayed(channel_index, parent_group, self.stream_name):
        #     set_channel_displayed(checked, channel_index, parent_group, self.stream_name)
        #     self.parent.update_channel_shown(channel_index, checked, parent_group)

    def channel_parent_group_changed(self, change_dict: dict):
        self.plot_format_widget.image_valid_update()
        self.parent.channel_group_changed(change_dict)

    def group_order_changed(self, new_group_order: dict):
        self.plot_format_widget.image_valid_update()
        self.parent.group_order_changed(new_group_order)

    @QtCore.pyqtSlot(dict)
    def plot_format_changed(self, info_dict: dict):
        # get current selected:
        group_item = self.stream_group_view.get_group_item(info_dict['group_name'])
        # parent (stream widget)'s group info should have been updated by this point, because the signal to plotformat changed is connected to parent (stream widget) first

        # if new format is image, we disable all child
        if info_dict['new_format'] == PlotFormat.IMAGE or info_dict['new_format'] == PlotFormat.IMAGE:
            self.stream_group_view.disable_channels_in_group(group_item=group_item)
        else:
            self.stream_group_view.enable_channels_in_group(group_item=group_item)

    def set_spectrogram_cmap(self, group_name: str):
        if not is_group_image_only(self.stream_name, group_name):
            self.parent.set_spectrogram_cmap(group_name)

    def set_selected_group(self, group_name: str):
        self.stream_group_view.select_group_item(group_name)

    def get_viz_components(self):
        return self.parent.get_viz_components()