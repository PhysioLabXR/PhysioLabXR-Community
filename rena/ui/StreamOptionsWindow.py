# This Python file uses the following encoding: utf-8
import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QIntValidator
from PyQt5.QtWidgets import QDialog, QTreeWidget, QLabel, QTreeWidgetItem, QPushButton

from rena import config_signal, config
from rena.config_ui import *
from rena.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from rena.ui.StreamGroupView import StreamGroupView
from rena.ui_shared import CHANNEL_ITEM_IS_DISPLAY_CHANGED, CHANNEL_ITEM_GROUP_CHANGED, num_points_shown_text
from rena.utils.settings_utils import is_channel_in_group, is_channel_displayed, set_channel_displayed, \
    collect_stream_all_groups_info, get_stream_preset_info, collect_stream_group_info
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup, init_label, init_button, init_scroll_label
from PyQt5 import QtCore, QtGui, QtWidgets


class StreamOptionsWindow(QDialog):
    # plot_format_on_change_signal = QtCore.pyqtSignal(dict)
    bar_chart_range_on_change_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, parent_stream_widget, stream_name, group_info, plot_format_changed_signal):
        """
        note that this class does not keep a copy of the group_info
        @param parent_stream_widget:
        @param stream_name:
        @param group_info:
        @param plot_format_changed_signal:
        """
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        self.ui = uic.loadUi("ui/StreamOptionsWindow.ui", self)
        self.parent = parent_stream_widget
        # add supported filter list
        # self.resize(1000, 1000)

        # self.setNominalSamplingRateBtn.clicked.connect(self.set_nominal_sampling_rate_btn)

        self.stream_name = stream_name
        self.setWindowTitle('Options for {}'.format(self.stream_name))

        # plot format
        self.plot_format_widget = OptionsWindowPlotFormatWidget(self, self.parent, stream_name, plot_format_changed_signal)
        plot_format_changed_signal.connect(self.plot_format_changed)
        self.image_change_signal = self.plot_format_widget.image_change_signal
        self.plot_format_widget.hide()
        self.actionsWidgetLayout.addWidget(self.plot_format_widget)

        # stream group tree view
        self.stream_group_view = StreamGroupView(parent_stream_options=self, stream_widget=parent_stream_widget, format_widget=self.plot_format_widget, stream_name=stream_name, group_info=group_info)
        self.SignalTreeViewLayout.addWidget(self.stream_group_view)
        self.stream_group_view.selection_changed_signal.connect(self.update_info_box)
        self.stream_group_view.update_info_box_signal.connect(self.update_info_box)
        self.stream_group_view.channel_is_display_changed_signal.connect(self.channel_is_display_changed)

        # nominal sampling rate UI elements
        self.nominalSamplingRateIineEdit.setValidator(QIntValidator())
        self.dataDisplayDurationLineEdit.setValidator(QIntValidator())
        self.load_sr_and_display_duration_from_settings_to_ui()
        self.nominalSamplingRateIineEdit.textChanged.connect(self.update_num_points_to_display)
        self.dataDisplayDurationLineEdit.textChanged.connect(self.update_num_points_to_display)

        self.add_group_btn = QPushButton()
        self.add_group_btn.setText('Create New Group')
        self.add_group_btn.hide()
        self.add_group_btn.clicked.connect(self.add_group_clicked)
        self.actionsWidgetLayout.addWidget(self.add_group_btn)

        self.update_num_points_to_display()

    def add_group_clicked(self):
        change_dict = self.stream_group_view.add_group()
        self.parent.channel_group_changed(change_dict)

    def update_num_points_to_display(self):
        num_points_to_plot, new_sampling_rate, new_display_duration = self.get_num_points_to_plot_info()
        if num_points_to_plot == 0: return
        num_points_to_plot = int(np.min([num_points_to_plot, config.settings.value('viz_data_buffer_max_size')]))
        self.numPointsShownLabel.setText(num_points_shown_text.format(num_points_to_plot))
        self.parent.on_num_points_to_display_change(num_points_to_plot, new_sampling_rate, new_display_duration)

    def get_display_duration(self):
        try:
            new_display_duration = abs(float(self.dataDisplayDurationLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_display_duration

    def get_nomimal_sampling_rate(self):
        try:
            new_sampling_rate = abs(float(self.nominalSamplingRateIineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return new_sampling_rate

    def get_num_points_to_plot_info(self):
        new_sampling_rate = self.get_nomimal_sampling_rate()
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
        else:
            group_name = selected_groups[0].data(0, 0)
            self.plot_format_widget.show()
            self.plot_format_widget.set_plot_format_widget_info(group_name=group_name, this_group_info=self.parent.group_info[group_name])

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

    def reload_preset_to_UI(self, group_info):
        self.reload_group_info_in_treeview(group_info)
        self.load_sr_and_display_duration_from_settings_to_ui()

    def reload_group_info_in_treeview(self, group_info):
        '''
        this function is called when the group info in the persistent settings
        is changed externally
        :return:
        '''
        self.stream_group_view.clear_tree_view()
        self.stream_group_view.create_tree_view(group_info)

    # def merge_groups_btn_clicked(self):
    #     selection_state, selected_groups, selected_channels = \
    #         self.stream_group_view.selection_state, self.stream_group_view.selected_groups, self.stream_group_view.selected_channels
    #
    #     root_group = selected_groups[0]
    #     other_groups = selected_groups[1:]
    #     for other_group in other_groups:
    #         # other_group_children = [child for child in other_group.get in range(0,)]
    #         other_group_children = self.stream_group_view.get_all_child(other_group)
    #         for other_group_child in other_group_children:
    #             self.stream_group_view.change_parent(other_group_child, root_group)
    #     self.stream_group_view.remove_empty_groups()

    # def init_create_new_group_widget(self):
    #     container_add_group, layout_add_group = init_container(parent=self.actionsWidgetLayout,
    #                                                            label='New Group from Selected Channels',
    #                                                            vertical=False,
    #                                                            label_position='centertop')
    #     _, self.newGroupNameTextbox = init_inputBox(parent=layout_add_group,
    #                                                 default_input='')
    #     add_group_btn = init_button(parent=layout_add_group, label='Create')
    #     add_group_btn.clicked.connect(self.create_new_group_btn_clicked)
    #
    # def create_new_group_btn_clicked(self):
    #     # group_names = self.signalTreeView.get_group_names()
    #     # selected_items = self.signalTreeView.selectedItems()
    #     new_group_name = self.newGroupNameTextbox.text()
    #
    #     self.stream_group_view.create_new_group(new_group_name=new_group_name)

        #
        # if new_group_name:
        #     if len(selected_items) == 0:
        #         dialog_popup('please select at least one channel to create a group')
        #     elif new_group_name in group_names:
        #         dialog_popup('Cannot Have duplicated Group Names')
        #         return
        #     else:
        #
        #         for selected_item in selected_items:
        #             if selected_item.item_type == 'group':
        #                 dialog_popup('group item cannot be selected while creating new group')
        #                 return
        #         new_group = self.signalTreeView.add_group(new_group_name)
        #         for selected_item in selected_items:
        #             self.signalTreeView.change_parent(item=selected_item, new_parent=new_group)
        #             selected_item.setCheckState(0, Qt.Checked)
        #     self.signalTreeView.remove_empty_groups()
        #     self.signalTreeView.expandAll()
        # else:
        #     dialog_popup('please enter your group name first')
        #     return

    def init_plot_format_widget(self, selected_group_name):
        pass
        # self.OptionsWindowPlotFormatWidget = OptionsWindowPlotFormatWidget(self.stream_name, selected_group_name)
        # self.infoWidgetLayout.addWidget(self.OptionsWindowPlotFormatWidget)

    def load_sr_and_display_duration_from_settings_to_ui(self):
        self.nominalSamplingRateIineEdit.setText(str(get_stream_preset_info(self.stream_name, 'NominalSamplingRate')))
        self.dataDisplayDurationLineEdit.setText(str(get_stream_preset_info(self.stream_name, 'DisplayDuration')))

    def set_nominal_sampling_rate_btn(self):
        new_nominal_sampling_rate = self.nominalSamplingRateIineEdit.text()
        if new_nominal_sampling_rate.isnumeric():
            new_nominal_sampling_rate = float(new_nominal_sampling_rate)
            if new_nominal_sampling_rate > 0:
                print(new_nominal_sampling_rate)  # TODO: update in preset and GUI
            else:
                dialog_popup('Please enter a valid positive number as Nominal Sampling Rate')
        else:
            dialog_popup('Please enter a valid positive number as Nominal Sampling Rate')

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    @QtCore.pyqtSlot(tuple)
    def channel_is_display_changed(self, change: tuple):
        pass
        # channel_index, parent_group, checked = change
        # # check if changed from previous value
        # if checked != is_channel_displayed(channel_index, parent_group, self.stream_name):
        #     set_channel_displayed(checked, channel_index, parent_group, self.stream_name)
        #     self.parent.update_channel_shown(channel_index, checked, parent_group)

    def channel_parent_group_changed(self, change_dict: dict):
        self.parent.channel_group_changed(change_dict)

    @QtCore.pyqtSlot(dict)
    def plot_format_changed(self, info_dict: dict):
        # get current selected:
        group_item = self.stream_group_view.get_group_item(info_dict['group_name'])
        # parent (stream widget)'s group info should have been updated by this point, because the signal to plotformat changed is connected to parent (stream widget) first

        # if new format is image, we disable all child
        if plot_format_index_dict[info_dict['new_format']] == 'image' or plot_format_index_dict[info_dict['new_format']] == 'bar_chart':
            self.stream_group_view.disable_channels_in_group(group_item=group_item)
        else:
            self.stream_group_view.enable_channels_in_group(group_item=group_item)


    # def get_group_info(self, group_name):
    #     group_info = self.parent.group_info[group_name]
    #     # parent (stream widget)'s group info should have been updated by this point, because the signal to plotformat changed is connected to parent (stream widget) first
    #     assert group_info == collect_stream_group_info(stream_name=self.stream_name, group_name=group_name)  # update the group info
    #     return group_info

