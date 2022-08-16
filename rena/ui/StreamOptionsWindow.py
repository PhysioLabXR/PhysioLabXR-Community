# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QIntValidator
from PyQt5.QtWidgets import QDialog, QTreeWidget, QLabel, QTreeWidgetItem

from rena import config_signal, config
from rena.config_ui import *
from rena.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from rena.ui.StreamGroupView import StreamGroupView
from rena.ui_shared import CHANNEL_ITEM_IS_DISPLAY_CHANGED, CHANNEL_ITEM_GROUP_CHANGED
from rena.utils.settings_utils import is_channel_in_group, is_channel_displayed, set_channel_displayed, \
    collect_stream_all_groups_info, get_stream_preset_info
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup, init_label, init_button, init_scroll_label
from PyQt5 import QtCore, QtGui, QtWidgets


class StreamOptionsWindow(QDialog):
    def __init__(self, parent, stream_name, group_info):
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindow.ui", self)
        self.parent = parent
        # add supported filter list
        self.resize(1000, 1000)

        # self.setNominalSamplingRateBtn.clicked.connect(self.set_nominal_sampling_rate_btn)

        self.stream_name = stream_name
        self.stream_group_view = StreamGroupView(parent=self, stream_name=stream_name, group_info=group_info)

        self.SignalTreeViewLayout.addWidget(self.stream_group_view)
        # self.signalTreeView.selectionModel().selectionChanged.connect(self.update_info_box)
        self.stream_group_view.selection_changed_signal.connect(self.update_info_box)
        # self.newGroupBtn.clicked.connect(self.newGropBtn_clicked)
        # self.signalTreeView.itemChanged[QTreeWidgetItem, int].connect(self.update_info_box)
        self.stream_group_view.update_info_box_signal.connect(self.update_info_box)

        # signals for processing changes in the tree view
        self.stream_group_view.channel_parent_group_changed_signal.connect(self.channel_parent_group_changed)
        self.stream_group_view.channel_is_display_changed_signal.connect(self.channel_is_display_changed)

        # nomiaml sampling rate UI elements
        self.nominalSamplingRateIineEdit.setValidator(QIntValidator())
        self.dataDisplayDurationLineEdit.setValidator(QIntValidator())
        self.load_default_sr_and_display_duration()
        self.nominalSamplingRateIineEdit.textChanged.connect(self.update_num_points_to_display)
        self.dataDisplayDurationLineEdit.textChanged.connect(self.update_num_points_to_display)

    def update_num_points_to_display(self):
        new_sampling_rate = 0 if self.nominalSamplingRateIineEdit.text() == '' else abs(int(self.nominalSamplingRateIineEdit.text()))


    @QtCore.pyqtSlot(str)
    def update_info_box(self, info):
        self.actionsWidgetLayout.addStretch()
        selection_state, selected_groups, selected_channels = \
            self.stream_group_view.selection_state, self.stream_group_view.selected_groups, self.stream_group_view.selected_channels
        self.clearLayout(self.actionsWidgetLayout)
################################################################################
        if selection_state == nothing_selected: # nothing selected
            text = 'Nothing selected'
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)

################################################################################
        elif selection_state == channel_selected: # only one channel selected
            text = ('Channel Name: ' + selected_channels[0].data(0, 0)) \
                   + ('\nLSL Channel Index: ' + str(selected_channels[0].lsl_index)) \
                   + ('\nChannel Display: ' + str(selected_channels[0].display))
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)
            self.init_create_new_group_widget()

################################################################################
        elif selection_state == mix_selected: #  both groups and channels are selected
            text = 'Cannot select both groups and channels'
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)

################################################################################
        elif selection_state == channels_selected: # channels selected
            text = ''
            for channel in selected_channels:
                text += ('\nChannel Name: ' + channel.data(0, 0)) \
                        + ('   LSL Channel Index: ' + str(channel.lsl_index))
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)
            self.init_create_new_group_widget()

################################################################################
        elif selection_state == group_selected: # one group selected

            is_group_displaying_msg = '\nIs Plotted: {0}'.format('Yes'if selected_groups[0].display else 'No')
            text = ('Group Name: ' + selected_groups[0].data(0, 0)) \
                   + is_group_displaying_msg \
                   + ('\nChannel Count: ' + str(selected_groups[0].childCount())) \
                   # + ('\nPlot Format: ' + str(selected_groups[0].plot_format))
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)

            self.init_plot_format_widget(selected_group_name=selected_groups[0].data(0, 0))

            # init image settings
            # time series check box
            # image checkbox image size input (width , height , d) enable the checkbox
            # barchart checkbox
            # line checkbox

        elif selection_state == groups_selected: # multiple groups selected
            merge_groups_btn = init_button(parent=self.actionsWidgetLayout, label='Merge Selected Groups',
                                           function=self.merge_groups_btn_clicked)

        self.actionsWidgetLayout.addStretch()

    def reload_group_info_in_treeview(self):
        '''
        this function is called when the group info in the persistent settings
        is changed externally
        :return:
        '''
        group_info = collect_stream_all_groups_info(self.stream_name)
        self.stream_group_view.clear_tree_view()
        self.stream_group_view.create_tree_view(group_info)

    def merge_groups_btn_clicked(self):
        selection_state, selected_groups, selected_channels = \
            self.stream_group_view.selection_state, self.stream_group_view.selected_groups, self.stream_group_view.selected_channels

        root_group = selected_groups[0]
        other_groups = selected_groups[1:]
        for other_group in other_groups:
            # other_group_children = [child for child in other_group.get in range(0,)]
            other_group_children = self.stream_group_view.get_all_child(other_group)
            for other_group_child in other_group_children:
                self.stream_group_view.change_parent(other_group_child, root_group)
        self.stream_group_view.remove_empty_groups()

    def init_create_new_group_widget(self):
        container_add_group, layout_add_group = init_container(parent=self.actionsWidgetLayout,
                                                               label='New Group from Selected Channels',
                                                               vertical=False,
                                                               label_position='centertop')
        _, self.newGroupNameTextbox = init_inputBox(parent=layout_add_group,
                                                    default_input='')
        add_group_btn = init_button(parent=layout_add_group, label='Create')
        add_group_btn.clicked.connect(self.create_new_group_btn_clicked)

    def create_new_group_btn_clicked(self):
        # group_names = self.signalTreeView.get_group_names()
        # selected_items = self.signalTreeView.selectedItems()
        new_group_name = self.newGroupNameTextbox.text()

        self.stream_group_view.create_new_group(new_group_name=new_group_name)

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
        self.OptionsWindowPlotFormatWidget = OptionsWindowPlotFormatWidget(self.stream_name, selected_group_name)
        self.actionsWidgetLayout.addWidget(self.OptionsWindowPlotFormatWidget)

    def load_default_sr_and_display_duration(self):
        self.nominalSamplingRateInputbox.setText(str(get_stream_preset_info(self.stream_name, 'NominalSamplingRate')))
        self.dataDisplayDurationLineEdit.setText(str(get_stream_preset_info(self.stream_name, 'DisplayDuration')))

    def set_nominal_sampling_rate_btn(self):
        new_nominal_sampling_rate = self.nominalSamplingRateInputbox.text()
        if new_nominal_sampling_rate.isnumeric():
            new_nominal_sampling_rate = float(new_nominal_sampling_rate)
            if new_nominal_sampling_rate > 0:
                print(new_nominal_sampling_rate) # TODO: update in preset and GUI
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
        channel_index, parent_group, checked = change
        # check if changed from previous value
        if checked != is_channel_displayed(channel_index, parent_group, self.stream_name):
            set_channel_displayed(checked, channel_index, parent_group, self.stream_name)
            self.parent.update_channel_shown(channel_index, checked, parent_group)


    @QtCore.pyqtSlot(tuple)
    def channel_parent_group_changed(self, change: tuple):
        channel_index, target_parent_group = change
        if not is_channel_in_group(channel_index, target_parent_group, self.stream_name): # check against the setting, see if the target parent group is the same as the one in the settings
            # the target parent group is different from the channel's original group
            # TODO
            pass

    # def export_preset(self):
    #     stream_root = self.signalTreeView.stream_root





