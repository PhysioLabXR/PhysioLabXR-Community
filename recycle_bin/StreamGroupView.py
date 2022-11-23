import sys
from collections import deque

import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from rena import config
from rena.config_ui import *
from rena.ui.OptionsWindowPlotFormatWidget import OptionsWindowPlotFormatWidget
from rena.ui.StreamGroupItems import ChannelItem, GroupItem
from rena.ui_shared import CHANNEL_ITEM_IS_DISPLAY_CHANGED
from rena.utils.settings_utils import get_stream_preset_info, is_group_shown
from rena.utils.ui_utils import dialog_popup
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt



class StreamGroupView(QTreeWidget):
    selection_changed_signal = QtCore.pyqtSignal(str)
    update_info_box_signal = QtCore.pyqtSignal(str)

    channel_parent_group_changed_signal = QtCore.pyqtSignal(tuple)
    channel_is_display_changed_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, parent, stream_name, group_info):
        super().__init__()
        self.parent = parent
        self.stream_name = stream_name

        self.setHeaderLabels(["Name", "LSL Index"])

        # self.setModel(self.model)
        self.groups_widgets = []
        self.channel_widgets = []
        self.stream_root = None
        self.create_tree_view(group_info)

        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        # selections:
        self.selection_state = nothing_selected
        self.selected_groups = []
        self.selected_channels = []

        # self.resize(500, 200)

        # helper fieds
        self.dragged = None

        self.resizeColumnToContents(0)

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.itemDoubleClicked.connect(self.item_double_clicked_handler)

    def item_double_clicked_handler(self, item:QTreeWidgetItem, column):
        if column == 0:
            self.editItem(item, column)
        else:
            dialog_popup('Warning: Cannot Modify LSL Index')
            # pass


    def clear_tree_view(self):
        self.selection_state = nothing_selected
        self.selected_groups = []
        self.selected_channels = []
        self.selectionModel().selectionChanged.disconnect(self.selection_changed)
        self.itemChanged[QTreeWidgetItem, int].disconnect(self.item_changed)
        self.clear()

    def create_tree_view(self, group_info):
        self.stream_root = QTreeWidgetItem(self)
        self.stream_root.item_type = 'stream_root'
        self.stream_root.setText(0, self.stream_name)
        self.stream_root.setFlags(self.stream_root.flags()
                                  & (~Qt.ItemIsDragEnabled)
                                  & (~Qt.ItemIsSelectable) & (~Qt.ItemIsEditable)) # the item should not be editable
        # self.stream_root.channel_group.setEditable(False)

        # get_childGroups_for_group('presets/')
        for group_name, group_values in group_info.items():

            group = self.add_group_item(parent_item=self.stream_root,
                                        group_name=group_name,
                                        plot_format=group_values['plot_format'])
            # self.groups_widgets.append(group)
            if len(group_values['channel_indices']) > config.settings.value("max_timeseries_num_channels"):
                continue  # skip adding channel items if exceeding maximum time series number of channels
            for channel_index_in_group, channel_index in enumerate(group_values['channel_indices']):
                channel = self.add_channel_item(parent_item=group,
                                                channel_name=
                                                get_stream_preset_info(self.stream_name, key='ChannelNames')[
                                                    int(channel_index)],
                                                is_shown=group_values['is_channels_shown'][channel_index_in_group],
                                                lsl_index=channel_index)
        self.expandAll()
        self.selectionModel().selectionChanged.connect(self.selection_changed)
        self.itemChanged[QTreeWidgetItem, int].connect(self.item_changed)

    def startDrag(self, actions):

        # self.selected_items = self.selectedItems()
        # # cannot drag groups and items at the same time:
        # self.moving_groups = False
        # self.moving_channels = False
        # for selected_item in self.selected_items:
        #     if selected_item.item_type == 'group':
        #         self.moving_groups = True
        #     if selected_item.item_type == 'channel':
        #         self.moving_channels = True
        #
        # if self.moving_groups and self.moving_channels:
        #     dialog_popup('Cannot move group and channels at the same time')
        #     self.clearSelection()
        #     return
        # if self.moving_groups:  # is moving groups, we cannot drag one group into another
        #     [group_widget.setFlags(group_widget.flags() & (~Qt.ItemIsDropEnabled)) for group_widget in
        #      self.groups_widgets]
        # if self.moving_channels:
        #     self.stream_root.setFlags(self.stream_root.flags() & (~Qt.ItemIsDropEnabled))
        #
        if self.selection_state==mix_selected:
            dialog_popup('Cannot move group and channels at the same time')
            self.clearSelection()
            return
        if self.selection_state == group_selected or self.selection_state == groups_selected:  # is moving groups, we cannot drag one group into another
            [group_widget.setFlags(group_widget.flags() & (~Qt.ItemIsDropEnabled)) for group_widget in
             self.groups_widgets]
        if self.selection_state==channel_selected or self.selection_state==channels_selected:
            self.stream_root.setFlags(self.stream_root.flags() & (~Qt.ItemIsDropEnabled))
        #
        # self.clearSelection()

        self.disconnect_selection_changed()
        QTreeWidget.startDrag(self, actions)
        # self.clearSelection()
        self.reconnect_selection_changed()
        self.dragged = self.selectedItems()


    def dropEvent(self, event):
        drop_target = self.itemAt(event.pos())
        if drop_target == None:
            self.reset_drag_drop()
            return
        elif drop_target.data(0, 0) == self.stream_name:
            self.reset_drag_drop()
            return
        else:
            QTreeWidget.dropEvent(self, event)
            self.reset_drag_drop()

            # check empty group
            self.remove_empty_groups()
            print(drop_target.checkState(0))

        if len(self.selected_channels) == 1 and type(self.selected_channels[0]) is ChannelItem:  # only one channel is being dragged
            target_parent_group = drop_target.parent().data(0, 0) if type(drop_target) is ChannelItem else drop_target.data(0, 0)
            channel_index = self.selected_channels[0].lsl_index
            self.channel_parent_group_changed_signal.emit((channel_index, target_parent_group))


    # def mousePressEvent(self, *args, **kwargs):
    #     super(StreamGroupView, self).mousePressEvent(*args, **kwargs)
    #     self.reset_drag_drop()

    def reset_drag_drop(self):
        self.stream_root.setFlags(self.stream_root.flags() | Qt.ItemIsDropEnabled)
        [group_widget.setFlags(group_widget.flags() | Qt.ItemIsDropEnabled) for group_widget in
         self.groups_widgets]

    def add_channel_item(self, parent_item, channel_name, is_shown, lsl_index):
        item = ChannelItem(parent=parent_item, is_shown=is_shown, lsl_index=lsl_index, channel_name=channel_name)
        # item.setText(0, channel_name)
        if is_shown == 1:
            item.setForeground(0, QBrush(QColor(color_green)))
            item.setCheckState(0, Qt.Checked)
        else:
            item.setCheckState(0, Qt.Unchecked)

        item.setFlags(
            item.flags()
            | Qt.ItemIsTristate
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
        )

        item.setFlags(item.flags() & (~Qt.ItemIsDropEnabled))

        self.channel_widgets.append(item)
        return item

    def add_group_item(self, parent_item, group_name, plot_format):
        is_shown = is_group_shown(group_name, self.stream_name)
        item = GroupItem(parent=parent_item, is_shown=is_shown, plot_format=plot_format, stream_name=self.stream_name, group_name=group_name)
        # item.setText(0, group_name)
        if is_shown:
            item.setForeground(0, QBrush(QColor(color_green)))
            item.setCheckState(0, Qt.Checked)
        else:
            item.setCheckState(0, Qt.Unchecked)

        item.setFlags(
            item.flags()
            | Qt.ItemIsTristate
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )

        self.groups_widgets.append(item)
        return item


    def get_group_names(self):
        group_names = []
        for index in range(0, self.stream_root.childCount()):
            group_names.append(self.stream_root.child(index).data(0, 0))
        return group_names

    def get_all_child(self, item):
        child_count = item.childCount()
        children = []
        for child_index in range(child_count):
            children.append(item.child(child_index))
        return children

    def remove_empty_groups(self):
        children_num = self.stream_root.childCount()
        empty_groups = []
        for child_index in range(0, children_num):
            group = self.stream_root.child(child_index)
            if group.childCount() == 0:
                empty_groups.append(group)
        for empty_group in empty_groups:
            self.groups_widgets.remove(empty_group)
            self.stream_root.removeChild(empty_group)

    def change_parent(self, item, new_parent):
        old_parent = item.parent()
        ix = old_parent.indexOfChild(item)
        item_without_parent = old_parent.takeChild(ix)
        new_parent.addChild(item_without_parent)

    @QtCore.pyqtSlot()
    def selection_changed(self):
        # print('TEST SELECTIONS')
        selected_items = self.selectedItems()
        selected_item_num = len(selected_items)
        selected_groups = []
        selected_channels = []
        for selected_item in selected_items:
            if type(selected_item) == GroupItem:
                selected_groups.append(selected_item)
            elif type(selected_item) == ChannelItem:
                selected_channels.append(selected_item)

        self.selected_groups, self.selected_channels = selected_groups, selected_channels
        if selected_item_num == 0:  # nothing is selected
            self.selection_state = nothing_selected
        elif len(selected_channels) == 1 and len(selected_groups) == 0:  # just one channel
            self.selection_state = channel_selected
        elif len(selected_channels) > 1 and len(selected_groups) == 0:  # multiple channels and no group
            self.selection_state = channels_selected
        elif len(selected_channels) == 0 and len(selected_groups) == 1:  # just one group
            self.selection_state = group_selected
        elif len(selected_channels) == 0 and len(selected_groups) > 1:  # multiple groups and no channel
            self.selection_state = groups_selected
        elif len(selected_channels) > 0 and len(selected_groups) > 0:  # channel(s) and group(s)
            self.selection_state = mix_selected
        else:
            print(": ) What are you doing???")

        self.selection_changed_signal.emit("Selection Changed")
        print("Selection Changed")

    # @QtCore.pyqtSlot()
    def item_changed(self, item, column):  # check box on change

        print(item.data(0,0))

        if type(item) == GroupItem:
            self.update_info_box_signal.emit('Item changed')
        if type(item) == ChannelItem:
            checked = item.checkState(column) == QtCore.Qt.Checked
            parent_group = item.parent().data(0, 0)
            self.channel_is_display_changed_signal.emit((int(item.lsl_index), parent_group, checked))

    # print(item.data(0, 0))

    def create_new_group(self, new_group_name):
        group_names = self.get_group_names()
        selected_items = self.selectedItems()
        # new_group_name = self.newGroupNameTextbox.text()

        if new_group_name:
            if len(selected_items) == 0:
                dialog_popup('please select at least one channel to create a group')
            elif new_group_name in group_names:
                dialog_popup('Cannot Have duplicated Group Names')
                return
            else:
                for selected_item in selected_items:
                    if type(selected_item) == GroupItem:
                        dialog_popup('group item cannot be selected while creating new group')
                        return
                # create new group:

                # self.disconnect_selection_changed()
                self.clearSelection()

                new_group = self.add_group_item(parent_item=self.stream_root,
                                                group_name=new_group_name,
                                                display=any([item.display for item in selected_items]),
                                                plot_format='time_series')
                for selected_item in selected_items:
                    self.change_parent(item=selected_item, new_parent=new_group)

                # self.reconnect_selection_changed()

            self.remove_empty_groups()
            self.expandAll()
        else:
            dialog_popup('please enter your group name first')
            return

    def disconnect_selection_changed(self):
        self.selectionModel().selectionChanged.disconnect(self.selection_changed)

    def reconnect_selection_changed(self):
        self.selectionModel().selectionChanged.connect(self.selection_changed)
        self.selection_changed()


    def froze_group(self,group_item):
        # group_item is not dropable
        group_item.setFlags(
            group_item.flags()
            & (~Qt.ItemIsDropEnabled)
        )



        for i in range(0, group_item.childCount()):
            group_item.child(i).setDisabled(True)

    def defroze_group(self, group_item):

        group_item.setFlags(
            group_item.flags()
            | Qt.ItemIsDropEnabled
        )

        for i in range(0, group_item.childCount()):
            group_item.child(i).setDisabled(False)

    def update_group_child_selectable(self,group_name):
        pass
        # make all child unselectable
