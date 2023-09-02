from collections import defaultdict

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from physiolabxr.configs.config_ui import *
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.Presets import Presets
from physiolabxr.presets.presets_utils import get_stream_preset_info, get_stream_group_info
from physiolabxr.utils.ui_utils import dialog_popup


## Reference:
## https://stackoverflow.com/questions/13662020/how-to-implement-itemchecked-and-itemunchecked-signals-for-qtreewidget-in-pyqt4

class GroupItem(QTreeWidgetItem):
    item_type = 'group'

    def __init__(self, parent, is_shown, plot_format, stream_name, group_name, group_view):
        super().__init__(parent)
        self.parent = parent
        self.is_shown = is_shown  # show the channel plot or not
        self.plot_format = plot_format
        self.stream_name = stream_name
        self.group_name = group_name
        self.group_view = group_view
        self.setText(0, group_name)

        # self.OptionsWindowPlotFormatWidget = OptionsWindowPlotFormatWidget(self.stream_name, self.group_name)
    def setData(self, column, role, value):
        group_name_changed = False
        # check group name is being edited
        if type(value) is str and column == 0 and self.group_name != value:
            # check with StreamGroupView for duplicate group name
            if value in self.group_view.get_group_names():
                dialog_popup(f"Cannot have repeating group names for a stream: {value}", title="Warning")
                value = self.group_name  # revert to old group name
            else:
                self.group_view.change_group_name(new_group_name=value, old_group_name=self.group_name)
                self.group_name = value  # update self's group name
                group_name_changed = True

        if group_name_changed:
            self.group_view.set_enable_item_changed(False)
            check_state_before = self.checkState(column)
            super(GroupItem, self).setData(column, role, value)
            check_state_after = self.checkState(column)
            self.group_view.set_enable_item_changed(True)
        else:
            check_state_before = self.checkState(column)
            super(GroupItem, self).setData(column, role, value)
            check_state_after = self.checkState(column)

        if check_state_before != check_state_after:
            if check_state_after == Qt.CheckState.Checked or check_state_after == Qt.CheckState.PartiallyChecked:
                self.display=True
                self.setForeground(0, QBrush(QColor(color_green)))
            else:
                self.display=False
                self.setForeground(0, QBrush(QColor(color_white)))

    def children(self):
        return [self.child(i) for i in range(self.childCount())]


class ChannelItem(QTreeWidgetItem):
    def __init__(self, parent, is_shown, lsl_index, channel_name, group_view):
        super().__init__(parent)
        self.is_shown = is_shown  # show the channel plot or not
        self.lsl_index = lsl_index
        self.most_recent_change = None
        self.channel_name = channel_name
        self.setText(0, channel_name)
        self.setText(1, '['+str(lsl_index)+']')
        self.previous_parent = parent
        self.group_view = group_view

    def setData(self, column, role, value):
        parent_check_state_before = self.parent().checkState(column)
        item_check_state_before = self.checkState(column)

        channel_name_changed = False
        if role == Qt.ItemDataRole.EditRole and type(value) is str and column == 0:
            # editing the name
            if value in self.group_view.get_channel_names():
                dialog_popup(f"Cannot have repeating channel names for a stream: {value}", title="Warning")
                value = self.channel_name  # revert to old group name
            else:
                self.group_view.change_channel_name(group_name=self.parent().group_name, new_channel_name=value, old_channel_name=self.channel_name, lsl_index=self.lsl_index)
                self.channel_name = value
                channel_name_changed= True

        if channel_name_changed:
            self.group_view.set_enable_item_changed(False)
            super(ChannelItem, self).setData(column, role, value)
            item_check_state_after = self.checkState(column)
            parent_check_state_after = self.parent().checkState(column)
            self.group_view.set_enable_item_changed(True)
        else:
            super(ChannelItem, self).setData(column, role, value)
            item_check_state_after = self.checkState(column)
            parent_check_state_after = self.parent().checkState(column)

        if role == Qt.ItemDataRole.CheckStateRole and item_check_state_before != item_check_state_after:
            # set text to green
            if item_check_state_after == Qt.CheckState.Checked or item_check_state_after == Qt.CheckState.PartiallyChecked:
                self.display = True
                self.setForeground(0, QBrush(QColor(color_green)))
            else:
                self.display = False
                self.setForeground(0, QBrush(QColor(color_white)))

            if parent_check_state_after != parent_check_state_before:
                if parent_check_state_after == Qt.CheckState.Checked or parent_check_state_after == Qt.CheckState.PartiallyChecked:
                    self.parent().display = True
                    self.parent().setForeground(0, QBrush(QColor(color_green)))
                else:
                    self.parent().display = False
                    self.parent().setForeground(0, QBrush(QColor(color_white)))


class StreamGroupView(QTreeWidget):
    selection_changed_signal = QtCore.pyqtSignal(str)
    update_info_box_signal = QtCore.pyqtSignal(str)

    channel_is_display_changed_signal = QtCore.pyqtSignal(tuple)

    def __init__(self, parent_stream_options, stream_widget, format_widget, data_processing_widget, stream_name):
        super().__init__()
        self.parent = parent_stream_options
        self.stream_widget = stream_widget
        self.format_widget = format_widget
        self.data_processing_widget = data_processing_widget
        self.stream_name = stream_name

        self.setHeaderLabels(["Group/Channel", "Data Frame Index"])

        self.group_widgets = {}  # group name: str -> group item: GroupItem)
        self.channel_widgets = []
        self.create_tree_view()

        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

        # selections:
        self.selection_state = nothing_selected
        self.selected_groups = []
        self.selected_channels = []

        # self.resize(500, 200)

        # helper fieds
        self.dragged = None

        self.resizeColumnToContents(0)

        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
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
        self.group_widgets = dict()
        self.clear()

    def create_tree_view(self, image_ir_only=False):
        # self.stream_root = QTreeWidgetItem(self)
        # self.stream_root.item_type = 'stream_root'
        # self.stream_root.setText(0, self.stream_name)
        # self.stream_root.setFlags(self.stream_root.flags()
        #                           & (~Qt.ItemIsDragEnabled)
        #                           & (~Qt.ItemIsSelectable))
        # self.stream_root.channel_group.setEditable(False)

        # get_childGroups_for_group('presets/')
        group_info = get_stream_group_info(self.stream_name)
        channel_names = get_stream_preset_info(self.stream_name, key='channel_names')
        for group_name, group_entry in group_info.items():

            group = self.add_existing_group_item(parent_item=self,
                                                 group_name=group_name,
                                                 plot_format=group_entry.selected_plot_format,
                                                 is_shown=group_entry.is_group_shown)

            if group_entry.channel_indices is not None:  # only display channels if there are not too many of them

                if len(group_entry.channel_indices) > AppConfigs().max_timeseries_num_channels_per_group:
                    dialog_popup(f'Warning: Number of Channels for stream {self.stream_name}\' group {group_name} Exceeds Maximum Number of Channels Allowed. Additional Channels Will Not Be Displayed.', mode='modeless')
                    continue  # skip adding channel items if exceeding maximum time series number of channels

                for channel_index_in_group, channel_index in enumerate(group_entry.channel_indices):
                    channel = self.add_channel_item(parent_item=group,
                                                    channel_name= channel_names[int(channel_index)],
                                                    is_shown=group_entry.is_channels_shown[channel_index_in_group],
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
            [group_widget.setFlags(group_widget.flags() & (~Qt.ItemFlag.ItemIsDropEnabled)) for group_widget in
             self.group_widgets.values()]
        # if self.selection_state==channel_selected or self.selection_state==channels_selected:
        #     self.stream_root.setFlags(self.stream_root.flags() & (~Qt.ItemIsDropEnabled))
        # #
        # self.clearSelection()

        self.disconnect_selection_changed()
        QTreeWidget.startDrag(self, actions)
        # self.clearSelection()
        self.reconnect_selection_changed()
        self.dragged = self.selectedItems()

    def dropEvent(self, event):
        """
        Handles the following events:
        1. reordering groups by dragging group(s)
        Constraints:
        1. channel and group cannot be moved at the same time, this is handled in the startDrag function.
        Thus, dropEvent only handles the case of moving groups or channels
        2. cannot drop a group into a channel, this is handled by disabling the drop event for the channel at the startDrag
        event of group(s)
        @param event: the qt event of the drop action
        @return: Given by the overridden function
        """
        pos = event.position()
        drop_target = self.itemAt(int(pos.x()), int(pos.y()))
        if drop_target == None:
            self.reenable_dropdrag_for_root_and_group_items()
            return
        # if drop_target.parent() is None:
        #     self.reenable_dropdrag_for_root_and_group_items()
        #     dialog_popup(f"Cannot put a channel at root (group level)", title="Warning")
        #     return
        else:
            QTreeWidget.dropEvent(self, event)
            self.reenable_dropdrag_for_root_and_group_items()
            self.remove_empty_groups()  # check empty group
            # print(drop_target.checkState(0))

        # group and channels cannot be moved at the same time

        if len(self.selected_channels) > 0 and len(self.selected_groups) == 0:
            change_dict = {}  # group name -> channel name, lsl indices

            if self.selected_channels[0].parent() is None:  # if the drop target is the root, create a new group
                new_group_name, new_group_item = self.add_new_group()
                for c in self.selected_channels:  # put the selected channels into the new group
                    index = self.invisibleRootItem().indexOfChild(c)
                    channel_without_parent = self.takeTopLevelItem(index)
                    new_group_item.addChild(channel_without_parent)
                change_dict[new_group_name] = new_group_item.children()  # get the indices of the changed group
                target_group_item = new_group_item
            else:
                target_group_item = drop_target.parent() if type(drop_target) is ChannelItem else drop_target
                change_dict[target_group_item.group_name] = target_group_item.children()  # get the indices of the changed group

            for selected_c in self.selected_channels:
                if selected_c not in change_dict[target_group_item.group_name]:
                    change_dict[target_group_item.group_name].append(selected_c)
                selected_c_previous_group = selected_c.previous_parent
                if selected_c_previous_group.group_name not in change_dict.keys():  # add the other affected groups (the selected channels' previous groups/parents)
                    change_dict[selected_c_previous_group.group_name] = selected_c_previous_group.children()  # get the indices of the changed group
                selected_c.previous_parent = target_group_item  # set the parent to be the drop target

            print('StreamGroupView: Changed groups: {}'.format(change_dict))
            self.parent.channel_parent_group_changed(change_dict)  # notify streamOptionsWindow of the change
        elif len(self.selected_groups) > 0 and len(self.selected_channels) == 0:
            new_group_order = []  # group name -> this group's new index
            super().dropEvent(event)  # accept the drop event
            for i in range(self.topLevelItemCount()):
                item = self.topLevelItem(i)
                new_group_order.append(item.text(0))
            self.parent.group_order_changed(new_group_order)
        else:
            raise ValueError('StreamGroupView: dropEvent: Cannot move groups and channels at the same time')
        event.accept()

    def get_selected_channel_groups(self):
        rtn = defaultdict(list)  # group name -> channel name, lsl indices
        for selected_c in self.selected_channels:
            this_changed_group = selected_c.parent()
            rtn[this_changed_group.group_name].append(selected_c) # get the indices of the changed group
        return rtn

    def get_group_item(self, group_name):
        return self.group_widgets[group_name]

    def reenable_dropdrag_for_root_and_group_items(self):
        # self.stream_root.setFlags(self.stream_root.flags() | Qt.ItemIsDropEnabled)
        [group_widget.setFlags(group_widget.flags() | Qt.ItemFlag.ItemIsDropEnabled) for group_widget in
         self.group_widgets.values()]

    def add_channel_item(self, parent_item, channel_name, is_shown, lsl_index):
        item = ChannelItem(parent=parent_item, is_shown=is_shown, lsl_index=lsl_index, channel_name=channel_name, group_view=self)
        # item.setText(0, channel_name)
        if is_shown == 1:
            item.setForeground(0, QBrush(QColor(color_green)))
            item.setCheckState(0, Qt.CheckState.Checked)
        else:
            item.setCheckState(0, Qt.CheckState.Unchecked)

        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsAutoTristate
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsDragEnabled
        )

        item.setFlags(item.flags() & (~Qt.ItemFlag.ItemIsDropEnabled))

        self.channel_widgets.append(item)
        return item

    def add_new_group_item(self, parent_item, group_name):
        return self.add_group_item(parent_item, group_name, new_group_default_plot_format, True)

    def add_existing_group_item(self, parent_item, group_name, plot_format, is_shown):
        try:
            assert group_name not in self.group_widgets.keys()
        except AssertionError:
            raise Exception(f"There can't be duplicate group names {group_name}")
        return self.add_group_item(parent_item, group_name, plot_format, is_shown)

    def add_group_item(self, parent_item, group_name, plot_format, is_shown, is_expanded=True):
        item = GroupItem(parent=parent_item, is_shown=is_shown, plot_format=plot_format, stream_name=self.stream_name,
                         group_name=group_name, group_view=self)
        if is_shown:
            item.setForeground(0, QBrush(QColor(color_green)))
            item.setCheckState(0, Qt.CheckState.Checked)
        else:
            item.setCheckState(0, Qt.CheckState.Unchecked)
        item.setExpanded(is_expanded)

        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsAutoTristate
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        self.group_widgets[group_name] = item

        return item

    def get_group_names(self):
        group_names = []
        for index in range(self.topLevelItemCount()):
            group_names.append(self.topLevelItem(index).data(0, 0))
        return group_names

    def get_all_child(self, item):
        child_count = item.childCount()
        children = []
        for child_index in range(child_count):
            children.append(item.child(child_index))
        return children

    def remove_empty_groups(self):
        children_num = self.topLevelItemCount()
        empty_groups = []
        for child_index in range(children_num):
            tree_item = self.topLevelItem(child_index)
            if isinstance(tree_item, GroupItem) and tree_item.childCount() == 0:
                empty_groups.append(tree_item.group_name)
        for empty_group in empty_groups:
            removed_group_widget = self.group_widgets.pop(empty_group)
            self.takeTopLevelItem(self.indexOfTopLevelItem((removed_group_widget)))

    # def change_parent(self, item, new_parent):
    #     old_parent = item.parent
    #     ix = old_parent.indexOfChild(item)
    #     item_without_parent = old_parent.takeChild(ix)
    #     new_parent.addChild(item_without_parent)

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
            print(f"Unrecognized selections: selected channels = {selected_channels}. selected groups = {selected_groups}."
                  f"Please report this bug to the developer")

        self.selection_changed_signal.emit("Selection Changed")
        print("Selection Changed")

    def item_changed(self, item, column):  # check box on change
        if type(item) == GroupItem:
            self.update_info_box_signal.emit('Item changed')
        if type(item) == ChannelItem:
            checked = item.checkState(column) == QtCore.Qt.CheckState.Checked
            parent_group = item.parent().data(0, 0)
            self.channel_is_display_changed_signal.emit((int(item.lsl_index), parent_group, checked))

    def disconnect_selection_changed(self):
        self.selectionModel().selectionChanged.disconnect(self.selection_changed)

    def reconnect_selection_changed(self):
        self.selectionModel().selectionChanged.connect(self.selection_changed)
        self.selection_changed()

    def disable_channels_in_group(self, group_item):
        # group_item is not dropable
        group_item.setFlags(
            group_item.flags()
            & (~Qt.ItemFlag.ItemIsDropEnabled)
        )

        for i in range(0, group_item.childCount()):
            group_item.child(i).setDisabled(True)
            # group_item.child(i).setFlags(
            #     group_item.child(i).flags()
            #     & (~Qt.ItemIsTristate)
            #     & (~Qt.ItemIsUserCheckable)
            #     & (~Qt.ItemIsEditable)
            #     & (~Qt.ItemIsDragEnabled)
            #     | (~Qt.ItemIsDropEnabled)
            # )

    def enable_channels_in_group(self, group_item):
        group_item.setFlags(
            group_item.flags()
            | Qt.ItemFlag.ItemIsDropEnabled
        )

        for i in range(0, group_item.childCount()):
            group_item.child(i).setDisabled(False)


    def add_group(self):
        self.disconnect_selection_changed()
        new_group_name, new_group_item = self.add_new_group()
        selected = self.get_selected_channel_groups()

        for old_group_name, channels in selected.items():
            old_group = self.group_widgets[old_group_name]
            for c in channels:
                index = old_group.indexOfChild(c)
                channel_item_without_parent = old_group.takeChild(index)  # this will remove the child from selected channels
                new_group_item.addChild(channel_item_without_parent)

        change_dict = {}  # group name -> channel name, lsl indices
        change_dict[new_group_name] = new_group_item.children()  # get the indices of the changed group
        for selected_c in self.selected_channels:
            if selected_c not in change_dict[new_group_name]:
                change_dict[new_group_name].append(selected_c)
            this_changed_group = selected_c.previous_parent
            if this_changed_group.group_name not in change_dict.keys():
                change_dict[this_changed_group.group_name] = this_changed_group.children()  # get the indices of the changed group
            selected_c.previous_parent = new_group_item # set the parent to be the drop target
        print('StreamGroupView: Changed groups: {}'.format(change_dict))
        self.remove_empty_groups()
        self.reconnect_selection_changed()
        return change_dict

    def add_new_group(self):
        new_group_name = Presets().stream_presets[self.stream_name].get_next_available_groupname()
        new_group_item = self.add_new_group_item(parent_item=self, group_name=new_group_name)  # default plot as time series
        return new_group_name, new_group_item

    def change_group_name(self, new_group_name, old_group_name):
        self.group_widgets[new_group_name] = self.group_widgets.pop(old_group_name)
        self.stream_widget.change_group_name(new_group_name, old_group_name)
        self.format_widget.change_group_name(new_group_name)
        self.data_processing_widget.change_group_name(new_group_name)

    def change_channel_name(self, group_name, new_channel_name, old_channel_name, lsl_index):
        self.stream_widget.change_channel_name(group_name, new_channel_name, old_channel_name, lsl_index)

    def set_enable_item_changed(self, is_enable):
        if is_enable:
            self.itemChanged[QTreeWidgetItem, int].connect(self.item_changed)
        else:
            self.itemChanged[QTreeWidgetItem, int].disconnect(self.item_changed)

    def get_channel_names(self):
        return [x.channel_name for x in self.channel_widgets]

    def select_group_item(self, group_name):
        self.setCurrentItem(self.group_widgets[group_name])