import sys
from collections import deque
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from rena.config_ui import *
from rena.utils.ui_utils import dialog_popup
from PyQt5 import QtCore, QtGui, QtWidgets


class SignalTreeViewWindow(QTreeWidget):

    selection_changed_signal = QtCore.pyqtSignal(str)
    item_changed_signal = QtCore.pyqtSignal(str)

    def __init__(self, parent, preset):
        # super(SignalTreeViewWindow, self).__init__(parent=parent)
        super().__init__()
        self.parent = parent
        self.preset = preset

        # self.model = QStandardItemModel()
        # self.model.setHorizontalHeaderLabels(['Display', 'Name'])

        # self.header().setDefaultSectionSize(180)
        self.setHeaderHidden(True)
        # self.setModel(self.model)
        self.groups_widgets = []
        self.channel_widgets = []

        self.create_tree_view()
        self.expandAll()

        # self.setSelectionMode(self.SingleSelection)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)

        # selections:
        self.selection_state = nothing_selected
        self.selected_groups = []
        self.selected_channels = []
        self.selectionModel().selectionChanged.connect(self.selection_changed)
        # self.setAcceptDrops(False)
        # self.setAcceptDrops(True)
        # self.setDropIndicatorShown(True)
        # self.setWindowFlag(Qt.ItemIsDropEnabled)

        self.itemChanged[QTreeWidgetItem, int].connect(self.item_changed)



    def create_tree_view(self):

        self.stream_root = QTreeWidgetItem(self)
        self.stream_root.setText(0, self.preset['StreamName'])
        self.stream_root.setFlags(self.stream_root.flags()
                                  & (~Qt.ItemIsDragEnabled)
                                  & (~Qt.ItemIsSelectable) | Qt.ItemIsEditable)
        # self.stream_root.channel_group.setEditable(False)
        channel_groups_information = self.preset['GroupChannelsInPlot']
        print(channel_groups_information)

        for group_name in channel_groups_information:
            group_info = channel_groups_information[group_name]
            channel_group = self.add_item(parent_item=self.stream_root,
                                          display_text=group_name,
                                          plot_format=group_info['plot_format'],
                                          item_type='group',
                                          display=group_info['group_display'])
            self.groups_widgets.append(channel_group)
            for channel_index_in_group, channel_index in enumerate(group_info['channels']):
                print(channel_index)
                channel = self.add_item(parent_item=channel_group,
                                        display_text=self.preset['ChannelNames'][channel_index],
                                        item_type='channel',
                                        display=group_info['channels_display'][channel_index_in_group],
                                        item_index=channel_index)

                channel.setFlags(channel.flags() & (~Qt.ItemIsDropEnabled))
                self.channel_widgets.append(channel)

    def startDrag(self, actions):

        self.selected_items = self.selectedItems()
        # cannot drag groups and items at the same time:
        self.moving_groups = False
        self.moving_channels = False
        for selected_item in self.selected_items:
            if selected_item.item_type == 'group':
                self.moving_groups = True
            if selected_item.item_type == 'channel':
                self.moving_channels = True

        if self.moving_groups and self.moving_channels:
            dialog_popup('Cannot move group and channels at the same time')
            self.clearSelection()
            return
        if self.moving_groups:  # is moving groups, we cannot drag one group into another
            [group_widget.setFlags(group_widget.flags() & (~Qt.ItemIsDropEnabled)) for group_widget in
             self.groups_widgets]

        if self.moving_channels:
            self.stream_root.setFlags(self.stream_root.flags() & (~Qt.ItemIsDropEnabled))

        return QTreeWidget.startDrag(self, actions)

    def dropEvent(self, event):
        drop_target = self.itemAt(event.pos())
        if drop_target == None:
            self.reset_drag_drop()
            return
        elif drop_target.data(0, 0) == self.preset['StreamName']:
            self.reset_drag_drop()
            return
        else:
            QTreeWidget.dropEvent(self, event)
            self.reset_drag_drop()

            # check empty group
            self.remove_empty_groups()
            print(drop_target.checkState(0))

    def mousePressEvent(self, *args, **kwargs):
        super(SignalTreeViewWindow, self).mousePressEvent(*args, **kwargs)
        self.reset_drag_drop()

    def reset_drag_drop(self):
        self.stream_root.setFlags(self.stream_root.flags() | Qt.ItemIsDropEnabled)
        [group_widget.setFlags(group_widget.flags() | Qt.ItemIsDropEnabled) for group_widget in
         self.groups_widgets]

        self.moving_groups = False
        self.moving_channels = False
        # self.stream_root.setCheckState(0, Qt.Checked)

    def add_item(self, parent_item, display_text, item_type, plot_format=None, display=None, item_index=None):
        item = QTreeWidgetItem(parent_item)
        item.setText(0, display_text)
        item.item_type = item_type
        item.display = display
        if plot_format is not None:
            item.plot_format = 'time_series'
        if item_index is not None:
            item.item_index = item_index
        if display is not None:
            if display == 1:
                item.setForeground(0, QBrush(QColor(color_green)))
                item.setCheckState(0, Qt.Checked)
            else:
                item.setCheckState(0, Qt.Unchecked)
        else:
            item.setCheckState(0, Qt.Unchecked)
        # item.setForeground(0, QBrush(QColor("#123456")))
        # channel.setCheckState(0, Qt.Unchecked)
        # channel_group.setText(1, group_name)
        # channel_group.setEditable(False)
        item.setFlags(
            item.flags()
            | Qt.ItemIsTristate
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEditable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsDropEnabled
        )

        return item

    def add_group(self, display_text, item_type='group', display=1, item_index=None):
        new_group = self.add_item(self.stream_root, display_text, item_type, plot_format='time_series', display=display,
                                  item_index=item_index)
        self.groups_widgets.append(new_group)
        return new_group

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
        selected_items = self.selectedItems()
        selected_item_num = len(selected_items)
        selected_groups = []
        selected_channels = []
        for selected_item in selected_items:
            if selected_item.item_type == 'group':
                selected_groups.append(selected_item)
            elif selected_item.item_type == 'channel':
                selected_channels.append(selected_item)

        self.selected_groups, self.selected_channels = selected_groups, selected_channels
        if selected_item_num == 0:
            self.selection_state = nothing_selected
            # return nothing_selected, selected_groups, selected_channels
        elif len(selected_channels) == 1 and len(selected_groups) == 0:
            self.selection_state = channel_selected
            # return channel_selected, selected_groups, selected_channels
        elif len(selected_channels) > 1 and len(selected_groups) == 0:
            self.selection_state = channels_selected
            # return channels_selected, selected_groups, selected_channels
        elif len(selected_channels) == 0 and len(selected_groups) == 1:
            self.selection_state = group_selected
            # return group_selected, selected_groups, selected_channels
        elif len(selected_channels) == 0 and len(selected_groups) > 1:
            self.selection_state = groups_selected
            # return groups_selected, selected_groups, selected_channels
        elif len(selected_channels) > 0 and len(selected_groups) > 0:
            self.selection_state = mix_selected
            # return mix_selected, selected_groups, selected_channels
        else:
            print(": ) What are you doing???")

        self.selection_changed_signal.emit("Selection Changed")

    # @QtCore.pyqtSlot()
    def item_changed(self, item, column):  # check box on change
        if item.checkState(column) == Qt.Checked or item.checkState(column) == Qt.PartiallyChecked:
            item.setForeground(0, QBrush(QColor(color_green)))
            item.display = 1
        else:
            item.setForeground(0, QBrush(QColor(color_white)))
            item.display = 0

        self.item_changed_signal.emit('check box on change')