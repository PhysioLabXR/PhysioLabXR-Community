# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QDialog, QTreeWidget, QLabel, QTreeWidgetItem

from rena import config_signal
from rena.config_ui import *
from rena.ui.SignalTreeViewWindow import SignalTreeViewWindow
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup, init_label


class OptionsWindow(QDialog):
    def __init__(self, parent, preset):
        super().__init__(parent=parent)
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """

        self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindow.ui", self)
        self.parent = parent
        self.preset = preset
        # add supported filter list
        self.resize(1000, 1000)

        self.signalTreeView = SignalTreeViewWindow(parent=self, preset=self.preset)
        self.signalTreeView.selectionModel().selectionChanged.connect(self.update_info_box)
        self.SignalTreeViewLayout.addWidget(self.signalTreeView)
        self.newGroupBtn.clicked.connect(self.newGropBtn_clicked)
        self.signalTreeView.itemChanged[QTreeWidgetItem, int].connect(self.update_info_box)



    def newGropBtn_clicked(self):
        group_names = self.signalTreeView.get_group_names()
        new_group_name = self.newGroupNameTextbox.text()
        selected_items = self.signalTreeView.selectedItems()

        if new_group_name:
            if len(selected_items) == 0:
                dialog_popup('please select at least one channel to create a group')
            elif new_group_name in group_names:
                dialog_popup('Cannot Have duplicated Group Names')
                return
            else:

                for selected_item in selected_items:
                    if selected_item.item_type == 'group':
                        dialog_popup('group item cannot be selected while creating new group')
                        return
                new_group = self.signalTreeView.add_group(new_group_name)
                for selected_item in selected_items:
                    self.signalTreeView.change_parent(item=selected_item, new_parent=new_group)
                    selected_item.setCheckState(0, Qt.Checked)
            self.signalTreeView.remove_empty_groups()
            self.signalTreeView.expandAll()
        else:
            dialog_popup('please enter your group name first')
            return

    def update_info_box(self):
        selection_state, selected_groups, selected_channels = self.signalTreeView.return_selection_state()
        self.clearLayout(self.actionsWidgetLayout)

        if selection_state == nothing_selected:
            text = 'Nothing selected'
            init_label(parent=self.actionsWidgetLayout, text=text)
        elif selection_state == channel_selected:
            text = ('Channel Name: '+selected_channels[0].data(0,0))\
                   +('\nChannel Index: '+str(selected_channels[0].item_index))\
                   +('\nChannel Display: '+ str(selected_channels[0].display))
            init_label(parent=self.actionsWidgetLayout, text=text)
        elif selection_state == mix_selected:
            text = 'Cannot select groups and channels'
            init_label(parent=self.actionsWidgetLayout, text=text)



        # selected_items = self.signalTreeView.selectedItems()
        # selected_item_num = len(selected_items)
        # selected_groups = []
        # selected_channels = []
        # for selected_item in selected_items:
        #     if selected_item.item_type == 'group':
        #         selected_groups.append(selected_item)
        #     elif selected_item.item_type == 'channel':
        #         selected_channels.append(selected_item)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())









        # tree_view_selection
        # select group: apply filter, add descriptions, change plotting format
        # select groups: apply merge group
        # select channel: add description, display channel index
        # select channels: create new groups
        # select group(s) and channel(s): display nothing

    def description(self):
        pass