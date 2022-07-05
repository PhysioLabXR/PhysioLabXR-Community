# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QDialog, QTreeWidget, QLabel, QTreeWidgetItem

from rena import config_signal
from rena.config_ui import *
from rena.ui.SignalTreeViewWindow import SignalTreeViewWindow
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup, init_label, init_button, init_scroll_label


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
        self.set_nominal_sampling_rate_textbox()
        self.signalTreeView.selectionModel().selectionChanged.connect(self.update_info_box)
        self.SignalTreeViewLayout.addWidget(self.signalTreeView)
        # self.newGroupBtn.clicked.connect(self.newGropBtn_clicked)
        self.signalTreeView.itemChanged[QTreeWidgetItem, int].connect(self.update_info_box)



    def update_info_box(self):
        self.actionsWidgetLayout.addStretch()
        selection_state, selected_groups, selected_channels = self.signalTreeView.return_selection_state()
        self.clearLayout(self.actionsWidgetLayout)

        if selection_state == nothing_selected:
            text = 'Nothing selected'
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)
        elif selection_state == channel_selected:
            text = ('Channel Name: '+selected_channels[0].data(0,0))\
                   +('\nChannel Index: '+str(selected_channels[0].item_index))\
                   +('\nChannel Display: '+ str(selected_channels[0].display))
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)
            self.init_create_new_group_widget()

        elif selection_state == mix_selected:
            text = 'Cannot select both groups and channels'
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)
        elif selection_state == channels_selected:
            text = ''
            for channel in selected_channels:
                text+= ('\nChannel Name: '+channel.data(0,0))\
                   +('   LSL Channel Index: '+str(channel.item_index))
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)

            self.init_create_new_group_widget()

        elif selection_state == group_selected:
            text = ('Group Name: '+selected_groups[0].data(0,0))\
                   +('\nGroup Display: '+ str(selected_groups[0].display))\
                   +('\nChannel Count: '+ str(selected_groups[0].childCount()))\
                   +('\nPlot Format: '+ str(selected_groups[0].plot_format))
            init_scroll_label(parent=self.actionsWidgetLayout, text=text)


        elif selection_state == groups_selected:
            merge_groups_btn = init_button(parent=self.actionsWidgetLayout, label='Merge Selected Groups', function=self.merge_groups_btn_clicked)

        self.actionsWidgetLayout.addStretch()

    def merge_groups_btn_clicked(self):
        selection_state, selected_groups, selected_channels = self.signalTreeView.return_selection_state()
        root_group = selected_groups[0]
        other_groups = selected_groups[1:]
        for other_group in other_groups:
            # other_group_children = [child for child in other_group.get in range(0,)]
            other_group_children = self.signalTreeView.get_all_child(other_group)
            for other_group_child in other_group_children:
                self.signalTreeView.change_parent(other_group_child, root_group)
        self.signalTreeView.remove_empty_groups()


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

    def set_nominal_sampling_rate_textbox(self):
        self.nominalSamplingRateInputbox.setText(str(self.preset['NominalSamplingRate']))


    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

