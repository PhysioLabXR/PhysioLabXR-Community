# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QDialog, QTreeWidget

from rena import config_signal
from rena.ui.SignalTreeViewWindow import SignalTreeViewWindow
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup


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
        self.resize(600, 600)

        self.signalTreeView = SignalTreeViewWindow(parent=self, preset=self.preset)
        self.SignalTreeViewLayout.addWidget(self.signalTreeView)
        self.newGroupBtn.clicked.connect(self.newGropBtn_clicked)

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
            self.signalTreeView.remove_empty_groups()
            self.signalTreeView.expandAll()
        else:
            dialog_popup('please enter your group name first')
            return