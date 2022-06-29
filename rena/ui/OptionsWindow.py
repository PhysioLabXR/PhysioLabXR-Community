# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QDialog, QTreeWidget

from rena import config_signal
from rena.ui.SignalTreeViewWindow import SignalTreeViewWindow
from rena.utils.ui_utils import init_container, init_inputBox


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
        self.verticalLayout_5.addWidget(self.signalTreeView)
        # self.signalTreeView = self.signalTreeView
    #
    # def create_signal_tree_view(self):
    #     self.model = QStandardItemModel()
    #     self.model.setHorizontalHeaderLabels(['Name', 'Height', 'Weight'])
    #     self.signalTreeView.setDefaultSectionSize(180)
    #     self.signalTreeView(self.model)