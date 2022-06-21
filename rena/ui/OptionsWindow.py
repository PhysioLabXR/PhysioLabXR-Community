# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtWidgets import QDialog

from rena import config_signal
from rena.utils.ui_utils import init_container, init_inputBox


class OptionsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__(parent=parent)

        self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindow.ui", self)
        self.parent = parent
        # add supported filter list
        self.resize(600, 600)



    #
    def create_signal_tree_view(self):

        pass