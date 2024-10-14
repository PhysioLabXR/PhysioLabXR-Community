# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QMovie

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.utils.ui_utils import show_label_movie


class CloseDialog(QtWidgets.QDialog):
    close_success_signal = pyqtSignal()

    def __init__(self, abort_callback: callable):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()

        self.setWindowTitle("Closing...")
        self.ui = uic.loadUi(AppConfigs()._ui_CloseDialog, self)
        self.loading_label.setMovie(QMovie(AppConfigs()._icon_load_48px))
        show_label_movie(self.loading_label, True)

        # define the action to the button box's Abort button
        self.abort_callback = abort_callback
        self.buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Abort).clicked.connect(self.abort_clicked)

        self.close_success_signal.connect(self.properly_closed)

        self.show()
        self.activateWindow()

    def abort_clicked(self):
        self.abort_callback()
        self.close()

    # close event has the same effect as abort
    def closeEvent(self, event):
        self.abort_clicked()
        event.accept()

    def properly_closed(self):
        self.close()