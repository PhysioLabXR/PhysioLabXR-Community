from PyQt5 import QtWidgets, uic

from rena import config
from rena.utils.settings_utils import get_all_presets
import pyqtgraph as pg


class AddStreamWidget(QtWidgets.QWidget):
    def __init__(self, ):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/AddWidget.ui", self)
        for i in get_all_presets() + config.settings.value('cameras'):
            self.stream_name_combo_box.addItem(i)

    def select_by_stream_name(self, stream_name):
        index = self.stream_name_combo_box.findText(stream_name, pg.QtCore.Qt.MatchFixedString)
        self.stream_name_combo_box.setCurrentIndex(index)

    def get_selected_stream_name(self):
        return self.stream_name_combo_box.currentText()