from PyQt5 import QtWidgets, uic

from rena import config
from rena.utils.settings_utils import get_all_preset_names
import pyqtgraph as pg

from rena.utils.ui_utils import add_presets_to_combobox


class AddStreamWidget(QtWidgets.QWidget):
    def __init__(self, ):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/AddWidget.ui", self)
        add_presets_to_combobox(self.stream_name_combo_box)


    def select_by_stream_name(self, stream_name):
        index = self.stream_name_combo_box.findText(stream_name, pg.QtCore.Qt.MatchFixedString)
        self.stream_name_combo_box.setCurrentIndex(index)

    def get_selected_stream_name(self):
        return self.stream_name_combo_box.currentText()

    def set_selection_text(self, stream_name):
        self.stream_name_combo_box.setText(stream_name)