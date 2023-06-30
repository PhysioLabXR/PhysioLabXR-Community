import pyqtgraph as pg
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import QCompleter

from rena.presets.Presets import PresetType, DataType
from rena.ui.CustomPropertyWidget import CustomPropertyWidget
from rena.presets.presets_utils import get_preset_category, get_stream_preset_info, get_stream_preset_custom_info
from rena.ui_shared import add_icon
from rena.utils.ui_utils import add_presets_to_combobox, update_presets_to_combobox


class AddStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi("ui/AddCustomDataStreamWidget.ui", self)

        self.locateBtn.clicked.connect(self.on_locate_btn_clicked)
        self.createBtn.clicked.connect(self.on_create_btn_clicked)