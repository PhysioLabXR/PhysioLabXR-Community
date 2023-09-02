from PyQt6 import QtWidgets, uic

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.presets_utils import set_stream_preset_info


class CustomPropertyWidget(QtWidgets.QWidget):
    def __init__(self, parent, stream_name, property_name, property_value):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi(AppConfigs()._ui_CustomPropertyWidget, self)

        self.stream_name = stream_name

        self.set_property_label(property_name)
        self.set_property_value(property_value)

        self.PropertyLineEdit.textChanged.connect(self.update_value_in_settings)

    def set_property_label(self, label_name):
        self.PropertyLabel.setText(label_name)

    def set_property_value(self, value):
        self.PropertyLineEdit.setText(str(value))

    def update_value_in_settings(self):
        set_stream_preset_info(self.stream_name, self.PropertyLabel.text(), self.PropertyLineEdit.text())
