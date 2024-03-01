from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import QFileDialog

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType, DataType, RPCLanguage
from physiolabxr.presets.ScriptPresets import ScriptOutput
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.ui_utils import add_enum_values_to_combobox


class RPCOutputWidget(QtWidgets.QWidget):
    def __init__(self, scripting_widget, rpc_language=RPCLanguage.Python):
        """


        @param parent:
        @param stream_name:
        @param num_channels:
        @param interface_type: default is LSL when first added using the add output button in the scripting widget.
        It won't use the default value when the output is loaded from a preset.
        @param port_number:

        """
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_RPCOutputWidget, self)
        self.parent = scripting_widget

        self.remove_button.setIcon(AppConfigs()._icon_minus)

        add_enum_values_to_combobox(self.language_combobox, RPCLanguage)
        self.language_combobox.setCurrentText(rpc_language.name)
        self.language_combobox.currentTextChanged.connect(self.on_language_changed)

    def on_language_changed(self):
        pass

    def set_remove_button_callback(self, callback):
        self.remove_button.clicked.connect(callback)

    def on_location_button_pressed(self):
        selected_data_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.location_line_edit.setText(selected_data_dir)
