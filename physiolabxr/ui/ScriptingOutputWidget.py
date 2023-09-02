from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QIntValidator

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType, DataType
from physiolabxr.presets.ScriptPresets import ScriptOutput
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.ui_utils import add_enum_values_to_combobox


class ScriptingOutputWidget(QtWidgets.QWidget):
    def __init__(self, parent, stream_name, num_channels, port_number, data_type=DataType.float32, interface_type=PresetType.LSL):
        """


        @param parent:
        @param stream_name:
        @param num_channels:
        @param interface_type: default is LSL when first added using the add output button in the scripting widget.
        It won't use the default value when the output is loaded from a preset.
        @param port_number:

        """
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_ScriptingOutputWidget, self)
        self.parent = parent
        self.label.setText(stream_name)

        self.numChan_lineEdit.setValidator(NoCommaIntValidator())
        self.numChan_lineEdit.textChanged.connect(self.on_channel_num_changed)
        self.set_output_num_channels(num_channels)

        self.port_lineEdit.setValidator(NoCommaIntValidator())
        self.port_lineEdit.setText(str(port_number))

        add_enum_values_to_combobox(self.data_type_comboBox, DataType)
        self.data_type_comboBox.setCurrentText(data_type.name)
        self.data_type_comboBox.currentTextChanged.connect(self.on_data_type_changed)

        self.interface_type_comboBox.addItems([PresetType.LSL.name, PresetType.ZMQ.name])
        self.interface_type_comboBox.currentTextChanged.connect(self.on_interface_type_changed)
        self.interface_type_comboBox.setCurrentText(interface_type.name)

        self.button.setIcon(AppConfigs()._icon_minus)
        self.on_channel_num_changed(export_to_settings=False)
        self.on_interface_type_changed(export_to_settings=False)
        # no addition _ui change is needed for on_data_type_changed, so no need to call it

    def set_button_callback(self, callback):
        self.button.clicked.connect(callback)

    def set_shape_text(self, text):
        self.shapeLabel.setText(text)

    def get_label_text(self):
        return self.label.text()

    def get_num_channels(self):
        return 0 if self.numChan_lineEdit.text() == '' else int(self.numChan_lineEdit.text())

    def get_port_number(self):
        return 0 if self.port_lineEdit.text() == '' else int(self.port_lineEdit.text())

    def get_data_type(self):
        return DataType[self.data_type_comboBox.currentText()]

    def get_interface_type(self):
        return PresetType[self.interface_type_comboBox.currentText()]

    def set_output_num_channels(self, num_channels):
        self.numChan_lineEdit.setText(str(num_channels))

    def on_channel_num_changed(self, export_to_settings=True):
        self.shapeLabel.setText('[1, {0}]'.format(self.numChan_lineEdit.text()))
        if export_to_settings: self.parent.export_script_args_to_settings()  # save the new number of output channels

    def on_interface_type_changed(self, export_to_settings=True):
        if self.interface_type_comboBox.currentText() == PresetType.LSL.name:
            self.port_lineEdit.setVisible(False)
        elif self.interface_type_comboBox.currentText() == PresetType.ZMQ.name:
            self.port_lineEdit.setVisible(True)
        else:
            raise ValueError(f'Unknown interface type for output widget {self.interface_type_comboBox.currentText()}')
        if export_to_settings: self.parent.export_script_args_to_settings()

    def on_data_type_changed(self):
        self.parent.export_script_args_to_settings()

    def get_output_preset(self):
        return ScriptOutput(stream_name=self.get_label_text(),
                            num_channels=self.get_num_channels(),
                            interface_type=self.get_interface_type(),
                            port_number=self.get_port_number(),
                            data_type=self.get_data_type())
#         # give the dialog warning for data type
#         try:
#             lsl_data_type = data_type.get_lsl_type()
#         except ValueError as e:
#             dialog_popup(msg=
#                          f"""
# {e},
# LSL supports {[dtype.name for dtype in DataType.get_lsl_supported_types()]}
# LSL outlet will use default float32. Unexpected casting may occur if your outlet data type is not float32.
#             """, title='WARNING')
#             lsl_data_type = DataType.float32.get_lsl_type()