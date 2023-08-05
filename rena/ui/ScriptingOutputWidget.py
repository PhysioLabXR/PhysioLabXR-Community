from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QIntValidator

from rena.configs.configs import AppConfigs
from rena.utils.Validators import NoCommaIntValidator


class ScriptingOutputWidget(QtWidgets.QWidget):
    def __init__(self, parent, label_text, num_channels):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_ScriptingOutputWidget, self)
        self.parent = parent
        self.label.setText(label_text)

        self.numChan_lineEdit.setValidator(NoCommaIntValidator())
        self.numChan_lineEdit.textChanged.connect(self.on_channel_num_changed)
        self.set_output_num_channels(num_channels)
        self.button.setIcon(AppConfigs()._icon_minus)

        self.on_channel_num_changed()

    def set_button_callback(self, callback):
        self.button.clicked.connect(callback)

    def set_shape_text(self, text):
        self.shapeLabel.setText(text)

    def get_label_text(self):
        return self.label.text()

    def set_output_num_channels(self, num_channels):
        self.numChan_lineEdit.setText(str(num_channels))

    def on_channel_num_changed(self):
        self.shapeLabel.setText('[1, {0}]'.format(self.numChan_lineEdit.text()))
        self.parent.export_script_args_to_settings()  # save the new number of output channels

    def get_num_channels(self):
        return 0 if self.numChan_lineEdit.text() == '' else int(self.numChan_lineEdit.text())

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