from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator

from rena.ui_shared import minus_icon


class ScriptingOutputWidget(QtWidgets.QWidget):
    def __init__(self, parent, label_text, num_channels):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingOutputWidget.ui", self)
        self.parent = parent
        self.label.setText(label_text)
        self.numChan_lineEdit.textChanged.connect(self.on_channel_num_changed)
        self.numChan_lineEdit.setValidator(QIntValidator())
        self.set_output_num_channels(num_channels)
        self.button.setIcon(minus_icon)

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