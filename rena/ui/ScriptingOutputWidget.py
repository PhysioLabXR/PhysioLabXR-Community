from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator


class ScriptingOutputWidget(QtWidgets.QWidget):
    def __init__(self, label_text):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingOutputWidget.ui", self)
        self.label.setText(label_text)
        self.numChan_lineEdit.textChanged.connect(self.on_channel_num_changed)
        self.numChan_lineEdit.setValidator(QIntValidator())

    def set_button_callback(self, callback):
        self.button.clicked.connect(callback)

    def set_shape_text(self, text):
        self.shapeLabel.setText(text)

    def get_label_text(self):
        return self.label.text()

    def on_channel_num_changed(self):
        self.shapeLabel.setText('[1, {0}]'.format(self.numChan_lineEdit.text()))