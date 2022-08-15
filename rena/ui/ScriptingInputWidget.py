from PyQt5 import QtWidgets, uic


class ScriptingInputWidget(QtWidgets.QWidget):
    def __init__(self, label_text):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingInputWidget.ui", self)

        self.label.setText(label_text)

    def set_button_callback(self, callback):
        self.button.clicked.connect(callback)

    def set_input_info_text(self, text):
        self.label_2.setText(text)

    def get_input_info_text(self):
        return self.label_2.text

    def get_input_name_text(self):
        return self.label.text()