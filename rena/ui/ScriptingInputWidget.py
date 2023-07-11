from PyQt6 import QtWidgets, uic

from rena import ui_shared
from rena.ui_shared import minus_icon


class ScriptingInputWidget(QtWidgets.QWidget):
    def __init__(self, input_stream_name):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingInputWidget.ui", self)

        self.label_stream_name.setText(input_stream_name)

        self.label_stream_name.setToolTip(ui_shared.scripting_input_widget_name_label_tooltip)
        self.label_input_shape.setToolTip(ui_shared.scripting_input_widget_shape_label_tooltip)
        self.remove_btn.setToolTip(ui_shared.scripting_input_widget_button_tooltip)
        self.remove_btn.setIcon(minus_icon)

    def set_button_callback(self, callback):
        self.remove_btn.clicked.connect(callback)

    def set_input_info_text(self, text):
        self.label_input_shape.setText(text)

    def get_input_info_text(self):
        return self.label_input_shape.text

    def get_input_name_text(self):
        return self.label_stream_name.text()