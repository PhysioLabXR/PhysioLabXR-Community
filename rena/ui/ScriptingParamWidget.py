from pydoc import locate

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtWidgets import QCheckBox, QLineEdit

from rena import ui_shared
from rena.shared import ParamChange
from rena.ui_shared import minus_icon


class ScriptingParamWidget(QtWidgets.QWidget):
    def __init__(self, parent, param_name, type_text, value_text):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingParamWidget.ui", self)
        self.parent = parent

        self.label_param_name.setText(param_name)

        self.remove_btn.setIcon(minus_icon)

        self.value_widget = None
        self.on_type_combobox_changed()
        self.type_comboBox.currentIndexChanged.connect(self.on_type_combobox_changed)
        self.type_comboBox.currentIndexChanged.connect(self.on_param_changed)

        if type_text is not None and value_text is not None:
            self.set_type_and_value_from_text(type_text, value_text)

    def on_type_combobox_changed(self):
        if self.value_widget is not None:  # will be none on startup
            self.top_layout.removeWidget(self.value_widget)
        selected_type_text = self.type_comboBox.currentText()
        if selected_type_text == 'bool':
            self.value_widget = QCheckBox()
            self.value_widget.stateChanged.connect(self.on_param_changed)
        else:
            self.value_widget = QLineEdit()
            self.value_widget.textChanged.connect(self.on_param_changed)
        self.top_layout.insertWidget(1, self.value_widget)

    def set_button_callback(self, callback):
        self.remove_btn.clicked.connect(callback)

    def get_value(self):
        selected_type_text = self.type_comboBox.currentText()
        if selected_type_text == 'bool':
            return self.value_widget.isChecked()
        elif selected_type_text == 'str':
            return self.value_widget.text()
        else:  # numeric types: int, float, complex
            selected_type = locate(selected_type_text)
            try:
                return selected_type(self.value_widget.text())
            except ValueError:  # if failed to convert from string
                return selected_type(0)

    def set_type_and_value_from_text(self, type_text: str, value_text: str):
        # first process the type change
        index = self.type_comboBox.findText(type_text, QtCore.Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.type_comboBox.setCurrentIndex(index)
        else: raise NotImplementedError

        if type_text == 'bool':  # the value widget should have long been changed by this time
            self.value_widget.setChecked(value_text == 'True')
        else:
            self.value_widget.setText(value_text)

    def get_type_text(self):
        return self.type_comboBox.currentText()

    def get_value_text(self):
        selected_type_text = self.type_comboBox.currentText()
        if selected_type_text == 'bool':
            return str(self.value_widget.isChecked())
        else:
            return self.value_widget.text()

    def get_param_name(self):
        return self.label_param_name.text()

    def on_param_changed(self):
        self.parent.param_change(ParamChange.CHANGE, self.get_param_name(), value=self.get_value())

