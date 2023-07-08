from pydoc import locate

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QCheckBox, QLineEdit

from rena import ui_shared
from rena.scripting.scripting_enums import ParamChange, ParamType
from rena.ui_shared import minus_icon
from rena.utils.ui_utils import add_enum_values_to_combobox


class ParamWidget(QtWidgets.QWidget):
    def __init__(self, parent, param_name, param_type, value_text):
        super().__init__()
        self.ui = uic.loadUi("ui/ParamWidget.ui", self)
        self.parent = parent

        self.label_param_name.setText(param_name)
        self.remove_btn.setIcon(minus_icon)

        self.value_widget = None

        add_enum_values_to_combobox(self.type_comboBox, ParamType)
        self.type_comboBox.currentIndexChanged.connect(self.on_type_combobox_changed)
        self.type_comboBox.currentIndexChanged.connect(self.on_param_changed)
        self.type_comboBox.setCurrentIndex(self.type_comboBox.findText(param_type.name, QtCore.Qt.MatchFlag.MatchFixedString))# set to default type: bool
        self.on_type_combobox_changed()  # call on type changed to create the value widget

        # self.set_type_and_value_from_text(param_type, value_text)

    def on_type_combobox_changed(self):
        if self.value_widget is not None:  # will be none on startup
            self.top_layout.removeWidget(self.value_widget)
        selected_type = ParamType[self.type_comboBox.currentText()]
        if selected_type is ParamType.bool:
            self.value_widget = QCheckBox()
            self.value_widget.stateChanged.connect(self.on_param_changed)
        elif selected_type is ParamType.list:
            pass
        else:
            self.value_widget = QLineEdit()
            if selected_type is ParamType.float:
                self.value_widget.setValidator(QDoubleValidator())
            elif selected_type is ParamType.int:
                self.value_widget.setValidator(QIntValidator())

            self.value_widget.textChanged.connect(self.on_param_changed)
        self.top_layout.insertWidget(1, self.value_widget)

    def set_remove_button_callback(self, callback: callable):
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

    # def set_type_and_value_from_text(self, param_type: ParamType, value_text: str):
    #     # first process the type change
    #     index = self.type_comboBox.findText(param_type.name, QtCore.Qt.MatchFlag.MatchFixedString)
    #     if index >= 0:
    #         self.type_comboBox.setCurrentIndex(index)
    #     else: raise NotImplementedError
    #
    #     if param_type == 'bool':  # the value widget should have long been changed by this time
    #         self.value_widget.setChecked(value_text == 'True')
    #     else:
    #         self.value_widget.setText(value_text)

    def get_param_type(self):
        return ParamType[self.type_comboBox.currentText()]

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

