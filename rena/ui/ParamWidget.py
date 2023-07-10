from pydoc import locate

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QCheckBox, QLineEdit, QScrollArea, QWidget

from rena import ui_shared
from rena.presets.ScriptPresets import ParamPreset
from rena.scripting.scripting_enums import ParamChange, ParamType
from rena.ui_shared import minus_icon, add_icon
from rena.utils.ui_utils import add_enum_values_to_combobox


class ParamWidget(QtWidgets.QWidget):
    def __init__(self, scripting_widget, param_name, param_type=ParamType.bool, value_text='', is_top_level=True, top_param_widget=None):
        super().__init__()
        self.ui = uic.loadUi("ui/ParamWidget.ui", self)
        self.scripting_widget = scripting_widget
        self.scroll_area.setWidgetResizable(True)


        if not is_top_level:  # only top level param has name
            self.label_param_name.setVisible(False)
            assert top_param_widget is not None, "top_param_widget must be provided if is_top_level is False"
            self.top_param_widget = top_param_widget
        else:
            self.top_param_widget = self
            self.label_param_name.setText(param_name)
        self.remove_btn.setIcon(minus_icon)
        self.add_to_list_button.setIcon(add_icon)
        self.add_to_list_button.clicked.connect(self.add_to_list_button_pressed)

        self.value_widget = None

        add_enum_values_to_combobox(self.type_comboBox, ParamType)
        self.type_comboBox.currentIndexChanged.connect(self.on_type_combobox_changed)
        self.type_comboBox.setCurrentIndex(self.type_comboBox.findText(param_type.name, QtCore.Qt.MatchFlag.MatchFixedString))# set to default type: bool
        self.process_param_type_change(param_type)  # call on type changed to create the value widget

    def on_type_combobox_changed(self):
        selected_type = ParamType[self.type_comboBox.currentText()]
        self.process_param_type_change(selected_type)
        self.on_param_changed()

    def process_param_type_change(self, new_type: ParamType):
        if self.value_widget is not None:  # will be none on startup
            if self.value_widget != self.list_widget:
                self.top_layout.removeWidget(self.value_widget)
            else:  # if the previous selected type is list and the current is not
                self.list_widget.setVisible(False)

        if new_type is ParamType.bool:
            self.value_widget = QCheckBox()
            self.value_widget.stateChanged.connect(self.on_param_changed)

        elif new_type is ParamType.list:
            self.make_list_param()
        else:
            self.value_widget = QLineEdit()
            if new_type is ParamType.float:
                self.value_widget.setValidator(QDoubleValidator())
            elif new_type is ParamType.int:
                self.value_widget.setValidator(QIntValidator())
            self.value_widget.textChanged.connect(self.on_param_changed)

        if new_type is not ParamType.list:
            self.list_widget.setVisible(False)
            self.top_layout.insertWidget(1, self.value_widget, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    def make_list_param(self):
        self.list_widget.setVisible(True)
        self.value_widget = self.list_content_frame_widget

    def set_remove_button_callback(self, callback: callable):
        self.remove_btn.clicked.connect(callback)

    def get_value(self):
        selected_type_text = ParamType[self.type_comboBox.currentText()]
        if selected_type_text == ParamType.bool:
            return self.value_widget.isChecked()
        elif selected_type_text == ParamType.str:
            return self.value_widget.text()
        elif selected_type_text == ParamType.list:
            return [child.get_value() for child in self.value_widget.children() if isinstance(child, ParamWidget)]
        else:  # numeric types: int, float, complex
            selected_type = selected_type_text.value
            try:
                return selected_type(self.value_widget.text())
            except ValueError:  # if failed to convert from string
                return selected_type(0)

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
        self.scripting_widget.notify_script_process_param_change(ParamChange.CHANGE, self.get_param_name(), value=self.get_value())

    def get_param_preset_recursive(self):
        if self.get_param_type() == ParamType.list:
            rtn = ParamPreset(name=self.get_param_name(), type=self.get_param_type(), value=[])
            for i in range(self.value_widget.layout().count()):
                item = self.value_widget.layout().itemAt(i)
                if item.widget() is not None and item.widget().isWidgetType() and isinstance(item.widget(), ParamWidget):
                    widget = item.widget()
                    print(f"Widget found: {widget}")
                    rtn.value.append(item.widget().get_param_preset_recursive())
            return rtn
        else:
            return ParamPreset(name=self.get_param_name(), type=self.get_param_type(), value=self.get_value())

    def add_to_list_button_pressed(self):
        assert self.value_widget == self.list_content_frame_widget, "add_to_list_button_pressed should only be called when the param type is list"
        param_name = self.get_param_name()
        param_widget = ParamWidget(self.scripting_widget, param_name, is_top_level=False, top_param_widget=self.top_param_widget)
        self.value_widget.layout().insertWidget(0, param_widget)
        self.value_widget.layout().setAlignment(param_widget, QtCore.Qt.AlignmentFlag.AlignTop)
        def remove_btn_clicked():
            self.value_widget.layout().removeWidget(param_widget)
            param_widget.deleteLater()
            self.scripting_widget.export_script_args_to_settings()
            self.scripting_widget.notify_script_process_param_change(ParamChange.CHANGE, param_name, value=self.top_param_widget.get_value())
        param_widget.set_remove_button_callback(remove_btn_clicked)
        self.scripting_widget.notify_script_process_param_change(ParamChange.CHANGE, param_name, value=self.top_param_widget.get_value())
        self.scripting_widget.export_script_args_to_settings()
