from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QCheckBox, QLineEdit

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.ScriptPresets import ScriptParam
from physiolabxr.scripting.scripting_enums import ParamChange, ParamType
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.ui_utils import add_enum_values_to_combobox


class ParamWidget(QtWidgets.QWidget):
    def __init__(self, scripting_widget, param_name, param_type=ParamType.bool, value=None, is_top_level=True, top_param_widget=None):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_ParamWidget, self)
        self.scripting_widget = scripting_widget
        self.scroll_area.setWidgetResizable(True)

        if not is_top_level:  # only top level param has name
            self.label_param_name.setVisible(False)
            assert top_param_widget is not None, "top_param_widget must be provided if is_top_level is False"
            self.top_param_widget = top_param_widget
        else:
            self.top_param_widget = self
            self.label_param_name.setText(param_name)
        self.remove_btn.setIcon(AppConfigs()._icon_minus)
        self.add_to_list_button.setIcon(AppConfigs()._icon_add)
        self.add_to_list_button.clicked.connect(self.add_to_list_button_pressed)
        self.expand_collapse_button.clicked.connect(self.expand_collapse_button_pressed)

        self.value_widget = None
        add_enum_values_to_combobox(self.type_comboBox, ParamType)
        self.type_comboBox.currentIndexChanged.connect(self.on_type_combobox_changed)
        self.type_comboBox.setCurrentIndex(self.type_comboBox.findText(param_type.name, QtCore.Qt.MatchFlag.MatchFixedString))# set to default type: bool
        self.process_param_type_change(param_type)  # call on type changed to create the value widget

        if value is not None:
            self.set_value_recursive(value)

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
                self.value_widget.setValidator(NoCommaIntValidator())
            self.value_widget.textChanged.connect(self.on_param_changed)

        if new_type is not ParamType.list:
            self.list_widget.setVisible(False)
            self.top_layout.insertWidget(2, self.value_widget, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
            self.expand_collapse_button.setVisible(False)

    def make_list_param(self):
        self.list_widget.setVisible(True)
        self.value_widget = self.list_content_frame_widget
        self.expand_collapse_button.setVisible(True)
        self.expand_collapse_button.setIcon(AppConfigs()._icon_collapse)

    def expand_collapse_button_pressed(self):
        if self.list_widget.isVisible():
            self.expand_collapse_button.setIcon(AppConfigs()._icon_expand)
            self.list_widget.setVisible(False)
        else:
            self.expand_collapse_button.setIcon(AppConfigs()._icon_collapse)
            self.list_widget.setVisible(True)

    def set_remove_button_callback(self, callback: callable):
        self.remove_btn.clicked.connect(callback)

    def get_value(self):
        """
        removing a param widget will call this function as it will send the param change to the Rena Script.

        """
        selected_type = self.get_param_type()
        if selected_type == ParamType.bool:
            return self.value_widget.isChecked()
        elif selected_type == ParamType.str:
            return self.value_widget.text()
        elif selected_type == ParamType.list:
            rtn = []
            for i in range(self.value_widget.layout().count()):
                item = self.value_widget.layout().itemAt(i)
                if item.widget() is not None and item.widget().isWidgetType() and isinstance(item.widget(), ParamWidget):
                    rtn.append(item.widget().get_value())
            return rtn
        else:  # numeric types: int, float, complex
            selected_type = selected_type.value
            try:
                return selected_type(self.value_widget.text())
            except ValueError:  # if failed to convert from string
                return selected_type(0)

    def set_value_recursive(self, value):
        selected_type = self.get_param_type()
        if selected_type == ParamType.bool:
            self.value_widget.setChecked(value)
        elif selected_type == ParamType.str:
            self.value_widget.setText(value)
        elif selected_type == ParamType.list:
            self.add_to_list_recursive(value=value)
        else:
            self.value_widget.setText(str(value))

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
        self.scripting_widget.notify_script_process_param_change(ParamChange.CHANGE, self.top_param_widget.get_param_name(), value=self.top_param_widget.get_value())

    def get_param_preset_recursive(self):
        if self.get_param_type() == ParamType.list:
            rtn = ScriptParam(name=self.get_param_name(), type=self.get_param_type(), value=[])
            for i in range(self.value_widget.layout().count()):
                item = self.value_widget.layout().itemAt(i)
                if item.widget() is not None and item.widget().isWidgetType() and isinstance(item.widget(), ParamWidget):
                    widget = item.widget()
                    # print(f"Widget found: {widget}")
                    rtn.value.append(item.widget().get_param_preset_recursive())
            return rtn
        else:
            return ScriptParam(name=self.get_param_name(), type=self.get_param_type(), value=self.get_value())

    def add_to_list_button_pressed(self):
        assert self.value_widget == self.list_content_frame_widget, "add_to_list_button_pressed should only be called when the param type is list"
        self.add_to_list_recursive()

    def add_to_list_recursive(self, value=None, param_type=ParamType.bool):
        param_name = self.get_param_name()
        if isinstance(value, list):
            for val in value:
                assert isinstance(val, ScriptParam), f"add_to_list_recursive should only be called with ParamPreset or list of ParamPreset, got {type(val)}"
                self.add_param_to_list(param_name, param_type=val.type, value=val.value)
        else:
            self.add_param_to_list(param_name, param_type, value)

    def add_param_to_list(self, param_name, param_type, value):
        param_widget = ParamWidget(self.scripting_widget, param_name, param_type=param_type, value=value, is_top_level=False, top_param_widget=self.top_param_widget)
        self.value_widget.layout().insertWidget(self.value_widget.layout().count()-2, param_widget)
        self.value_widget.layout().setAlignment(param_widget, QtCore.Qt.AlignmentFlag.AlignTop)
        def remove_btn_clicked():
            self.value_widget.layout().removeWidget(param_widget)
            param_widget.deleteLater()
            self.scripting_widget.export_script_args_to_settings()
            self.scripting_widget.notify_script_process_param_change(ParamChange.CHANGE, param_name, value=self.top_param_widget.get_value())
        param_widget.set_remove_button_callback(remove_btn_clicked)
        self.scripting_widget.notify_script_process_param_change(ParamChange.CHANGE, param_name, value=self.top_param_widget.get_value())
        self.scripting_widget.export_script_args_to_settings()