# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets, uic

from rena import config
from rena.configs.GlobalSignals import GlobalSignals
from rena.configs.configs import AppConfigs
from rena.presets.Presets import Presets
from rena.ui.ScriptingWidget import ScriptingWidget


class ScriptingTab(QtWidgets.QWidget):
    """
    ScriptingTab receives data from streamwidget during the call of process_LSLStream_data
    ScriptingTab forward the data to the scriptingwidget that is actively running. ScriptingTab
    does so by calling the push_data function in scripting widget which forward the data
    through ZMQ to the scripting process

    """
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_ScriptingTab, self)
        self.parent = parent

        self.script_widgets = []

        self.AddScriptBtn.setIcon(AppConfigs()._icon_add)
        self.AddScriptBtn.clicked.connect(self.add_script_clicked)

        GlobalSignals().stream_presets_entry_changed_signal.connect(self.update_script_widget_input_combobox)

        # load scripting widget from settings
        self.add_script_widgets_from_settings()

    def add_script_clicked(self):
        self.add_script_widget()

    def add_script_widget(self, script_preset=None):
        script_widget = ScriptingWidget(self, port=config.scripting_port + 4 * len(self.script_widgets), script_preset=script_preset, layout=self.ScriptingWidgetScrollLayout)  # reverse three ports for each scripting widget
        self.script_widgets.append(script_widget)
        self.ScriptingWidgetScrollLayout.addWidget(script_widget)

    def forward_data(self, data_dict):
        for script_widget in self.script_widgets:
            if script_widget.is_running and data_dict['stream_name'] in script_widget.get_inputs():
                script_widget.send_input(data_dict)

    def try_close(self):
        for script_widget in self.script_widgets:
            script_widget.try_close()
        return True

    def add_script_widgets_from_settings(self):
        for script_preset in Presets().script_presets.values():
            self.add_script_widget(script_preset)

    def update_script_widget_input_combobox(self):
        for script_widget in self.script_widgets:
            script_widget.update_input_combobox()

    def remove_script_widget(self, script_widget):
        self.script_widgets.remove(script_widget)


