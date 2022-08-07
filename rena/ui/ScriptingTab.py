# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets, uic

from rena import config
from rena.ui.ScriptingWidget import ScriptingWidget
from rena.ui_shared import add_icon


class ScriptingTab(QtWidgets.QWidget):
    """
    ScriptingTab receives data from streamwidget during the call of process_LSLStream_data
    ScriptingTab forward the data to the scriptingwidget that is actively running. ScriptingTab
    does so by calling the push_data function in scripting widget which forward the data
    through ZMQ to the scripting process

    """
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingTab.ui", self)
        self.parent = parent

        self.script_widgets = []

        self.AddScriptBtn.setIcon(add_icon)
        self.AddScriptBtn.clicked.connect(self.add_script_clicked)


    def add_script_clicked(self):
        script_widget = ScriptingWidget(self, port=config.scripting_port + 4 * len(self.script_widgets))  # reverse three ports for each scripting widget
        def remove_script_clicked():
            self.script_widgets.remove(script_widget)
            self.ScriptingWidgetScrollLayout.removeWidget(script_widget)
            script_widget.deleteLater()
        script_widget.set_remove_btn_callback(remove_script_clicked)
        self.script_widgets.append(script_widget)
        self.ScriptingWidgetScrollLayout.addWidget(script_widget)

    def forward_data(self, data_dict):
        for script_widget in self.script_widgets:
            if script_widget.is_running and data_dict['lsl_data_type'] in script_widget.get_inputs():
                script_widget.buffer_input(data_dict)