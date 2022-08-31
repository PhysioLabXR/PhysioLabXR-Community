# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets, uic

from rena import config
from rena.ui.ScriptingWidget import ScriptingWidget
from rena.ui_shared import add_icon
from rena.utils.settings_utils import get_script_widgets_args, remove_script_from_settings
from rena.utils.ui_utils import update_presets_to_combobox


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

        # load scripting widget from settings
        self.add_script_widgets_from_settings()

    def add_script_clicked(self):
        self.add_script_widget()

    def add_script_widget(self, args=None):
        script_widget = ScriptingWidget(self, port=config.scripting_port + 4 * len(
            self.script_widgets), args=args)  # reverse three ports for each scripting widget
        def remove_script_clicked():
            # if script_widget.try_close():
            self.script_widgets.remove(script_widget)
            self.ScriptingWidgetScrollLayout.removeWidget(script_widget)
            remove_script_from_settings(script_widget.id)
            script_widget.deleteLater()

        script_widget.set_remove_btn_callback(remove_script_clicked)
        self.script_widgets.append(script_widget)
        self.ScriptingWidgetScrollLayout.addWidget(script_widget)

    def forward_data(self, data_dict):
        for script_widget in self.script_widgets:
            if script_widget.is_running and data_dict['lsl_data_type'] in script_widget.get_inputs():
                script_widget.buffer_input(data_dict)

    def try_close(self):
        for script_widget in self.script_widgets:
            script_widget.try_close()
        return True

    def add_script_widgets_from_settings(self):
        script_widgets_args = get_script_widgets_args()
        for _, args in script_widgets_args.items():
            self.add_script_widget(args)

    def update_script_widget_input_combobox(self):
        for script_widget in self.script_widgets:
            script_widget.update_input_combobox()
