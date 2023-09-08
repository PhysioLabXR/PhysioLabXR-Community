# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets, uic

from physiolabxr.configs import config
from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.Presets import Presets
from physiolabxr.threadings.WaitThreads import start_wait_for_target_worker
from physiolabxr.ui.ScriptingWidget import ScriptingWidget




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

        self.script_widgets = {}  # dict of scripting widgets

        self.AddScriptBtn.setIcon(AppConfigs()._icon_add)
        self.AddScriptBtn.clicked.connect(self.add_script_clicked)

        GlobalSignals().stream_presets_entry_changed_signal.connect(self.update_script_widget_input_combobox)

        # load scripting widget from settings
        self.add_script_widgets_from_settings()

        self.wait_script_widgets_close_worker, self.wait_script_widgets_close_thread = None, None

    def add_script_clicked(self):
        self.add_script_widget()

    def add_script_widget(self, script_preset=None):
        script_widget = ScriptingWidget(self, self.parent, port=config.scripting_port + 4 * len(self.script_widgets), script_preset=script_preset, layout=self.ScriptingWidgetScrollLayout)  # reverse three ports for each scripting widget
        self.script_widgets[script_widget.id] = script_widget
        self.ScriptingWidgetScrollLayout.addWidget(script_widget)

    def forward_data(self, data_dict):
        for script_widget in self.script_widgets.values():
            if script_widget.is_running and data_dict['stream_name'] in script_widget.get_inputs():
                script_widget.send_input(data_dict)

    def try_close(self, close_finished_signal=None):
        """
        this function needs to iterate the ids because calling try_close will pop the script widget from the dict
        @return:
        """
        script_ids = list(self.script_widgets.keys())
        for script_widget_id in script_ids:
            script_widget = self.script_widgets[script_widget_id]
            script_widget.try_close()
        if close_finished_signal is not None:
            self.wait_script_widgets_close_worker, self.wait_script_widgets_close_thread \
                = start_wait_for_target_worker(self.script_widgets_empty, close_finished_signal)

    def need_to_wait_to_close(self):
        for script_widget in self.script_widgets.values():
            if script_widget.is_running:
                return True
        return False

    def kill_all_scripts(self):
        """
        this function is called by the CloseDialog when the user clicks abort
        @return:
        """
        script_ids = list(self.script_widgets.keys())
        for script_widget_id in script_ids:
            script_widget = self.script_widgets[script_widget_id]
            if script_widget.is_running:
                script_widget.kill_script_process()  # kill will cause the script widget to be popped from the dict

        # stop the wait process
        self.wait_script_widgets_close_worker.stop()
        self.wait_script_widgets_close_thread.requestInterruption()
        self.wait_script_widgets_close_thread.exit()
        self.wait_script_widgets_close_thread.wait()

    def add_script_widgets_from_settings(self):
        for script_preset in Presets().script_presets.values():
            self.add_script_widget(script_preset)

    def update_script_widget_input_combobox(self):
        for script_widget in self.script_widgets.values():
            script_widget.update_input_combobox()

    def remove_script_widget(self, script_widget):
        self.script_widgets.pop(script_widget.id)

    def script_widgets_empty(self):
        return len(self.script_widgets) == 0