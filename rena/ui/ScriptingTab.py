# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets, uic

from rena.ui.ScriptingWidget import ScriptingWidget


class ScriptingTab(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingTab.ui", self)
        self.parent = parent

        self.scripts = []

        self.AddScriptBtn.clicked.connect(self.add_script_clicked)

    def add_script_clicked(self):
        script_widget = ScriptingWidget(self)
        def remove_script_clicked():
            self.ScriptingWidgetScrollLayout.removeWidget(script_widget)
            script_widget.deleteLater()
        script_widget.set_remove_btn_callback(remove_script_clicked)
        self.ScriptingWidgetScrollLayout.addWidget(script_widget)

