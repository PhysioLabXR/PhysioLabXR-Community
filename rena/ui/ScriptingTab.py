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
        self.ScriptingWidgetScrollLayout.addWidget(script_widget)

    def remove_script_clicked(self):
        pass