# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSettings

from PyQt5.QtWidgets import QFileDialog

from rena import config_ui, config
from rena.startup import load_default_settings
from rena.ui_shared import add_icon
from rena.utils.settings_utils import get_all_presets

from rena.utils.ui_utils import stream_stylesheet, dialog_popup, add_presets_to_combobox
import pyqtgraph as pg

class ScriptingWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingWidget.ui", self)
        self.parent = parent

        self.script = None

        # add all presents to camera
        add_presets_to_combobox(self.inputComboBox)

        self.addInputBtn.setIcon(add_icon)
        self.addInputBtn.clicked.connect(self.add_input_clicked)

    def add_input_clicked(self):
        pass
