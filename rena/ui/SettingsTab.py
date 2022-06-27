# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSettings

from PyQt5.QtWidgets import QFileDialog

from rena import config_ui, config

from rena.utils.ui_utils import stream_stylesheet
import pyqtgraph as pg

class SettingsTab(QtWidgets.QWidget):
    def __init__(self, parent, settings):
        super().__init__()
        self.ui = uic.loadUi("ui/SettingsTab.ui", self)
        self.settings = settings
        try:
            self.theme = self.settings.value('theme')
        except AttributeError:
            self.settings.setValue('theme', config_ui.default_theme)
            self.theme = self.settings.value('theme')
        self.set_theme()

        self.LightThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)
        self.DarkThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)

        self.SelectDataDirBtn.clicked.connect(self.select_data_dir_btn_pressed)
        self.save_dir = config.DEFAULT_DATA_DIR if not os.path.isdir(config.USER_SETTINGS["USER_DATA_DIR"]) else config.USER_SETTINGS["USER_DATA_DIR"]
        self.saveRootTextEdit.setText(self.save_dir + '/')

    def toggle_theme_btn_pressed(self):
        print("toggle theme")

        if self.theme == 'dark':
            pg.setConfigOption('background', 'w')
            self.LightThemeBtn.setEnabled(False)
            self.DarkThemeBtn.setEnabled(True)
            self.theme = 'light'
            self.set_theme()
        else:
            pg.setConfigOption('background', 'k')

            self.LightThemeBtn.setEnabled(True)
            self.DarkThemeBtn.setEnabled(False)
            self.theme = 'dark'
            self.set_theme()
        self.settings.setValue('theme', self.theme)

    def set_theme(self):
        assert self.theme == 'light' or self.theme == 'dark'
        url = 'ui/stylesheet/light.qss' if self.theme == 'light' else 'ui/stylesheet/dark.qss'
        stream_stylesheet(url)

    def select_data_dir_btn_pressed(self):
        selected_data_dir = str(QFileDialog.getExistingDirectory(self.widget_3, "Select Directory"))
        if selected_data_dir != '':
            self.save_dir = selected_data_dir

        print("Selected data dir: ", self.save_dir)
        self.saveRootTextEdit.setText(self.save_dir + '/')
        config.USER_SETTINGS["USER_DATA_DIR"] = self.save_dir
        json.dump(config.USER_SETTINGS, open(config.USER_SETTINGS_PATH, 'w'))