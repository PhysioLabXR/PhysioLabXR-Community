# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic

from PyQt5.QtWidgets import QFileDialog

from src import config, config_ui

from src.utils.ui_utils import stream_stylesheet
import pyqtgraph as pg

class SettingsTab(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("src/ui/SettingsTab.ui", self)

        self.theme = config_ui.default_theme
        # 'light' or 'dark'
        if self.theme == 'light':
            self.LightThemeBtn.setEnabled(False)
            pg.setConfigOption('background', 'w')

        else:
            self.DarkThemeBtn.setEnabled(False)
            pg.setConfigOption('background', 'k')

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

            url = 'src/ui/stylesheet/light.qss'
            stream_stylesheet(url)
            self.theme = 'light'
        else:
            pg.setConfigOption('background', 'k')

            self.LightThemeBtn.setEnabled(True)
            self.DarkThemeBtn.setEnabled(False)
            url = 'src/ui/stylesheet/dark.qss'
            stream_stylesheet(url)
            self.theme = 'dark'

    def select_data_dir_btn_pressed(self):

        selected_data_dir = str(QFileDialog.getExistingDirectory(self.widget_3, "Select Directory"))
        if selected_data_dir != '':
            self.save_dir = selected_data_dir

        print("Selected data dir: ", self.save_dir)
        self.saveRootTextEdit.setText(self.save_dir + '/')
        config.USER_SETTINGS["USER_DATA_DIR"] = self.save_dir
        json.dump(config.USER_SETTINGS, open(config.USER_SETTINGS_PATH, 'w'))