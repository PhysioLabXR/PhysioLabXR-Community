# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSettings

from PyQt5.QtWidgets import QFileDialog

from rena import config_ui, config

from rena.utils.ui_utils import stream_stylesheet, dialog_popup
import pyqtgraph as pg

class SettingsTab(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/SettingsTab.ui", self)

        self.find_theme()
        self.set_theme(config.settings.value('theme'))

        self.LightThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)
        self.DarkThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)

        # resolve save directory
        self.SelectDataDirBtn.clicked.connect(self.select_data_dir_btn_pressed)
        self.save_dir = config.DEFAULT_DATA_DIR if not os.path.isdir(config.USER_SETTINGS["USER_DATA_DIR"]) else config.USER_SETTINGS["USER_DATA_DIR"]
        self.saveRootTextEdit.setText(self.save_dir + '/')

        # resolve recording file format
        for file_format in config.FILE_FORMATS:
            self.saveFormatComboBox.addItem(file_format)
        self.find_recording_file_format()
        self.saveFormatComboBox.activated.connect(self.recording_file_format_change)

        self.resetDefaultBtn.clicked.connect(self.reset_default)

    def toggle_theme_btn_pressed(self):
        print("toggling theme")

        if config.settings.value('theme') == 'dark':
            config.settings.setValue('theme', 'light')
        else:
            config.settings.setValue('theme', 'dark')
        self.set_theme(config.settings.value('theme'))

    def set_theme(self, theme):
        if theme == 'light':
            self.LightThemeBtn.setEnabled(False)
            self.DarkThemeBtn.setEnabled(True)
            pg.setConfigOption('background', 'w')
        else:
            self.LightThemeBtn.setEnabled(True)
            self.DarkThemeBtn.setEnabled(False)
            pg.setConfigOption('background', 'k')

        url = 'ui/stylesheet/light.qss' if theme == 'light' else 'ui/stylesheet/dark.qss'
        stream_stylesheet(url)

    def select_data_dir_btn_pressed(self):
        selected_data_dir = str(QFileDialog.getExistingDirectory(self.widget_3, "Select Directory"))
        if selected_data_dir != '':
            self.save_dir = selected_data_dir

        print("Selected data dir: ", self.save_dir)
        self.saveRootTextEdit.setText(self.save_dir + '/')
        config.USER_SETTINGS["USER_DATA_DIR"] = self.save_dir
        json.dump(config.USER_SETTINGS, open(config.USER_SETTINGS_PATH, 'w'))

    def recording_file_format_change(self):
        # recording_file_formats = ["Rena Native (.dats)", "MATLAB (.m)", "Pickel (.p)", "Comma separate values (.CSV)"]
        if self.saveFormatComboBox.currentText() != "Rena Native (.dats)":
            dialog_popup('Using data format other than Rena Native will result in a conversion time after finishing a '
                         'recording', title='Info', dialog_name='file_format_info', enable_dont_show=True)
        config.settings.setValue('file_format', self.saveFormatComboBox.currentText())

    def reset_default(self):
        config.settings.clear()
        self.find_theme()
        self.set_theme(config.settings.value('theme'))

    def find_theme(self):
        if not config.settings.contains('theme'):
            config.settings.setValue('theme', config_ui.default_theme)

    def find_recording_file_format(self):
        if not config.settings.contains('file_format'):
            config.settings.setValue('file_format', config.DEFAULT_FILE_FORMAT)
        self.saveFormatComboBox.setCurrentIndex(config.FILE_FORMATS.index(config.settings.value('file_format')))
