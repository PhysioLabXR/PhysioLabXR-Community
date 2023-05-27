# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import QFileDialog

from rena import config_ui, config
from rena.startup import load_settings

from rena.utils.ui_utils import stream_stylesheet, dialog_popup
import pyqtgraph as pg

class SettingsWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.ui = uic.loadUi("ui/SettingsWidget.ui", self)
        self.parent = parent
        self.set_theme(config.settings.value('theme'))

        self.LightThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)
        self.DarkThemeBtn.clicked.connect(self.toggle_theme_btn_pressed)

        # resolve save directory
        self.SelectDataDirBtn.clicked.connect(self.select_data_dir_btn_pressed)
        self.set_recording_file_location(config.settings.value('recording_file_location'))

        # resolve recording file format
        for file_format in config.FILE_FORMATS:
            self.saveFormatComboBox.addItem(file_format)
        self.set_recording_file_format()
        self.saveFormatComboBox.activated.connect(self.recording_file_format_change)

        self.resetDefaultBtn.clicked.connect(self.reset_default)

        self.plot_fps_lineedit.textChanged.connect(self.on_plot_fps_changed)
        onlyInt = QIntValidator()
        onlyInt.setRange(*config.plot_fps_range)
        self.plot_fps_lineedit.setValidator(onlyInt)
        self.plot_fps_lineedit.setText(str(int(1e3 / int(float(config.settings.value('visualization_refresh_interval'))))))

    def switch_to_tab(self, tab_name: str):
        if 'appearance' in tab_name.lower():
            self.settings_tabs.setCurrentWidget(self.settings_appearance_tab)
        elif 'recording' in tab_name.lower():
            self.settings_tabs.setCurrentWidget(self.settings_recordings_tab)
        elif 'video device' in tab_name.lower():
            self.settings_tabs.setCurrentWidget(self.settings_video_device_tab)
        elif 'streams' in tab_name.lower():
            self.settings_tabs.setCurrentWidget(self.settings_streams_tab)
        else:
            raise ValueError(f'SettingsWidget: unknown tab name: {tab_name}')

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
        selected_data_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.set_recording_file_location(selected_data_dir)

    def recording_file_format_change(self):
        # recording_file_formats = ["Rena Native (.dats)", "MATLAB (.m)", "Pickel (.p)", "Comma separate values (.CSV)"]
        if self.saveFormatComboBox.currentText() != "Rena Native (.dats)":
            dialog_popup('Using data format other than Rena Native will result in a conversion time after finishing a '
                         'recording', title='Info', dialog_name='file_format_info', enable_dont_show=True, mode="modeless")
        config.settings.setValue('file_format', self.saveFormatComboBox.currentText())

    def reset_default(self):
        config.settings.clear()
        load_settings()

        self.set_theme(config.settings.value('theme'))
        self.set_recording_file_format()
        self.set_recording_file_location(config.DEFAULT_DATA_DIR)

    def set_recording_file_format(self):
        self.saveFormatComboBox.setCurrentIndex(config.FILE_FORMATS.index(config.settings.value('file_format')))

    def set_recording_file_location(self, selected_data_dir: str):
        if selected_data_dir != '':
            config.settings.setValue('recording_file_location', selected_data_dir)
            print("Selected data dir: ", config.settings.value('recording_file_location'))
            self.saveRootTextEdit.setText(config.settings.value('recording_file_location'))
            self.parent.recording_tab.update_ui_save_file()

    def on_plot_fps_changed(self):
        print(f"plot_fps_lineedit changed value is {self.plot_fps_lineedit.text()}")

        if self.plot_fps_lineedit.text() != '':
            new_value = int(self.plot_fps_lineedit.text())
            if new_value in range(config.plot_fps_range[0], config.plot_fps_range[1]+1):
                config.settings.setValue('visualization_refresh_interval', 1e3 / new_value)
                new_refresh_interval = 1e3 / new_value
                print(f'Set viz refresh interval to {new_refresh_interval}')
            else:
                dialog_popup(f"Plot FPS range is {config.plot_fps_range}. Please input a number within this range.", enable_dont_show=True, dialog_name='PlotFPSOutOfRangePopup')