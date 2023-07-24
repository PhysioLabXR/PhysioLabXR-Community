# This Python file uses the following encoding: utf-8
import multiprocessing

import pyqtgraph as pg
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QThread
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import QFileDialog, QDialogButtonBox

from rena import config
from rena.configs.GlobalSignals import GlobalSignals
from rena.configs.configs import AppConfigs, LinechartVizMode, RecordingFileFormat
from rena.presets.Presets import Presets, PresetType, _load_video_device_presets
from rena.startup import load_settings
from rena.threadings.WaitThreads import WaitForProcessWorker, ProcessWithQueue
from rena.utils.ui_utils import stream_stylesheet, dialog_popup


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
        self.saveFormatComboBox.addItems([member.value for member in RecordingFileFormat.__members__.values()])
        self.saveFormatComboBox.activated.connect(self.recording_file_format_change)

        self.reset_default_button.clicked.connect(self.reset_default)
        self.reload_stream_preset_button.clicked.connect(self.reload_stream_presets)

        self.plot_fps_lineedit.textChanged.connect(self.on_plot_fps_changed)
        onlyInt = QIntValidator()
        onlyInt.setRange(*config.plot_fps_range)
        self.plot_fps_lineedit.setValidator(onlyInt)
        self.plot_fps_lineedit.setText(str(int(1e3 / int(float(AppConfigs().visualization_refresh_interval)))))

        self.linechart_viz_mode_combobox.addItems([member.value for member in LinechartVizMode.__members__.values()])
        self.linechart_viz_mode_combobox.activated.connect(self.on_linechart_viz_mode_changed)

        self.load_settings_to_ui()

        # start a thread to listen to video preset reloading
        # self.zmq_endpoint = "tcp://127.0.0.1:5550"
        self._load_video_device_process = None
        self.wait_process_thread = None
        self.wait_process_worker = None
        self.reload_video_device_button.clicked.connect(self.reload_video_device_presets)
        self.reload_video_device_presets()

    def reload_video_device_presets(self):
        """
        this function will start a separate process look for video devices.
        an outside qthread must monitor the return of this process and call Presets().add_video_presets(rtn), where
        rtn is the return of the process Presets()._load_video_device_process.


        """
        Presets().remove_video_presets()
        GlobalSignals().stream_presets_entry_changed_signal.emit()
        Presets().add_video_preset_by_fields('monitor 0', PresetType.MONITOR, 0)  # always add the monitor 0 preset
        self._load_video_device_process = ProcessWithQueue(target=_load_video_device_presets)
        self._load_video_device_process.start()
        self.wait_process_thread = QThread()
        self.wait_process_worker = WaitForProcessWorker(self._load_video_device_process)
        self.wait_process_worker.process_finished.connect(self.on_video_device_preset_reloaded)
        self.wait_process_worker.moveToThread(self.wait_process_thread)
        self.reload_video_device_button.setEnabled(False)
        self.reload_video_device_button.setText("Reloading...")
        self.wait_process_thread.started.connect(self.wait_process_worker.run)
        self.wait_process_thread.start()

    def on_video_device_preset_reloaded(self, video_presets):
        Presets().add_video_presets(video_presets)
        GlobalSignals().stream_presets_entry_changed_signal.emit()
        self.reload_video_device_button.setEnabled(True)
        self.reload_video_device_button.setText("Reload Video Devices")

    def load_settings_to_ui(self):
        self.linechart_viz_mode_combobox.setCurrentText(AppConfigs().linechart_viz_mode.value)
        self.saveFormatComboBox.setCurrentText(AppConfigs().recording_file_format.value)

    def switch_to_tab(self, tab_name: str):
        for index in range(self.settings_tabs.count()):
            tab_widget = self.settings_tabs.widget(index)
            if tab_name.lower() in tab_widget.objectName():
                self.settings_tabs.setCurrentIndex(index)
                return  # Exit the function once the tab is found
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
        if self.saveFormatComboBox.currentText() != RecordingFileFormat.dats.value:
            dialog_popup('Using data format other than Rena Native will result in a conversion time after finishing a '
                         'recording', title='Info', dialog_name='file_format_info', enable_dont_show=True, mode='modeless', main_parent=self.parent)
        AppConfigs().recording_file_format = RecordingFileFormat(self.saveFormatComboBox.currentText())
        print(f"recording_file_format_change: {AppConfigs().recording_file_format}")
        self.parent.recording_tab.update_ui_save_file()

    def reset_default(self):
        # marked for refactor
        config.settings.clear()
        load_settings()

        self.set_theme(config.settings.value('theme'))
        self.set_recording_file_location(config.DEFAULT_DATA_DIR)

        AppConfigs().revert_to_default()

        self.load_settings_to_ui()

    def set_recording_file_location(self, selected_data_dir: str):
        if selected_data_dir != '':
            config.settings.setValue('recording_file_location', selected_data_dir)
            print("Selected recording file location: ", config.settings.value('recording_file_location'))
            self.saveRootTextEdit.setText(config.settings.value('recording_file_location'))
            self.parent.recording_tab.update_ui_save_file()

    def on_plot_fps_changed(self):
        print(f"plot_fps_lineedit changed value is {self.plot_fps_lineedit.text()}")

        if self.plot_fps_lineedit.text() != '':
            new_value = int(self.plot_fps_lineedit.text())
            if new_value in range(config.plot_fps_range[0], config.plot_fps_range[1]+1):
                AppConfigs().visualization_refresh_interval = int(1e3 / new_value)
                new_refresh_interval = 1e3 / new_value
                print(f'Set viz refresh interval to {new_refresh_interval}')
            else:
                dialog_popup(f"Plot FPS range is {config.plot_fps_range}. Please input a number within this range.", enable_dont_show=True, dialog_name='PlotFPSOutOfRangePopup')

    def on_linechart_viz_mode_changed(self):
        AppConfigs().linechart_viz_mode = LinechartVizMode(self.linechart_viz_mode_combobox.currentText())
        print(f'Linechart viz mode changed to {AppConfigs().linechart_viz_mode}')

    def reload_stream_presets(self):
        if self.parent.is_any_stream_widget_added():
            dialog_popup('Please remove all stream widgets before reloading stream presets', title='Info', buttons=QDialogButtonBox.StandardButton.Ok)
            return
        Presets().reload_stream_presets()
        GlobalSignals().stream_presets_entry_changed_signal.emit()
        dialog_popup('Stream presets reloaded!', title='Info', buttons=QDialogButtonBox.StandardButton.Ok)

    def try_close(self):
        if self._load_video_device_process is not None and self._load_video_device_process.is_alive():
            self._load_video_device_process.terminate()
        self.wait_process_thread.quit()