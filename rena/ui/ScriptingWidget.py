# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import QSettings, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton

from exceptions.exceptions import RenaError
from rena import config_ui, config
from rena.startup import load_default_settings
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.threadings import workers
from rena.ui.ScriptingInputWidget import ScriptingInputWidget
from rena.ui.ScriptingOutputWidget import ScriptingOutputWidget
from rena.ui_shared import add_icon, minus_icon
from rena.utils.script_utils import *
from rena.utils.settings_utils import get_stream_preset_info, get_stream_preset_names

from rena.utils.ui_utils import stream_stylesheet, dialog_popup, add_presets_to_combobox, add_stream_presets_to_combobox
import pyqtgraph as pg

class ScriptingWidget(QtWidgets.QWidget):
    settings_changed_signal = pyqtSignal()  # TODO

    def __init__(self, parent, port):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingWidget.ui", self)
        self.parent = parent
        self.settings_changed_signal.connect(self.on_settings_changed)
        self.script = None
        self.input_widgets = []
        self.output_widgets = []

        # add all presents to camera
        add_stream_presets_to_combobox(self.inputComboBox)

        self.addInputBtn.setIcon(add_icon)
        self.addInputBtn.clicked.connect(self.add_input_clicked)

        self.addOutput_btn.setIcon(add_icon)
        self.addOutput_btn.clicked.connect(self.add_output_clicked)
        self.output_lineEdit.textChanged.connect(self.on_output_lineEdit_changed)

        self.addParam_btn.setIcon(add_icon)
        self.inputComboBox.currentTextChanged.connect(self.on_input_combobox_changed)
        self.timeWindowLineEdit.textChanged.connect(self.on_time_window_change)

        self.timeWindowLineEdit.setValidator(QIntValidator())
        self.frequencyLineEdit.setValidator(QIntValidator())

        # self.TopLevelLayout.setStyleSheet("background-color: rgb(36,36,36); margin:5px; border:1px solid rgb(255, 255, 255); ")

        self.removeBtn.setIcon(minus_icon)

        self.locateBtn.clicked.connect(self.on_locate_btn_clicked)
        self.is_running = False
        self.runBtn.clicked.connect(self.on_run_btn_clicked)
        self.script_process = None
        self.command_info_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING',
                                                       port_id=port,
                                                       identity='client',
                                                       pattern='router-dealer')
        # self.scripting_worker = None
        # self.worker_thread = None
        # self.worker_timer = None

    def create_scripting_worker(self):
        self.worker_thread = QThread(self)
        self.scripting_worker = workers.ScriptingWorker(self.command_info_interface)
        self.scripting_worker.stdout_signal.connect(self.redirect_script_stdout)
        self.scripting_worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
        self.worker_timer = QTimer()
        self.worker_timer.setInterval(config.SCRIPTING_UPDATE_REFRESH_INTERVA)  # for 1000 Hz refresh rate
        self.worker_timer.timeout.connect(self.scripting_worker.tick_signal.emit)
        self.worker_timer.start()

    def redirect_script_stdout(self, stdout_line: str):
        print('[Script]: ' + stdout_line)  # TODO move this console log

    def _validate_script_path(self, script_path):
        try:
            validate_script_path(script_path)
        except RenaError as error:
            dialog_popup(str(error), title='Error')
            return False
        else: return True

    def on_run_btn_clicked(self):
        script_path = self.scriptPathLineEdit.text()
        if not self._validate_script_path(script_path): return
        script_args = {'inputs': self.get_inputs(), 'input_shapes': self.get_input_shapes(),
                       'outputs': self.get_outputs(), 'output_num_channels': self.get_outputs_num_channels(),
                       'params': None, 'port': self.command_info_interface.port_id, 'run_frequency': int(self.frequencyLineEdit.text()), 'time_window': int(self.timeWindowLineEdit.text())}
        if not self.is_running:
            self.command_info_interface.send_string('Go')  # send an empty message, this is for setting up the routing id
            self.script_process = start_script(script_path, script_args)
            self.create_scripting_worker()
        else:
            stop_script(self.script_process)  # TODO implement closing of the script process
            #TODO close and stop the worker thread

        self.is_running = not self.is_running
        self.change_ui_on_run_stop(self.is_running)

    def on_locate_btn_clicked(self):
        script_path = str(QFileDialog.getOpenFileName(self, "Select File", filter="py(*.py)")[0])
        if script_path != '':
            if not self._validate_script_path(script_path): return
            self.scriptPathLineEdit.setText(script_path)
            self.scriptNameLabel.setText(get_target_class_name(script_path))
            self.runBtn.setEnabled(True)
        else:
            self.runBtn.setEnabled(False)
        print("Selected script path ", script_path)

    def change_ui_on_run_stop(self, is_run):
        self.widget_input.setEnabled(not is_run)
        self.widget_output.setEnabled(not is_run)
        self.frequencyLineEdit.setEnabled(not is_run)
        self.timeWindowLineEdit.setEnabled(not is_run)
        self.widget_script_info.setEnabled(not is_run)
        self.runBtn.setText('Run' if is_run else 'Stop')

    def add_input_clicked(self):
        input_preset_name = self.inputComboBox.currentText()
        input_widget = ScriptingInputWidget(input_preset_name)
        input_widget.set_input_info_text(self.get_preset_input_info_text(input_preset_name))
        self.inputLayout.addWidget(input_widget)

        def remove_btn_clicked():
            self.inputLayout.removeWidget(input_widget)
            input_widget.deleteLater()
            self.input_widgets.remove(input_widget)
            self.check_can_add_input()
        input_widget.set_button_callback(remove_btn_clicked)
        input_widget.button.setIcon(minus_icon)
        self.input_widgets.append(input_widget)
        self.check_can_add_input()
        print('Current items are {0}'.format(str(self.get_inputs())))

    def add_output_clicked(self):
        output_name = self.output_lineEdit.text()
        output_widget = ScriptingOutputWidget(output_name)
        output_widget.on_channel_num_changed()
        self.outputLayout.addWidget(output_widget)
        def remove_btn_clicked():
            self.outputLayout.removeWidget(output_widget)
            self.output_widgets.remove(output_widget)
            output_widget.deleteLater()
            self.check_can_add_output()
        output_widget.set_button_callback(remove_btn_clicked)
        output_widget.button.setIcon(minus_icon)
        self.output_widgets.append(output_widget)
        self.check_can_add_output()
        print('Current items are {0}'.format(str(self.get_outputs())))

    def get_inputs(self):
        return [w.get_input_name_text() for w in self.input_widgets]

    def get_input_shapes(self):
        rtn = []
        for w in self.input_widgets:
            input_preset_name = w.get_input_name_text()
            rtn.append(self.get_preset_expected_shape(input_preset_name))
        return rtn

    def get_outputs(self):
        return [w.get_label_text() for w in self.output_widgets]

    def get_outputs_num_channels(self):
        return [w.get_num_channels() for w in self.output_widgets]

    def check_can_add_input(self):
        """
        will disable the add button if duplicate input exists
        """
        input_preset_name = self.inputComboBox.currentText()
        if input_preset_name in self.get_inputs() or input_preset_name not in get_stream_preset_names():
            self.addInputBtn.setEnabled(False)
        else:
            self.addInputBtn.setEnabled(True)

    def check_can_add_output(self):
        output_name = self.output_lineEdit.text()
        if output_name in self.get_outputs():
            self.addOutput_btn.setEnabled(False)
        else:
            self.addOutput_btn.setEnabled(True)

    def on_time_window_change(self):
        self.update_input_info()

    def update_input_info(self):
        """
        update the information diplayed in the input box
        """
        for w in self.input_widgets:
            input_preset_name = w.get_input_name_text()
            w.set_input_info_text(self.get_preset_input_info_text(input_preset_name))

    def get_preset_input_info_text(self, preset_name):
        sampling_rate = get_stream_preset_info(preset_name, 'NominalSamplingRate')
        num_channel = get_stream_preset_info(preset_name, 'NumChannels')
        return '[{0}, {1}]'.format(num_channel, int(self.timeWindowLineEdit.text()) * sampling_rate)

    def get_preset_expected_shape(self, preset_name):
        sampling_rate = get_stream_preset_info(preset_name, 'NominalSamplingRate')
        num_channel = get_stream_preset_info(preset_name, 'NumChannels')
        return num_channel, int(self.timeWindowLineEdit.text()) * sampling_rate

    def on_settings_changed(self):
        """
        TODO should be called after every setting change
        """
        self.update_input_info()

    def on_time_window_chagned(self):
        self.update_input_info()

    def closeEvent(self, event):
        print('Script widget closed')
        event.accept()  # let the widget close

    def set_remove_btn_callback(self, callback):
        self.removeBtn.clicked.connect(callback)

    def on_input_combobox_changed(self):
        self.check_can_add_input()

    def on_output_lineEdit_changed(self):
        self.check_can_add_output()

