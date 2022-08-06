# This Python file uses the following encoding: utf-8
import json
import os

from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton

from exceptions.exceptions import RenaError
from rena import config_ui, config
from rena.startup import load_default_settings
from rena.ui.ScriptingInputWidget import ScriptingInputWidget
from rena.ui.ScriptingOutputWidget import ScriptingOutputWidget
from rena.ui_shared import add_icon, minus_icon
from rena.utils.script_utils import *
from rena.utils.settings_utils import get_stream_preset_info, get_stream_preset_names

from rena.utils.ui_utils import stream_stylesheet, dialog_popup, add_presets_to_combobox, add_stream_presets_to_combobox
import pyqtgraph as pg

class ScriptingWidget(QtWidgets.QWidget):
    settings_changed_signal = pyqtSignal()  # TODO

    def __init__(self, parent):
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

        self.is_running = False
        self.runBtn.clicked.connect(self.on_run_btn_clicked)

    def on_run_btn_clicked(self):
        script_path = self.scriptPathLineEdit.text()
        try:
            validate_script_path(script_path)
        except RenaError as error:
            dialog_popup(str(error), title='Error')
            return

        if self.is_running:
            self.widget_input.setEnabled(False)
            self.widget_output.setEnabled(False)
            self.runBtn.setText('Stop')
            start_script(script_path)
        else:
            self.widget_input.setEnabled(True)
            self.widget_output.setEnabled(True)
            self.runBtn.setText('Run')
            stop_script(script_path)
        self.is_running = not self.is_running


    def on_locate_btn_clicked(self):
        selected_script_path = str(QFileDialog.getOpenFileName(self, "Select File", filter="py(*.py)"))
        if selected_script_path != '':
            self.scriptPathLineEdit.setText(selected_script_path)
        print("Selected script path ", selected_script_path)


    def add_input_clicked(self):
        input_preset_name = self.inputComboBox.currentText()
        input_widget = ScriptingInputWidget(input_preset_name)
        input_widget.set_label_2_text(self.get_preset_input_info_text(input_preset_name))
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
        return [w.get_label_text() for w in self.input_widgets]

    def get_outputs(self):
        return [w.get_label_text() for w in self.output_widgets]

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
            input_preset_name = w.get_label_text()
            w.set_label_2_text(self.get_preset_input_info_text(input_preset_name))

    def get_preset_input_info_text(self, preset_name):
        sampling_rate = get_stream_preset_info(preset_name, 'NominalSamplingRate')
        num_channel = get_stream_preset_info(preset_name, 'NumChannels')
        return '[{0}, {1}]'.format(num_channel, int(self.timeWindowLineEdit.text()) * sampling_rate)

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

