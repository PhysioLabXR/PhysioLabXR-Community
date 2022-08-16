# This Python file uses the following encoding: utf-8
import json
import os
import shutil
import uuid

import numpy as np
from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtCore import QSettings, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QMessageBox

from exceptions.exceptions import RenaError, MissingPresetError
from rena import config_ui, config
from rena.config import STOP_PROCESS_KILL_TIMEOUT
from rena.shared import SCRIPT_STOP_SUCCESS, rena_base_script
from rena.startup import load_default_settings
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.threadings import workers
from rena.ui.ScriptConsoleLog import ScriptConsoleLog
from rena.ui.ScriptingInputWidget import ScriptingInputWidget
from rena.ui.ScriptingOutputWidget import ScriptingOutputWidget
from rena.ui_shared import add_icon, minus_icon, script_realtime_info_text
from rena.utils.general import DataBuffer, click_on_file
from rena.utils.networking_utils import recv_string_router, send_data_buffer
from rena.utils.script_utils import *
from rena.utils.settings_utils import get_stream_preset_info, get_stream_preset_names, check_preset_exists

from rena.utils.ui_utils import stream_stylesheet, dialog_popup, add_presets_to_combobox, \
    add_stream_presets_to_combobox, another_window
import pyqtgraph as pg


class ScriptingWidget(QtWidgets.QWidget):

    def __init__(self, parent, port, args):
        super().__init__()
        self.ui = uic.loadUi("ui/ScriptingWidget.ui", self)
        self.parent = parent
        self.port = port
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
        self.frequencyLineEdit.textChanged.connect(self.on_frequency_change)

        self.timeWindowLineEdit.setValidator(QIntValidator())
        self.frequencyLineEdit.setValidator(QIntValidator())

        # self.TopLevelLayout.setStyleSheet("background-color: rgb(36,36,36); margin:5px; border:1px solid rgb(255, 255, 255); ")

        self.removeBtn.setIcon(minus_icon)

        self.is_running = False
        self.is_simulating = False

        self.locateBtn.clicked.connect(self.on_locate_btn_clicked)
        self.createBtn.clicked.connect(self.on_create_btn_clicked)
        self.runBtn.clicked.connect(self.on_run_btn_clicked)
        self.runBtn.setEnabled(False)
        self.script_process = None

        # self.scripting_worker = None
        # self.worker_thread = None
        # self.worker_timer = None
        self.ConsoleLogBtn.clicked.connect(self.on_console_log_btn_clicked)
        self.script_console_log = ScriptConsoleLog()
        self.script_console_log_window = another_window('Console Log')
        self.script_console_log_window.get_layout().addWidget(self.script_console_log)
        self.script_console_log_window.hide()
        self.stdout_timer = None
        self.script_worker = None
        self.stdout_worker_thread = None
        self.create_stdout_worker()  # setup stdout worker

        self.info_socket_interface = None
        self.info_thread = None
        self.info_worker = None
        self.script_pid = None

        self.input_shape_dict = None
        self.forward_input_timer = QTimer()
        self.forward_input_timer.timeout.connect(self.forward_input)
        self.data_buffer = None
        self.forward_input_socket_interface = None

        # self.forward_data_worker = None
        # self.forward_data_thread = None
        if args is not None:  # loading from script preset
            self.id = args['id']
            self.import_script_args(args)
        else:
            self.id = uuid.uuid4()
            self.export_script_args_to_settings()

    def setup_info_worker(self, script_pid):
        self.info_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INFO',
                                                      port_id=self.port + 1,
                                                      identity='client',
                                                      pattern='router-dealer', add_poller=True)
        print('MainApp: Sending command info socket routing ID')
        self.info_socket_interface.send_string('Go')  # send an empty message, this is for setting up the routing id
        self.info_worker = workers.ScriptInfoWorker(self.info_socket_interface, script_pid)
        self.info_worker.abnormal_termination_signal.connect(self.on_script_abnormal_termination)
        self.info_worker.realtime_info_signal.connect(self.show_realtime_info)
        self.info_thread = QThread(
            self.parent)  # set thread to attach to the scriptingtab instead of the widget because it runs a timeout of 2 seconds in the event loop, causing problem when removing the scriptingwidget.
        self.info_worker.moveToThread(self.info_thread)
        self.info_thread.start()

        self.info_timer = QTimer()
        self.info_timer.setInterval(config.SCRIPTING_UPDATE_REFRESH_INTERVA)
        self.info_timer.timeout.connect(self.info_worker.tick_signal.emit)
        self.info_timer.start()

    def setup_forward_input(self, forward_interval, buffer_sizes):
        self.forward_input_timer.setInterval(forward_interval)
        self.data_buffer = DataBuffer(data_type_buffer_sizes=buffer_sizes)
        self.forward_input_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INPUT',
                                                               port_id=self.port + 2,
                                                               identity='client',
                                                               pattern='router-dealer',
                                                               disable_linger=True)
        self.forward_input_timer.start()

    def stop_forward_input(self):
        self.forward_input_timer.stop()
        del self.data_buffer, self.forward_input_socket_interface

    def setup_command_interface(self):
        self.command_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_COMMAND',
                                                         port_id=self.port + 3,
                                                         identity='client',
                                                         pattern='router-dealer', add_poller=True,
                                                         disable_linger=True)  # must disable lingering in case of dead script process
        self.command_socket_interface.send_string('Go')  # send an empty message, this is for setting up the routing id

    def close_command_interface(self):
        del self.command_socket_interface

    def show_realtime_info(self, realtime_info: list):
        self.realtimeInfoLabel.setText(script_realtime_info_text.format(*realtime_info))

    def create_stdout_worker(self):
        self.stdout_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_STDOUT',
                                                        port_id=self.port,
                                                        identity='client',
                                                        pattern='router-dealer')
        self.stdout_worker_thread = QThread(self.parent)
        self.stdout_worker = workers.ScriptingStdoutWorker(self.stdout_socket_interface)
        self.stdout_worker.stdout_signal.connect(self.redirect_script_stdout)
        self.stdout_worker.moveToThread(self.stdout_worker_thread)
        self.stdout_worker_thread.start()
        self.stdout_timer = QTimer()
        self.stdout_timer.setInterval(config.SCRIPTING_UPDATE_REFRESH_INTERVA)
        self.stdout_timer.timeout.connect(self.stdout_worker.tick_signal.emit)
        self.stdout_timer.start()

    def close_stdout(self):
        self.stdout_timer.stop()
        self.stdout_worker_thread.exit()
        # del self.stdout_timer, self.stdout_worker, self.stdout_worker_thread

    def close_info_interface(self):
        self.info_timer.stop()
        self.info_worker.deactivate()
        self.info_thread.exit()

    def on_script_abnormal_termination(self):
        self.stop_run(True)

    def redirect_script_stdout(self, stdout_line: str):
        # print('[Script]: ' + stdout_line)
        if stdout_line != '\n':
            self.script_console_log.print_msg(stdout_line)

    def _validate_script_path(self, script_path):
        try:
            validate_script_path(script_path)
        except RenaError as error:
            dialog_popup(str(error), title='Error')
            return False
        else:
            return True

    def on_run_btn_clicked(self):
        if not self.is_running:
            script_path = self.scriptPathLineEdit.text()
            if not self._validate_script_path(script_path): return
            script_args = self.get_script_args()
            forward_interval = 1e3 / float(self.frequencyLineEdit.text())
            buffer_sizes = [(input_name, input_shape[1] * 2) for input_name, input_shape in
                            zip(self.get_inputs(), self.get_input_shapes())]
            buffer_sizes = dict(buffer_sizes)
            self.script_console_log_window.show()
            self.stdout_socket_interface.send_string(
                'Go')  # send an empty message, this is for setting up the routing id
            self.script_process = start_script(script_path, script_args)
            self.script_pid = self.script_process.pid  # receive the PID
            print('MainApp: User script started on process with PID {}'.format(self.script_pid))
            self.setup_info_worker(self.script_pid)
            self.setup_command_interface()

            self.setup_forward_input(forward_interval, buffer_sizes)
            self.is_running = True
            self.is_simulating = self.simulateCheckbox.isChecked()
            self.change_ui_on_run_stop(self.is_running)
            self.input_shape_dict = self.get_input_shape_dict()
        else:
            self.stop_run(False)

    def stop_run(self, is_abnormal_termination):
        is_timeout_killed = False
        if not is_abnormal_termination:  # no need to call stop if the process is dead
            if not self.notify_script_to_stop():
                is_timeout_killed = True
                self.script_process.kill()
        self.close_info_interface()
        self.close_command_interface()
        self.stop_forward_input()
        del self.info_socket_interface
        self.script_console_log_window.hide()
        self.is_running = False
        self.change_ui_on_run_stop(self.is_running)

        if is_abnormal_termination:
            dialog_popup('Script process terminated abnormally.', title='ERROR')
        if is_timeout_killed:
            dialog_popup('Failed to terminate script process within timeout. Killing it', title='ERROR')

    # def process_command_return(self, command_return):
    #     command, is_success = command_return
    #     if command == SCRIPT_STOP_REQUEST:
    #         if not is_success:
    #             dialog_popup('Failed to terminate script process. Killing it')
    #             self.script_process.kill()
    #     else:
    #         raise NotImplementedError

    def on_console_log_btn_clicked(self):
        self.script_console_log_window.show()
        self.script_console_log_window.activateWindow()

    def on_locate_btn_clicked(self):
        script_path = str(QFileDialog.getOpenFileName(self, "Select File", filter="py(*.py)")[0])
        self.process_locate_script(script_path)
        self.export_script_args_to_settings()

    def process_locate_script(self, script_path):
        if script_path != '':
            if not self._validate_script_path(script_path): return
            self.load_script_name(script_path)
            self.runBtn.setEnabled(True)
        else:
            self.runBtn.setEnabled(False)
        print("Selected script path ", script_path)

    def on_create_btn_clicked(self):
        script_path, _ = QtWidgets.QFileDialog.getSaveFileName()
        if script_path:
            base_script_name = os.path.basename(os.path.normpath(script_path))
            this_script: str = rena_base_script[:]  # make a copy
            this_script = this_script.replace('ExampleRenaScript', base_script_name)
            script_path = script_path + '.py'
            with open(script_path, 'w') as f:
                f.write(this_script)
            try:
                self.load_script_name(script_path)
            except SyntaxError:
                dialog_popup(
                    'The name of the class in your script does not match Python Syntax: {0}. \nPlease change its name before starting'.format(
                        base_script_name), title='WARNING')
            self.runBtn.setEnabled(True)
            click_on_file(script_path)
        else:
            self.runBtn.setEnabled(False)
        print("Selected script path ", script_path)
        self.export_script_args_to_settings()


    def load_script_name(self, script_path):
        self.scriptPathLineEdit.setText(script_path)
        self.scriptNameLabel.setText(get_target_class_name(script_path))

    def change_ui_on_run_stop(self, is_run):
        self.widget_input.setEnabled(not is_run)
        self.widget_output.setEnabled(not is_run)
        self.frequencyLineEdit.setEnabled(not is_run)
        self.timeWindowLineEdit.setEnabled(not is_run)
        self.widget_script_info.setEnabled(not is_run)
        self.runBtn.setText('Run' if not is_run else 'Stop')
        self.simulateCheckbox.setEnabled(not is_run)

    def add_input_clicked(self):
        input_preset_name = self.inputComboBox.currentText()
        self.process_add_input(input_preset_name)
        self.export_script_args_to_settings()

    def process_add_input(self, input_preset_name):
        input_widget = ScriptingInputWidget(input_preset_name)
        try:
            input_widget.set_input_info_text(self.get_preset_input_info_text(input_preset_name))
        except MissingPresetError as e:
            print(str(e))
            return
        self.inputLayout.addWidget(input_widget)

        def remove_btn_clicked():
            self.inputLayout.removeWidget(input_widget)
            input_widget.deleteLater()
            self.input_widgets.remove(input_widget)
            self.check_can_add_input()
            self.export_script_args_to_settings()

        input_widget.set_button_callback(remove_btn_clicked)
        input_widget.button.setIcon(minus_icon)
        self.input_widgets.append(input_widget)
        self.check_can_add_input()
        print('Current items are {0}'.format(str(self.get_inputs())))

    def add_output_clicked(self):
        output_name = self.output_lineEdit.text()
        self.process_add_output(output_name)
        self.export_script_args_to_settings()

    def process_add_output(self, output_name, num_channels=1):
        output_widget = ScriptingOutputWidget(self, output_name, num_channels)
        output_widget.on_channel_num_changed()
        self.outputLayout.addWidget(output_widget)

        def remove_btn_clicked():
            self.outputLayout.removeWidget(output_widget)
            self.output_widgets.remove(output_widget)
            output_widget.deleteLater()
            self.check_can_add_output()
            self.export_script_args_to_settings()

        output_widget.set_button_callback(remove_btn_clicked)
        output_widget.button.setIcon(minus_icon)
        self.output_widgets.append(output_widget)
        self.check_can_add_output()
        print('Current items are {0}'.format(str(self.get_outputs())))

    def add_params_clicked(self):
        # TODO
        pass
        self.export_script_args_to_settings()

    def get_inputs(self):
        return [w.get_input_name_text() for w in self.input_widgets]

    def get_input_shapes(self):
        rtn = []
        for w in self.input_widgets:
            input_preset_name = w.get_input_name_text()
            rtn.append(self.get_preset_expected_shape(input_preset_name))
        return rtn

    def get_input_shape_dict(self):
        return dict([(i, s) for i, s in zip(self.get_inputs(), self.get_input_shapes())])

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
        self.export_script_args_to_settings()

    def on_frequency_change(self):
        self.export_script_args_to_settings()

    def update_input_info(self):
        """
        update the information diplayed in the input box
        """
        for w in self.input_widgets:
            input_preset_name = w.get_input_name_text()
            w.set_input_info_text(self.get_preset_input_info_text(input_preset_name))

    def get_preset_input_info_text(self, preset_name):
        if not check_preset_exists(preset_name):
            raise MissingPresetError(preset_name)
        sampling_rate = get_stream_preset_info(preset_name, 'NominalSamplingRate')
        num_channel = get_stream_preset_info(preset_name, 'NumChannels')
        _timewindow = 0 if self.timeWindowLineEdit.text() == '' else int(self.timeWindowLineEdit.text())
        return '[{0}, {1}]'.format(num_channel, _timewindow * sampling_rate)

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

    def try_close(self):
        if self.is_running:
            # reply = QMessageBox.question(self, 'Window Close', 'Exit Application?',
            #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            # if reply == QMessageBox.Yes:
            #     self.on_run_btn_clicked()
            # else:
            #     return False
            self.on_run_btn_clicked()
        self.close_stdout()
        print('Script widget closed')
        return True

    def set_remove_btn_callback(self, callback):
        self.removeBtn.clicked.connect(callback)

    def on_input_combobox_changed(self):
        self.check_can_add_input()

    def on_output_lineEdit_changed(self):
        self.check_can_add_output()

    def buffer_input(self, data_dict):
        self.data_buffer.update_buffers(data_dict)

    def forward_input(self):
        if self.is_simulating:
            buffer = dict([(input_name, (np.random.rand(*input_shape), np.random.rand(input_shape[1]))) for input_name, input_shape in self.input_shape_dict.items()])
        else:
            buffer = self.data_buffer.buffer
        send_data_buffer(buffer, self.forward_input_socket_interface)

    def notify_script_to_stop(self):
        print("MainApp: sending stop command")
        self.command_socket_interface.send_string(SCRIPT_STOP_REQUEST)
        self.forward_input()  # run the loop so it can process the stop command
        print("MainApp: waiting for stop success")
        events = self.command_socket_interface.poller.poll(STOP_PROCESS_KILL_TIMEOUT)
        if len(events) > 0:
            msg = self.command_socket_interface.socket.recv().decode('utf-8')
        else:
            msg = None
        if msg == SCRIPT_STOP_SUCCESS:
            return True
        else:
            return False

    def get_script_args(self):
        return {'inputs': self.get_inputs(), 'input_shapes': self.get_input_shapes(),
                'outputs': self.get_outputs(), 'output_num_channels': self.get_outputs_num_channels(),
                'params': [], 'port': self.stdout_socket_interface.port_id,
                'run_frequency': int(self.frequencyLineEdit.text()),
                'time_window': int(self.timeWindowLineEdit.text()),
                'script_path': self.scriptPathLineEdit.text(),
                'is_simulate': self.simulateCheckbox.isChecked()}

    def export_script_args_to_settings(self):
        config.settings.beginGroup('scripts/{0}/'.format(self.id))
        config.settings.setValue('inputs', self.get_inputs())
        config.settings.setValue('outputs', self.get_outputs())
        config.settings.setValue('output_num_channels', self.get_outputs_num_channels())
        config.settings.setValue('run_frequency', self.frequencyLineEdit.text())
        config.settings.setValue('time_window', self.timeWindowLineEdit.text())
        config.settings.setValue('script_path', self.scriptPathLineEdit.text())
        config.settings.setValue('is_simulate', self.simulateCheckbox.isChecked())
        config.settings.endGroup()

    def import_script_args(self, args):
        self.process_locate_script(args['script_path'])

        self.frequencyLineEdit.setText(args['run_frequency'])
        self.timeWindowLineEdit.setText(args['time_window'])
        self.simulateCheckbox.setChecked(args['is_simulate'] == 'true')  # is checked?

        for input_preset_name in args['inputs']:
            self.process_add_input(input_preset_name)
        for output_name, output_num_channel in zip(args['outputs'], args['output_num_channels']):
            self.process_add_output(output_name, num_channels=output_num_channel)
