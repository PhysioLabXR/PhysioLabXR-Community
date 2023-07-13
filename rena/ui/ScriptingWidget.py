# This Python file uses the following encoding: utf-8
import json
import os
import uuid

import numpy as np
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtGui import QIntValidator

from PyQt6.QtWidgets import QFileDialog, QLayout

from rena.exceptions.exceptions import MissingPresetError
from rena.config import STOP_PROCESS_KILL_TIMEOUT, SCRIPTING_UPDATE_REFRESH_INTERVA
from rena.presets.Presets import Presets
from rena.presets.ScriptPresets import ScriptPreset
from rena.scripting.RenaScript import RenaScript
from rena.scripting.script_utils import start_rena_script, get_target_class_name
from rena.scripting.scripting_enums import ParamChange, ParamType
from rena.shared import SCRIPT_STOP_SUCCESS, rena_base_script, SCRIPT_PARAM_CHANGE, SCRIPT_STOP_REQUEST
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.threadings import workers
from rena.ui.PoppableWidget import Poppable
from rena.ui.ScriptConsoleLog import ScriptConsoleLog
from rena.ui.ScriptingInputWidget import ScriptingInputWidget
from rena.ui.ScriptingOutputWidget import ScriptingOutputWidget
from rena.ui.ParamWidget import ParamWidget
from rena.ui_shared import add_icon, minus_icon, script_realtime_info_text
from rena.utils.buffers import DataBuffer, click_on_file
from rena.utils.networking_utils import send_data_dict
from rena.presets.presets_utils import get_stream_preset_names, get_experiment_preset_streams, \
    get_experiment_preset_names, get_stream_preset_info, check_preset_exists, remove_script_from_settings

from rena.utils.ui_utils import dialog_popup, add_presets_to_combobox, \
    another_window, update_presets_to_combobox, validate_script_path


class ScriptingWidget(Poppable, QtWidgets.QWidget):

    def __init__(self, parent_widget: QtWidgets, port, script_preset: ScriptPreset, layout: QLayout):
        super().__init__('Rena Script', parent_widget, layout, self.remove_script_clicked)
        self.ui = uic.loadUi("ui/ScriptingWidget.ui", self)
        self.set_pop_button(self.PopWindowBtn)

        self.parent = parent_widget
        self.port = port
        self.script = None
        self.input_widgets = []
        self.output_widgets = []
        self.param_widgets = []

        # add all presents to camera
        add_presets_to_combobox(self.inputComboBox)

        # set up the add buttons
        self.removeBtn.clicked.connect(self.remove_script_clicked)
        self.addInputBtn.setIcon(add_icon)
        self.addInputBtn.clicked.connect(self.add_input_clicked)
        self.inputComboBox.lineEdit().textChanged.connect(self.on_input_combobox_changed)
        self.inputComboBox.lineEdit().returnPressed.connect(self.addInputBtn.click)

        self.addOutput_btn.setIcon(add_icon)
        self.addOutput_btn.clicked.connect(self.add_output_clicked)
        self.output_lineEdit.textChanged.connect(self.on_output_lineEdit_changed)
        self.output_lineEdit.returnPressed.connect(self.addOutput_btn.click)

        self.addParam_btn.setIcon(add_icon)
        self.addParam_btn.clicked.connect(self.add_params_clicked)
        self.param_lineEdit.textChanged.connect(self.check_can_add_param)
        self.param_lineEdit.returnPressed.connect(self.addParam_btn.click)

        self.timeWindowLineEdit.textChanged.connect(self.on_time_window_change)
        self.frequencyLineEdit.textChanged.connect(self.on_frequency_change)

        self.timeWindowLineEdit.setValidator(QIntValidator())
        self.frequencyLineEdit.setValidator(QIntValidator())

        self.simulateCheckbox.stateChanged.connect(self.onSimulationCheckboxChanged)
        # self.TopLevelLayout.setStyleSheet("background-color: rgb(36,36,36); margin:5px; border:1px solid rgb(255, 255, 255); ")

        self.removeBtn.setIcon(minus_icon)

        self.is_running = False
        self.is_simulating = False

        self.locateBtn.clicked.connect(self.on_locate_btn_clicked)
        self.createBtn.clicked.connect(self.on_create_btn_clicked)
        self.runBtn.clicked.connect(self.on_run_btn_clicked)
        self.runBtn.setEnabled(False)
        self.script_process = None

        # Fields for the console output window #########################################################################
        self.ConsoleLogBtn.clicked.connect(self.on_console_log_btn_clicked)
        self.script_console_log = ScriptConsoleLog()
        self.script_console_log_window = another_window('Console Log')
        self.script_console_log_window.get_layout().addWidget(self.script_console_log)
        self.script_console_log_window.hide()
        self.stdout_timer = None
        self.script_worker = None
        self.stdout_worker_thread = None
        self.create_stdout_worker()  # setup stdout worker

        # Fields readin information from the script process, including timing performance and check for abnormal termination
        self.info_socket_interface = None
        self.info_thread = None
        self.info_worker = None
        self.script_pid = None

        # Fields for the console output window #########################################################################
        self.input_shape_dict = None  # to keep the input shape for each forward input callback, so we don't query the UI everytime for the input shapes
        self.run_signal_timer = QTimer()
        self.run_signal_timer.timeout.connect(self.run_signal)
        # self.data_buffer = None
        self.forward_input_socket_interface = None

        # loading from script preset from the persistent sittings ######################################################
        if script_preset is not None:
            self.id = script_preset.id
            self.import_script_args(script_preset)
        else:
            self.id = str(uuid.uuid4())
            self.export_script_args_to_settings()

        self.internal_data_buffer = None


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
        self.info_timer.setInterval(SCRIPTING_UPDATE_REFRESH_INTERVA)
        self.info_timer.timeout.connect(self.info_worker.tick_signal.emit)
        self.info_timer.start()

    def setup_forward_input(self, forward_interval, internal_buffer_sizes):
        self.run_signal_timer.setInterval(int(forward_interval))
        self.internal_data_buffer = DataBuffer(stream_buffer_sizes=internal_buffer_sizes)  # buffer that keeps data between run signals
        self.forward_input_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INPUT',
                                                               port_id=self.port + 2,
                                                               identity='client',
                                                               pattern='router-dealer',
                                                               disable_linger=True)
        self.run_signal_timer.start()

    def stop_run_signal_forward_input(self):
        self.run_signal_timer.stop()
        del self.internal_data_buffer, self.forward_input_socket_interface

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
        self.stdout_timer.setInterval(SCRIPTING_UPDATE_REFRESH_INTERVA)
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

    def on_run_btn_clicked(self):
        if not self.is_running:
            script_path = self.scriptPathLineEdit.text()
            if not validate_script_path(script_path, RenaScript): return
            forward_interval = 1e3 / float(self.frequencyLineEdit.text())

            self.script_console_log_window.show()
            self.stdout_socket_interface.send_string(
                'Go')  # send an empty message, this is for setting up the routing id

            script_args = self.get_script_args()
            self.script_process = start_rena_script(script_path, script_args)
            self.script_pid = self.script_process.pid  # receive the PID
            print('MainApp: User script started on process with PID {}'.format(self.script_pid))
            self.setup_info_worker(self.script_pid)
            self.setup_command_interface()

            internal_buffer_size = dict([(name, size * 2)for name, size in script_args['buffer_sizes'].items()])
            self.setup_forward_input(forward_interval, internal_buffer_size)
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
        self.stop_run_signal_forward_input()
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
            if not validate_script_path(script_path, RenaScript):
                self.runBtn.setEnabled(False)
                return
            self.load_script_name(script_path)
            self.runBtn.setEnabled(True)
        else:
            self.runBtn.setEnabled(False)
        print("Selected script path ", script_path)

    def on_create_btn_clicked(self):
        script_path, _ = QtWidgets.QFileDialog.getSaveFileName()
        self.create_script(script_path)

    def create_script(self, script_path, is_open_file=True):
        if script_path:
            base_script_name = os.path.basename(os.path.normpath(script_path))
            this_script: str = rena_base_script[:]  # make a copy
            class_name = base_script_name if not base_script_name.endswith('.py') else base_script_name.strip('.py')
            this_script = this_script.replace('BaseRenaScript', class_name)
            if not script_path.endswith('.py'):
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
            if is_open_file:
                click_on_file(script_path)
        else:
            self.runBtn.setEnabled(False)
        print("Selected script path ", script_path)
        self.export_script_args_to_settings()

    def load_script_name(self, script_path):
        self.scriptPathLineEdit.setText(script_path)
        self.scriptNameLabel.setText(get_target_class_name(script_path, RenaScript))

    def change_ui_on_run_stop(self, is_run):
        self.widget_input.setEnabled(not is_run)
        self.widget_output.setEnabled(not is_run)
        self.frequencyLineEdit.setEnabled(not is_run)
        self.timeWindowLineEdit.setEnabled(not is_run)
        self.widget_script_basic_info.setEnabled(not is_run)
        self.runBtn.setText('Run' if not is_run else 'Stop')
        self.simulateCheckbox.setEnabled(not is_run)

    def add_input_clicked(self):
        input_preset_name = self.inputComboBox.currentText()
        self.process_add_input(input_preset_name)
        self.export_script_args_to_settings()

    def process_add_input(self, input_preset_name):
        existing_inputs = self.get_inputs()
        if input_preset_name in get_stream_preset_names():
            self.add_input_widget(input_preset_name)
        elif input_preset_name in get_experiment_preset_names():
            stream_names = get_experiment_preset_streams(input_preset_name)
            for s_name in stream_names:
                if s_name not in existing_inputs:
                    self.add_input_widget(s_name)

    def add_input_widget(self, stream_name):
        input_widget = ScriptingInputWidget(stream_name)
        try:
            input_widget.set_input_info_text(self.get_preset_input_info_text(stream_name))
        except MissingPresetError as e:
            print(str(e))
            return
        self.inputLayout.addWidget(input_widget)
        self.inputLayout.setAlignment(input_widget, QtCore.Qt.AlignmentFlag.AlignTop)

        def remove_btn_clicked():
            self.inputLayout.removeWidget(input_widget)
            input_widget.deleteLater()
            self.input_widgets.remove(input_widget)
            self.check_can_add_input()
            self.export_script_args_to_settings()

        input_widget.set_button_callback(remove_btn_clicked)
        self.input_widgets.append(input_widget)
        self.check_can_add_input()
        print('Current items are {0}'.format(str(self.get_inputs())))

    def add_output_clicked(self):
        output_name = self.output_lineEdit.text()
        self.process_add_output(output_name)
        self.export_script_args_to_settings()

    def process_add_output(self, output_name, num_channels=1):
        output_widget = ScriptingOutputWidget(self, output_name, num_channels)
        self.outputLayout.addWidget(output_widget)
        self.outputLayout.setAlignment(output_widget, QtCore.Qt.AlignmentFlag.AlignTop)

        def remove_btn_clicked():
            self.outputLayout.removeWidget(output_widget)
            self.output_widgets.remove(output_widget)
            output_widget.deleteLater()
            self.check_can_add_output()
            self.export_script_args_to_settings()

        output_widget.set_button_callback(remove_btn_clicked)
        self.output_widgets.append(output_widget)
        self.check_can_add_output()
        print('Current items are {0}'.format(str(self.get_outputs())))

    def add_params_clicked(self):
        param_name = self.param_lineEdit.text()
        self.process_add_param(param_name)
        self.export_script_args_to_settings()

    def process_add_param(self, param_name, param_type=ParamType.bool, value=None):
        param_widget = ParamWidget(self, param_name, param_type=param_type, value=value)
        self.paramsLayout.addWidget(param_widget)
        self.paramsLayout.setAlignment(param_widget, QtCore.Qt.AlignmentFlag.AlignTop)

        def remove_btn_clicked():
            self.paramsLayout.removeWidget(param_widget)
            self.param_widgets.remove(param_widget)
            param_widget.deleteLater()
            self.check_can_add_param()
            self.export_script_args_to_settings()
            self.notify_script_process_param_change(ParamChange.REMOVE, param_name)

        param_widget.set_remove_button_callback(remove_btn_clicked)
        self.param_widgets.append(param_widget)
        self.check_can_add_param()
        self.notify_script_process_param_change(ParamChange.ADD, param_name, value=param_widget.get_value())

    def notify_script_process_param_change(self, change: ParamChange, name, value=None):
        '''
        send changed params to the script process
        @return:
        '''
        print('Param {} changed: {}, {}'.format(name, change, value))
        self.export_script_args_to_settings()
        if change == ParamChange.CHANGE:
            assert value is not None
        if self.is_running:
            self.forward_param_change(change, name, value)

    def get_inputs(self):
        return [w.get_input_name_text() for w in self.input_widgets]

    def get_input_shapes(self):
        rtn = dict()
        for w in self.input_widgets:
            input_preset_name = w.get_input_name_text()
            rtn[input_preset_name] = self.get_preset_expected_shape(input_preset_name)
        return rtn

    def get_input_shape_dict(self):
        return dict([(i, s) for i, s in zip(self.get_inputs(), self.get_input_shapes())])

    def get_outputs(self):
        return [w.get_label_text() for w in self.output_widgets]

    def get_outputs_num_channels(self):
        return [w.get_num_channels() for w in self.output_widgets]

    def get_params(self):
        return [w.get_param_name() for w in self.param_widgets]

    # def get_param_value_texts(self):
    #     return [w.get_value_text() for w in self.param_widgets]

    def get_param_types(self):
        return [w.get_param_type() for w in self.param_widgets]

    def get_params_presets_recursive(self):
        params_presets = [x.get_param_preset_recursive() for x in self.param_widgets]
        return params_presets

    def get_param_dict(self):
        return dict([(w.get_param_name(), w.get_value()) for w in self.param_widgets])

    def check_can_add_input(self):
        """
        will disable the add button if duplicate input exists
        """
        experiment_presets = get_experiment_preset_names()
        input_preset_name = self.inputComboBox.currentText()
        if input_preset_name in self.get_inputs() or input_preset_name not in get_stream_preset_names() + experiment_presets:
            self.addInputBtn.setEnabled(False)
        else:
            self.addInputBtn.setEnabled(True)

        if input_preset_name in experiment_presets:
            if np.all([x in self.get_inputs() for x in get_experiment_preset_streams(input_preset_name)]):
                self.addInputBtn.setEnabled(False)
            else:
                self.addInputBtn.setEnabled(True)

    def check_can_add_output(self):
        output_name = self.output_lineEdit.text()
        if output_name in self.get_outputs():
            self.addOutput_btn.setEnabled(False)
        else:
            self.addOutput_btn.setEnabled(True)

    def check_can_add_param(self):
        param_name = self.param_lineEdit.text()
        if param_name in self.get_params():
            self.addParam_btn.setEnabled(False)
        else:
            self.addParam_btn.setEnabled(True)

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
        sampling_rate = get_stream_preset_info(preset_name, 'nominal_sampling_rate')
        num_channel = get_stream_preset_info(preset_name, 'num_channels')
        _timewindow = 0 if self.timeWindowLineEdit.text() == '' else int(self.timeWindowLineEdit.text())
        return '[{0}, {1}]'.format(num_channel, _timewindow * sampling_rate)

    def get_preset_expected_shape(self, preset_name):
        sampling_rate = get_stream_preset_info(preset_name, 'nominal_sampling_rate')
        num_channel = get_stream_preset_info(preset_name, 'num_channels')
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

    def remove_script_clicked(self):
        # self.ScriptingWidgetScrollLayout.removeWidget(script_widget)
        if self.is_popped:
            self.delete_window()
        self.deleteLater()
        remove_script_from_settings(self.id)
        self.script_console_log_window.close()

    def on_input_combobox_changed(self):
        self.check_can_add_input()

    def on_output_lineEdit_changed(self):
        self.check_can_add_output()

    def send_input(self, data_dict):
        if np.any(np.array(data_dict["timestamps"])< 100):
            print('skipping input with timestamp < 100')
        self.internal_data_buffer.update_buffer(data_dict)
        # send_data_dict(data_dict, self.forward_input_socket_interface)

    def run_signal(self):
        # if self.is_simulating:
        #     buffer = dict([(input_name, (np.random.rand(*input_shape), np.random.rand(input_shape[1]))) for
        #                    input_name, input_shape in self.input_shape_dict.items()])
        # else:
        buffer = self.internal_data_buffer.buffer
        send_data_dict(buffer, self.forward_input_socket_interface)
        self.internal_data_buffer.clear_buffer()

    def notify_script_to_stop(self):
        print("MainApp: sending stop command")
        self.command_socket_interface.send_string(SCRIPT_STOP_REQUEST)
        self.run_signal()  # run the loop so it can process the stop command
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
        buffer_sizes = [(input_name, input_shape[1]) for input_name, input_shape in
                        self.get_input_shapes().items()]
        buffer_sizes = dict(buffer_sizes)
        return {'inputs': self.get_inputs(),
                'input_shapes': self.get_input_shapes(),
                'buffer_sizes': buffer_sizes,
                'outputs': self.get_outputs(), 'output_num_channels': self.get_outputs_num_channels(),
                'params': self.get_param_dict(), 'port': self.stdout_socket_interface.port_id,
                'run_frequency': int(self.frequencyLineEdit.text()),
                'time_window': int(self.timeWindowLineEdit.text()),
                'script_path': self.scriptPathLineEdit.text(),
                'is_simulate': self.simulateCheckbox.isChecked()}

    def export_script_args_to_settings(self):
        script_preset = ScriptPreset(id=self.id, inputs=self.get_inputs(), outputs=self.get_outputs(), output_num_channels=self.get_outputs_num_channels(),
                                     # params=self.get_params(), params_types=self.get_param_types(), params_value_strs=self.get_param_value_texts(),
                                     param_presets = self.get_params_presets_recursive(),
                                     run_frequency=self.frequencyLineEdit.text(), time_window=self.timeWindowLineEdit.text(),
                                     script_path=self.scriptPathLineEdit.text(), is_simulate=self.simulateCheckbox.isChecked())
        Presets().script_presets[self.id] = script_preset

    def import_script_args(self, script_preset: ScriptPreset):
        self.process_locate_script(script_preset.script_path)

        self.frequencyLineEdit.setText(script_preset.run_frequency)
        self.timeWindowLineEdit.setText(script_preset.time_window)
        self.simulateCheckbox.setChecked(script_preset.is_simulate)  # is checked?

        for input_preset_name in script_preset.inputs:
            self.process_add_input(input_preset_name)
        for output_name, output_num_channel in zip(script_preset.outputs, script_preset.output_num_channels):
            self.process_add_output(output_name, num_channels=output_num_channel)

        for param_preset in script_preset.param_presets:
            self.process_add_param(param_preset.name, param_type=param_preset.type, value=param_preset.value)

    def update_input_combobox(self):
        update_presets_to_combobox(self.inputComboBox)

    def forward_param_change(self, change: ParamChange, name, value):
        self.command_socket_interface.socket.send_string(SCRIPT_PARAM_CHANGE)
        self.command_socket_interface.socket.send_string('|'.join([change.value, name, type(value).__name__]))
        self.command_socket_interface.socket.send_string(json.dumps(value))

    def onSimulationCheckboxChanged(self):
        print('Script {} simulating input.'.format('is' if self.simulateCheckbox.isChecked() else 'isn\'t'))
