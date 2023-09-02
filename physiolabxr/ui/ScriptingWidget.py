# This Python file uses the following encoding: utf-8
import json
import os
import uuid
from typing import List

import numpy as np
import psutil
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtGui import QMovie

from PyQt6.QtWidgets import QFileDialog, QLayout

from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.exceptions.exceptions import MissingPresetError, UnsupportedLSLDataTypeError, RenaError
from physiolabxr.configs.config import SCRIPTING_UPDATE_REFRESH_INTERVAL
from physiolabxr.presets.PresetEnums import DataType, PresetType
from physiolabxr.presets.Presets import Presets
from physiolabxr.presets.ScriptPresets import ScriptPreset, ScriptOutput
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.script_utils import start_rena_script, get_target_class_name
from physiolabxr.scripting.scripting_enums import ParamChange, ParamType
from physiolabxr.configs.shared import SCRIPT_STOP_SUCCESS, SCRIPT_PARAM_CHANGE, SCRIPT_STOP_REQUEST
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.threadings import workers
from physiolabxr.threadings.WaitThreads import start_wait_for_response
from physiolabxr.ui.PoppableWidget import Poppable
from physiolabxr.ui.ScriptConsoleLog import ScriptConsoleLog
from physiolabxr.ui.ScriptingInputWidget import ScriptingInputWidget
from physiolabxr.ui.ScriptingOutputWidget import ScriptingOutputWidget
from physiolabxr.ui.ParamWidget import ParamWidget
from physiolabxr.ui.ui_shared import script_realtime_info_text
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.buffers import DataBuffer, click_on_file
from physiolabxr.utils.networking_utils import send_data_dict
from physiolabxr.presets.presets_utils import get_stream_preset_names, get_experiment_preset_streams, \
    get_experiment_preset_names, get_stream_preset_info, is_stream_name_in_presets, remove_script_from_settings

from physiolabxr.utils.ui_utils import dialog_popup, add_presets_to_combobox, \
    another_window, update_presets_to_combobox, validate_script_path, show_label_movie


class ScriptingWidget(Poppable, QtWidgets.QWidget):

    def __init__(self, parent_widget: QtWidgets, main_window, port, script_preset: ScriptPreset, layout: QLayout):
        super().__init__('Rena Script', parent_widget, layout, self.remove_script_clicked)
        self.ui = uic.loadUi(AppConfigs()._ui_ScriptingWidget, self)
        self.set_pop_button(self.PopWindowBtn)

        self.parent = parent_widget
        self.port = port
        self.script = None
        self.input_widgets = []
        self.output_widgets = []
        self.param_widgets = []
        self.main_window = main_window

        # add all presents to camera
        add_presets_to_combobox(self.inputComboBox)

        # set up the add buttons
        self.removeBtn.clicked.connect(self.remove_script_clicked)
        self.addInputBtn.setIcon(AppConfigs()._icon_add)
        self.addInputBtn.clicked.connect(self.add_input_clicked)
        self.inputComboBox.lineEdit().textChanged.connect(self.on_input_combobox_changed)
        self.inputComboBox.lineEdit().returnPressed.connect(self.addInputBtn.click)

        self.addOutput_btn.setIcon(AppConfigs()._icon_add)
        self.addOutput_btn.clicked.connect(self.add_output_clicked)
        self.output_lineEdit.textChanged.connect(self.on_output_lineEdit_changed)
        self.output_lineEdit.returnPressed.connect(self.addOutput_btn.click)

        self.addParam_btn.setIcon(AppConfigs()._icon_add)
        self.addParam_btn.clicked.connect(self.add_params_clicked)
        self.param_lineEdit.textChanged.connect(self.check_can_add_param)
        self.param_lineEdit.returnPressed.connect(self.addParam_btn.click)

        self.timeWindowLineEdit.textChanged.connect(self.on_time_window_change)
        self.frequencyLineEdit.textChanged.connect(self.on_frequency_change)

        self.timeWindowLineEdit.setValidator(NoCommaIntValidator())
        self.frequencyLineEdit.setValidator(NoCommaIntValidator())

        self.simulateCheckbox.stateChanged.connect(self.onSimulationCheckboxChanged)
        # self.TopLevelLayout.setStyleSheet("background-color: rgb(36,36,36); margin:5px; border:1px solid rgb(255, 255, 255); ")

        self.removeBtn.setIcon(AppConfigs()._icon_minus)

        self.is_running = False
        self.is_simulating = False
        self.needs_to_close = False

        self.locateBtn.clicked.connect(self.on_locate_btn_clicked)
        self.createBtn.clicked.connect(self.on_create_btn_clicked)
        self.runBtn.clicked.connect(self.on_run_btn_clicked)
        self.runBtn.setEnabled(False)
        self.script_process = None

        # _ui elements #####################
        self.stopping_label.setVisible(False)
        self.loading_movie = QMovie(AppConfigs()._icon_load_48px)
        self.stopping_label.setMovie(self.loading_movie)
        self.wait_for_response_worker, self.wait_response_thread = None, None
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

        # global signals
        GlobalSignals().stream_preset_nominal_srate_changed.connect(self.on_stream_nominal_sampling_rate_change)

    def setup_info_worker(self, script_pid):
        self.info_socket_interface = RenaTCPInterface(stream_name='RENA_SCRIPTING_INFO',
                                                      port_id=self.port + 1,
                                                      identity='client',
                                                      pattern='router-dealer', add_poller=True)
        print('MainApp: Sending command info socket routing ID')
        self.info_socket_interface.send_string('Go')  # send an empty message, this is for setting up the routing id
        self.info_worker = workers.ScriptInfoWorker(self.info_socket_interface, script_pid)
        self.info_worker.abnormal_termination_signal.connect(self.kill_script_process)
        self.info_worker.realtime_info_signal.connect(self.show_realtime_info)
        self.info_thread = QThread(
            self.parent)  # set thread to attach to the scriptingtab instead of the widget because it runs a timeout of 2 seconds in the event loop, causing problem when removing the scriptingwidget.
        self.info_worker.moveToThread(self.info_thread)
        self.info_thread.start()

        self.info_timer = QTimer()
        self.info_timer.setInterval(SCRIPTING_UPDATE_REFRESH_INTERVAL)
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
        self.stdout_worker.std_signal.connect(self.redirect_script_std)
        self.stdout_worker.moveToThread(self.stdout_worker_thread)
        self.stdout_worker_thread.start()
        self.stdout_timer = QTimer()
        self.stdout_timer.setInterval(SCRIPTING_UPDATE_REFRESH_INTERVAL)
        self.stdout_timer.timeout.connect(self.stdout_worker.tick_signal.emit)
        self.stdout_timer.start()

    def close_stdout(self):
        self.stdout_timer.stop()
        self.stdout_worker_thread.requestInterruption()
        self.stdout_worker_thread.exit()
        self.stdout_worker_thread.wait()
        del self.stdout_socket_interface
        # del self.stdout_timer, self.stdout_worker, self.stdout_worker_thread

    def close_info_interface(self):
        self.info_timer.stop()
        self.info_worker.deactivate()
        self.info_thread.requestInterruption()
        self.info_thread.exit()
        self.info_thread.wait()

    def redirect_script_std(self, std_message):
        # print('[Script]: ' + stdout_line)
        # if std_message[1] != '\n':
        self.script_console_log.print_msg(*std_message)

    def on_run_btn_clicked(self):
        if not self.is_running:
            script_path = self.scriptPathLineEdit.text()
            if not validate_script_path(script_path, RenaScript): return
            try:
                script_args = self.get_verify_script_args()
            except RenaError as e:
                dialog_popup(str(e), title='Error', main_parent=self.main_window)
                return

            forward_interval = 1e3 / float(self.frequencyLineEdit.text())

            self.script_console_log_window.show()
            self.stdout_socket_interface.send_string('Go')  # send an empty message, this is for setting up the routing id

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
            self.info_worker.deactivate()
            show_label_movie(self.stopping_label, True)
            self.runBtn.setText('Kill')
            self.runBtn.clicked.disconnect()
            self.runBtn.clicked.connect(self.kill_script_process)
            self.notify_script_to_stop()

        # if is_abnormal_termination:
        #     dialog_popup('Script process terminated abnormally.', title='ERROR')
        # if is_timeout_killed:
        #     dialog_popup('Failed to terminate script process within timeout. Killing it', title='ERROR')


    def stop_message_is_available(self):
        """
        this function is connected to the wait for response worker's result available signal
        when stop button is clicked, it calls notify_script_to_stop, which sends a message to the script process,
        and create a wait for response worker to wait for the script process to send back a message to confirm
        when the message is received, this function is called,
        if the response is not right, this function calls kill_script_process to kill the script process

        @return:
        """
        msg = self.command_socket_interface.socket.recv().decode('utf-8')
        if msg != SCRIPT_STOP_SUCCESS:
            self.kill_script_process()
        else:
            self.clean_up_after_stop()

    def kill_script_process(self):
        """
        This function is called from
        * when the stop button is clicked, it calls notify_script_to_stop, which sends a message to the script process,
          but the message received is not the right response to a stop request, so it calls this function to kill the script process
        * when ScriptInfoWorker cannot find the script process, the worker will deactivate itself and emit abnormal_termination_signal
        * when the user click the kill button, it calls this function to kill the script process

        In the later two cases, the info worked needs to be deactivated so it stops checking pid.
        otherwise this function will be called again when the info worker finds the script process missing
        @return:
        """
        self.info_worker.deactivate()
        # stop the wait threads and processes
        self.wait_for_response_worker.stop()
        self.wait_response_thread.requestInterruption()
        self.wait_response_thread.exit()
        self.wait_response_thread.wait()
        if psutil.pid_exists(self.script_pid):
            self.script_process.kill()
        self.clean_up_after_stop()

    def clean_up_after_stop(self):
        """
        this is the final step of stopping the script process
        @return:
        """
        self.close_command_interface()
        self.stop_run_signal_forward_input()
        self.close_info_interface()
        del self.info_socket_interface
        self.script_console_log_window.hide()
        self.is_running = False
        self.change_ui_on_run_stop(self.is_running)
        self.runBtn.clicked.disconnect()
        self.runBtn.clicked.connect(self.on_run_btn_clicked)
        show_label_movie(self.stopping_label, False)
        self.wait_for_response_worker, self.wait_response_thread = None, None
        if self.needs_to_close:  # this is set to true when try_close is called
            self.finish_close()

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
        if script_path == '':
            return
        self.process_locate_script(script_path)
        self.export_script_args_to_settings()

    def process_locate_script(self, script_path):
        if script_path != '':
            if not validate_script_path(script_path, RenaScript):
                self.runBtn.setEnabled(False)
                return
            self.load_script_name(script_path)
            self.runBtn.setEnabled(True)
            print("Selected script path ", script_path)
        else:
            self.runBtn.setEnabled(False)

    def on_create_btn_clicked(self):
        script_path, _ = QtWidgets.QFileDialog.getSaveFileName()
        if script_path == '':
            return
        self.create_script(script_path)

    def create_script(self, script_path, is_open_file=True):
        if script_path:
            base_script_name = os.path.basename(os.path.normpath(script_path))
            this_script: str = AppConfigs()._rena_base_script[:]  # make a copy
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
            raise ValueError('Script path cannot be empty in process_locate_script')
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
        self.process_add_output(output_name, num_channels=1, port_number=self.get_next_available_output_port(), data_type=DataType.float32, interface_type=PresetType.LSL)
        self.export_script_args_to_settings()

    def process_add_output(self, stream_name, num_channels, port_number, data_type, interface_type):
        output_widget = ScriptingOutputWidget(self, stream_name, num_channels, port_number=port_number, data_type=data_type, interface_type=interface_type)
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

    def get_input_shape_dict(self):
        rtn = dict()
        for w in self.input_widgets:
            input_preset_name = w.get_input_name_text()
            rtn[input_preset_name] = self.get_preset_expected_shape(input_preset_name)
        return rtn

    def get_outputs(self):
        return [w.get_label_text() for w in self.output_widgets]

    def get_outputs_num_channels(self):
        return [w.get_num_channels() for w in self.output_widgets]

    def get_output_presets(self) -> List[ScriptOutput]:
        return [w.get_output_preset() for w in self.output_widgets]

    def get_output_ports(self):
        return [w.get_port_number() for w in self.output_widgets]

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
        if not is_stream_name_in_presets(preset_name):
            raise MissingPresetError(preset_name)
        return '[{0}, {1}]'.format(*self.get_preset_expected_shape(preset_name))

    def get_preset_expected_shape(self, preset_name):
        sampling_rate = get_stream_preset_info(preset_name, 'nominal_sampling_rate')
        num_channel = get_stream_preset_info(preset_name, 'num_channels')
        return num_channel, int(int(self.timeWindowLineEdit.text()) * sampling_rate)

    def try_close(self):
        """
        if the script is running, it will call the stop button, which will call on_run_btn_clicked and close routine.
        otherwise it will call finish_close
        @return: None
        """
        if self.is_running:
            if self.wait_for_response_worker is None:  # if is not already stopping the script
                self.on_run_btn_clicked()
            self.needs_to_close = True  # will be used in clean_up_after_stop
        else:
            self.finish_close()

    def finish_close(self):
        self.close_stdout()
        if self.is_popped:
            self.delete_window()
        self.script_console_log_window.close()
        self.deleteLater()
        print('Script widget closed')
        self.parent.remove_script_widget(self)

    def remove_script_clicked(self):
        self.try_close()
        remove_script_from_settings(self.id)

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

        self.wait_for_response_worker, self.wait_response_thread = start_wait_for_response(self.command_socket_interface.socket)
        self.wait_for_response_worker.result_available.connect(self.stop_message_is_available)

    def get_verify_script_args(self):
        buffer_sizes = [(input_name, input_shape[1]) for input_name, input_shape in self.get_input_shape_dict().items()]
        buffer_sizes = dict(buffer_sizes)
        rtn = {'inputs': self.get_inputs(),
                'input_shapes': self.get_input_shape_dict(),
                'buffer_sizes': buffer_sizes,
                'outputs': self.get_output_presets(),
                'params': self.get_param_dict(), 'port': self.stdout_socket_interface.port_id,
                'run_frequency': int(self.frequencyLineEdit.text()),
                'time_window': int(self.timeWindowLineEdit.text()),
                'script_path': self.scriptPathLineEdit.text(),
                'is_simulate': self.simulateCheckbox.isChecked(),
                'presets': Presets()}
        lsl_supported_types = DataType.get_lsl_supported_types()
        lsl_output_data_types = {(o_preset.stream_name, o_preset.data_type) for o_preset in rtn['outputs'] if o_preset.interface_type == PresetType.LSL}
        for output_name, dtype in lsl_output_data_types:
            try:
                assert dtype in lsl_supported_types
            except AssertionError:
                raise UnsupportedLSLDataTypeError(f'{output_name} has unsupported data type {dtype}')
        return rtn

    def export_script_args_to_settings(self):
        script_preset = ScriptPreset(id=self.id, inputs=self.get_inputs(), output_presets=self.get_output_presets(),
                                     param_presets=self.get_params_presets_recursive(),
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
        for output_preset in script_preset.output_presets:
            self.process_add_output(**output_preset.__dict__)

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

    def on_stream_nominal_sampling_rate_change(self, change):
        """
        change what's displayed in the input widgets
        @return:
        """
        if change[0] in self.get_inputs():
            self.update_input_info()

    def get_next_available_output_port(self):
        existing_ports = self.get_output_ports()
        if len(existing_ports) == 0:
            return AppConfigs().output_stream_starting_port
        else:
            return max(existing_ports) + 1