# This Python file uses the following encoding: utf-8
from multiprocessing import Process

import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog

from rena import config, shared
from rena.sub_process.ReplayServer import start_replay_server
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.ui.PlayBackWidget import PlayBackWidget
from rena.utils.ui_utils import another_window
from rena.utils.ui_utils import dialog_popup


class ReplayTab(QtWidgets.QWidget):
    playback_position_signal = pyqtSignal(int)
    play_pause_signal = pyqtSignal(bool)

    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/ReplayTab.ui", self)
        self.is_replaying = False

        self.StartStopReplayBtn.clicked.connect(self.start_stop_replay_btn_pressed)
        self.SelectDataDirBtn.clicked.connect(self.select_data_dir_btn_pressed)

        self.StartStopReplayBtn.setEnabled(False)
        self.parent = parent

        self.file_loc = config.DEFAULT_DATA_DIR
        self.ReplayFileLoc.setText('')
        self.stream_names = []

        # start replay client
        self.command_info_interface = RenaTCPInterface(stream_name='RENA_REPLAY',
                                                       port_id=config.replay_port,
                                                       identity='client',
                                                       pattern='router-dealer')
        self._create_playback_widget()

        # self.replay_timer = QTimer()
        # self.replay_timer.setInterval(config.REFRESH_INTERVAL)
        # self.replay_timer.timeout.connect(self.ticks)

        self.replay_server_process = Process(target=start_replay_server)
        self.replay_server_process.start()

    def _create_playback_widget(self):
        self._init_playback_widget()
        # open in a separate window
        # window = AnotherWindow(self.playback_widget, self.stop_replay_btn_pressed)
        self.playback_window = another_window('Playback')
        self.playback_window.get_layout().addWidget(self.playback_widget)
        # self.playback_window.setFixedWidth(620)
        # self.playback_window.setFixedHeight(300)
        self.playback_window.hide()

    def _init_playback_widget(self):
        self.playback_widget = PlayBackWidget(self, self.command_info_interface)
        # self.playback_widget.playback_signal.connect(self.on_playback_slider_changed)
        # self.playback_widget.play_pause_signal.connect(self.on_play_pause_toggle)
        # self.playback_widget.stop_signal.connect(self.stop_replay_btn_pressed)

        # connect signal emitted from the replayworker to playback widget
        # self.lsl_replay_worker.replay_progress_signal.connect(self.playback_widget.on_replay_tick)

    def select_data_dir_btn_pressed(self):
        selected_file = QFileDialog.getOpenFileName(self.widget_3, "Select File")[0]
        self.select_file(selected_file)

    def select_file(self, selected_file):
        if selected_file != '':
            self.file_loc = selected_file

        # self.file_loc = self.file_loc + 'data.dats'

        print("Selected file: ", self.file_loc)
        self.ReplayFileLoc.setText(self.file_loc + '/')
        self.StartStopReplayBtn.setEnabled(True)

    def start_stop_replay_btn_pressed(self):
        """
        callback function when start_stop button is pressed.
        """
        if not self.is_replaying:
            print('Sending start command with file location to ReplayClient')  # TODO change the send to a progress bar
            self.command_info_interface.send_string(shared.START_COMMAND + self.file_loc)
            client_info = self.command_info_interface.recv_string()
            if client_info.startswith(shared.FAIL_INFO):
                dialog_popup(client_info.strip(shared.FAIL_INFO), title="ERROR")
            elif client_info.startswith(shared.START_SUCCESS_INFO):
                time_info = self.command_info_interface.socket.recv()
                start_time, end_time, total_time, virtual_clock_offset = np.frombuffer(time_info)
                self.playback_window.show()
                self.playback_window.activateWindow()
                self.playback_widget.start_replay(start_time, end_time, total_time, virtual_clock_offset)
                self.stream_names = self.command_info_interface.recv_string().split('|')
                self.parent.add_streams_from_replay(self.stream_names)
                print('Received replay starts successfully from ReplayClient')  # TODO change the send to a progress bar
                self.is_replaying = True
                self.StartStopReplayBtn.setText('Stop Replay')
            else:
                raise ValueError("ReplayTab.start_replay_btn_pressed: unsupported info from ReplayClient: " + client_info)
        else:
            self.stop_replay_btn_pressed()  # it is not known if the replay has successfully stopped yet
            self.playback_window.hide()
            self.replay_successfully_stopped()

    def stop_replay_btn_pressed(self):
        self.playback_widget.issue_stop_command()

    # def replay_successfully_paused(self):
        # self.StartStopReplayBtn.setText('Resume Replay')
        # self.is_replaying = False

    def replay_successfully_stopped(self):
        self.StartStopReplayBtn.setText('Start Replay')
        self.is_replaying = False

    # def replay_successfully_resumed(self):
        # self.StartStopReplayBtn.setText('Pause Replay')
        # self.is_replaying = False

    def openWindow(self):
        self.window = QtWidgets.QMainWindow()

    def try_close(self):
        self.playback_widget.issue_terminate_command()
        self.replay_server_process.join(timeout=1)
        if self.replay_server_process.is_alive():
            self.replay_server_process.kill()
        return True

    # def ticks(self):
    #     self.lsl_replay_worker.tick_signal.emit()

    # def on_play_pause_toggle(self):
    #     self.is_replaying = not self.is_replaying
    #     self.play_pause_signal.emit(self.is_replaying)

    def on_playback_slider_changed(self, new_playback_position):
        print("adjust playback position to:", new_playback_position)
        self.playback_position_signal.emit(new_playback_position)

    def get_num_replay_channels(self):
        return len(self.stream_names)