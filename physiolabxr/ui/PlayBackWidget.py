import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer, QThread

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.threadings.workers import PlaybackWorker


class PlayBackWidget(QtWidgets.QWidget):

    def __init__(self, parent, command_info_interface):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_PlayBackWidget, self)
        self.parent = parent
        self.command_info_interface = command_info_interface
        self.slider_pressed_position = None

        # playback status
        # self.horizontalSlider.valueChanged.connect(self.emit_playback_position)
        self.playPauseButton.clicked.connect(self.issue_play_pause_command)
        self.stopButton.clicked.connect(self.start_stop_replay)

        self.slider_is_dragging = False
        self.horizontalSlider.sliderPressed.connect(self.slider_pressed)
        self.horizontalSlider.sliderReleased.connect(self.slider_released)
        # self.horizontalSlider.sliderMoved.connect(self.issue_slider_moved_command)

        # create worker listening the playback position from the server
        # Initialize playback worker
        self.playback_command_interface_timer = QTimer()
        self.playback_command_interface_timer.setInterval(int(float(AppConfigs().visualization_refresh_interval)))
        self.playback_command_interface_timer.timeout.connect(self.ticks)

        self.playback_thread = QThread(self.parent)
        self.playback_worker = PlaybackWorker(self.command_info_interface)
        self.playback_worker.moveToThread(self.playback_thread)
        self.playback_worker.replay_progress_signal.connect(self.update_playback_position)
        self.playback_worker.replay_stopped_signal.connect(self.replay_stopped_signal_callback)
        self.playback_worker.replay_terminated_signal.connect(self.replay_terminated_signal_callback)
        self.playback_worker.replay_play_pause_signal.connect(self.replay_play_pause_signal_callback)
        self.playback_thread.start()

        self.start_time, self.end_time, self.total_time, self.virtual_clock_offset = [None] * 4

    def start_replay(self, start_time, end_time, total_time, virtual_clock_offset):
        self.start_time, self.end_time, self.total_time, self.virtual_clock_offset = start_time, end_time, total_time, virtual_clock_offset
        self.playPauseButton.setIcon(AppConfigs()._icon_pause)
        self.stopButton.setIcon(AppConfigs()._icon_terminate)
        self.playPauseButton.setEnabled(True)
        self.playback_worker.start_run()
        self.playback_command_interface_timer.start()  # timer should stop when the replay is paused, over, or stopped

    def virtual_time_to_playback_position_value(self, virtual_clock):
        # TODO: do not hardcode playback range (100)
        return (virtual_clock - self.start_time) * 100 / self.total_time

    def update_playback_position(self, virtual_clock):
        """
        Callback function for replay_progress_signal emitted from playback_worker.
        Update horizontalSlider's playback position if the slider is not being dragged.
        """
        if not self.slider_is_dragging:
            playback_percent = self.virtual_time_to_playback_position_value(virtual_clock)
            self.horizontalSlider.setValue(round(playback_percent))
            self.currentTimestamplabel.setText('{:.2f}'.format(virtual_clock + self.virtual_clock_offset))
            self.timeSinceStartedLabel.setText('{:.2f}/{:.2f}'.format(virtual_clock - self.start_time, self.total_time))
            self.percentageReplayedLabel.setText('{:.1f} %'.format(playback_percent))
            # print('Virtual Clock {0}'.format(virtual_clock))
            # print('Time since start {0}/{1}'.format(virtual_clock - self.start_time, self.total_time))
            # print('Playback percent {0}'.format(playback_percent))

    def start_stop_replay(self):
        """
        Called when stopButton is clicked.
        Replay will initiate or terminate depending on the `is_replaying` status of ReplayTab (parent).
        """
        if self.parent.is_replaying:
            self.parent.start_stop_replay_btn_pressed()
            if not self.parent.is_replaying:
                self.stopButton.setIcon(AppConfigs()._icon_start)
                self.playPauseButton.setEnabled(False)
        else:
            self.parent.start_stop_replay_btn_pressed()
            if self.parent.is_replaying:
                self.stopButton.setIcon(AppConfigs()._icon_terminate)
                self.playPauseButton.setEnabled(True)
            # self.stopButton.setIcon(stop_stream_icon)

    def ticks(self):
        self.playback_worker.playback_tick_signal.emit()

    def pause_replay(self):
        '''
        called when pause command is executed.
        add any playback widget specific clean up steps here when replay is paused.
        '''
        self.playPauseButton.setIcon(AppConfigs()._icon_start)

    def resume_replay(self):
        '''
        called when resume command is executed.
        add any playback widget specific set-up steps here when replay is resumed.
        '''
        self.playPauseButton.setIcon(AppConfigs()._icon_pause)

    def slider_pressed(self):
        """
        called when the user starts dragging horizontalSlider.
        """
        self.slider_pressed_position = self.horizontalSlider.value()
        self.slider_is_dragging = True

    def slider_released(self):
        """
        Called when the user stops dragging horizontalSlider.
        Makes a function call to issue_slider_moved_command() so that replay can be updated to the new playback position.
        """
        self.slider_is_dragging = False
        playback_position = (self.horizontalSlider.value() + 1) * 1e-2
        set_to_time = self.total_time * playback_position
        before_press_time = self.total_time * (self.slider_pressed_position + 1) * 1e-2

        slider_offset_time = set_to_time - before_press_time
        self.issue_slider_moved_command(np.array([set_to_time, slider_offset_time]))

    # def slider_moved(self):
    #     """
    #     called when the position of horizontalSlider is changed through dragging.
    #     """
    #     self.slider_

    def replay_play_pause_signal_callback(self, play_pause_command):
        # relay the signal to the parent (replay tab) and then use that information in a method in replay tab
        if play_pause_command == 'pause':
            self.pause_replay()
        else:  # play_pause_command is 'resume':
            self.resume_replay()

    def replay_stopped_signal_callback(self):
        '''
        called when received 'replay stopped successful' message from replay server
        :return:
        '''
        self.reset_playback()
        self.parent.replay_successfully_stopped()
        self.playback_command_interface_timer.stop()

    def replay_terminated_signal_callback(self):
        self.playback_command_interface_timer.stop()
        self.playback_thread.exit()
        del self.command_info_interface  # close the socket

    def reset_playback(self):
        self.horizontalSlider.setValue(0)
        self.currentTimestamplabel.setText('')
        self.timeSinceStartedLabel.setText('')
        self.percentageReplayedLabel.setText('')
        self.stopButton.setIcon(AppConfigs()._icon_start)
        self.playPauseButton.setEnabled(False)
        self.playPauseButton.setIcon(AppConfigs()._icon_pause)

    def issue_play_pause_command(self):
        # prevent is_paused status from changing when the replay is not running
        if self.parent.is_replaying:
            self.playback_worker.queue_play_pause_command()

    def issue_stop_command(self):
        self.playback_worker.queue_stop_command()

    def issue_terminate_command(self):
        self.playback_worker.queue_terminate_command()

    def issue_slider_moved_command(self, command):
        self.playback_worker.queue_slider_moved_command(command)

    def try_close(self):
        # Initialize playback worker
        # print("PlayBackWidget: stopping timer")
        self.playback_command_interface_timer.stop()
        # print("PlayBackWidget: timer stopped, requesting interruption to playback thread")
        self.playback_thread.requestInterruption()
        # print("PlayBackWidget: interruption requested, exiting playback thread")
        self.playback_thread.exit()
        # print("PlayBackWidget: playback thread exited, waiting for playback thread to finish")
        self.playback_thread.wait(2000)
        # terminate the thread if it is still running
        # if self.playback_thread.isRunning():
        #     print("PlayBackWidget: playback thread is still running, terminating playback thread")
        #     self.playback_thread.terminate()
        #     print("PlayBackWidget: playback thread terminated")
        # print("PlayBackWidget: playback thread finished, deleting playback widget")
        self.deleteLater()
        # print("PlaybackWidget try close finished")