from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal, QTimer
import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel

from rena import config
from rena.threadings.workers import PlaybackWorker
from rena.ui_shared import start_stream_icon, stop_stream_icon, pause_icon


class PlayBackWidget(QtWidgets.QWidget):

    def __init__(self, parent, command_info_interface):
        super().__init__()
        self.ui = uic.loadUi("ui/PlayBackWidget.ui", self)
        self.parent = parent
        self.command_info_interface = command_info_interface

        # playback status
        # self.horizontalSlider.valueChanged.connect(self.emit_playback_position)
        # self.playPauseButton.clicked.connect(self.emit_play_pause_button_clicked)
        # self.stopButton.clicked.connect(self.emit_playback_stop)

        # create worker listening the playback position from the server
        # Initialize playback worker
        self.timer = QTimer()
        self.timer.setInterval(config.VISUALIZATION_REFRESH_INTERVAL)
        self.timer.timeout.connect(self.ticks)

        self.playback_thread = pg.QtCore.QThread(self.parent)
        self.playback_worker = PlaybackWorker(self.command_info_interface)
        self.playback_worker.moveToThread(self.playback_thread)
        self.playback_worker.replay_progress_signal.connect(self.update_playback_position)
        self.playback_worker.replay_stopped_signal.connect(self.replay_stopped_signal_callback)
        self.playback_thread.start()

        self.start_time, self.end_time, self.total_time, self.virtual_clock_offset = [None] * 4

        # start the play pause button

    def start_replay(self, start_time, end_time, total_time, virtual_clock_offset):
        self.start_time, self.end_time, self.total_time, self.virtual_clock_offset = start_time, end_time, total_time, virtual_clock_offset
        self.playPauseButton.setIcon(pause_icon)
        self.playback_worker.start_run()
        self.timer.start()  # timer should stop when the replay is paused, over, or stopped

    def play_pause_button_clicked(self):
        # TODO add play pause feature
        pass

    def virtual_time_to_playback_position_value(self, virtual_clock):
        # TODO: do not hardcode playback range (100)
        return (virtual_clock - self.start_time) * 100 / self.total_time

    def update_playback_position(self, virtual_clock):
        # print("slider value is being updated ", replay_progress)
        playback_percent = self.virtual_time_to_playback_position_value(virtual_clock)
        self.horizontalSlider.setValue(playback_percent)
        self.currentTimestamplabel.setText('{:.2f}'.format(virtual_clock + self.virtual_clock_offset))
        self.timeSinceStartedLabel.setText('{:.2f}/{:.2f}'.format(virtual_clock - self.start_time, self.total_time))
        self.percentageReplayedLabel.setText('{:.1f} %'.format(playback_percent))
        # print('Virtual Clock {0}'.format(virtual_clock))
        # print('Time since start {0}/{1}'.format(virtual_clock - self.start_time, self.total_time))
        # print('Playback percent {0}'.format(playback_percent))

    # def emit_play_pause_button_clicked(self):
    #     print("Its clicked in playbackwidget")
    #     if not self.parent.is_replaying:  # set in reverse
    #         self.playPauseButton.setIcon(stop_stream_icon)
    #         # self.playPauseButton.setIconSize(QtCore.QSize(100, 100))
    #     else:
    #         self.playPauseButton.setIcon(start_stream_icon)
    #         # self.playPauseButton.setIconSize(QtCore.QSize(100, 100))
    #     self.play_pause_signal.emit(self.is_playing)
    #
    # def emit_playback_stop(self):
    #     # self.playing = False
    #     # self.parent.stop_replay_btn_pressed()
    #     self.stop_signal.emit()
    #
    # def emit_playback_position(self, event):
    #     # use signal
    #     self.playback_signal.emit(event)

    def ticks(self):
        self.playback_worker.playback_tick_signal.emit()

    def replay_stopped_signal_callback(self):
        '''
        called when received 'replay stopped successful' message from replay server
        :return:
        '''
        self.reset_playback()
        self.parent.replay_successfully_stopped()
        self.timer.stop()

    def reset_playback(self):
        self.horizontalSlider.setValue(0)
        self.currentTimestamplabel.setText('')
        self.timeSinceStartedLabel.setText('')
        self.percentageReplayedLabel.setText('')

    def issue_stop_command(self):
        self.playback_worker.send_stop_command()
