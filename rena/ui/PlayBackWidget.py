from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg

from rena.threadings.workers import PlaybackWorker
from rena.ui_shared import start_stream_icon, stop_stream_icon


class PlayBackWidget(QtWidgets.QWidget):
    playback_signal = pyqtSignal(int)
    play_pause_signal = pyqtSignal(bool)
    stop_signal = pyqtSignal()

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
        # Initialize replay worker
        self.lsl_replay_thread = pg.QtCore.QThread(self.parent)
        self.lsl_replay_worker = PlaybackWorker(self.command_info_interface)
        self.lsl_replay_worker.moveToThread(self.lsl_replay_thread)
        self.lsl_replay_thread.started.connect(self.lsl_replay_worker.run)
        self.lsl_replay_thread.start()

        self.lsl_replay_worker.replay_progress_signal.connect(self.update_playback_position)

    def start_replay(self):
        self.lsl_replay_worker.stop = False

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

    def update_playback_position(self, replay_progress):
        # print("slider value is being updated ", replay_progress)
        self.horizontalSlider.setValue(replay_progress)
