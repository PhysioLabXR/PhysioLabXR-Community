from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
from PyQt5.QtCore import QTimer

import threadings.workers as workers


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, eeg_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("ui/mainwindow.ui", self)

        # create sensor threads, worker threads for different sensors
        self.worker_threads = None
        self.workers = None

        # create workers for different sensors
        self.init_sensor_workers_threads(eeg_interface)

        # timer
        self.timer = QTimer()
        self.timer.setInterval(2)  # for 0.5 KHz refresh rate
        self.timer.timeout.connect(self.ticks)
        self.timer.start()

    def init_sensor_workers_threads(self, eeg_interface):
        self.worker_threads = {
            'eye': pg.QtCore.QThread(self),
            'eeg': pg.QtCore.QThread(self),
        }
        [w.start() for w in self.worker_threads.values()]  # start all the worker threads

        self.workers = {
            'eeg': workers.EEGWorker(eeg_interface),
        }
        self.workers['eeg'].moveToThread(self.worker_threads['eeg'])

    def ticks(self):
        """
        ticks every 'refresh' milliseconds
        """
        [w.tick_signal.emit() for w in self.workers.values()]