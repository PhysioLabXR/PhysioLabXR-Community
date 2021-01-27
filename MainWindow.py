from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QPushButton, QWidget
import numpy as np

import config
import threadings.workers as workers


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, eeg_interface, unityLSL_inferface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)

        # create sensor threads, worker threads for different sensors
        self.worker_threads = None
        self.workers = None

        # create workers for different sensors
        self.init_sensor_workers_threads(eeg_interface, unityLSL_inferface)

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.REFRESH_INTERVAL)  # for 0.5 KHz refresh rate
        self.timer.timeout.connect(self.ticks)
        self.timer.start()

        # bind buttons
        self.connect_sensor_btn.clicked.connect(self.workers['eeg'].connect)
        self.start_streaming_btn.clicked.connect(self.workers['eeg'].start_stream)
        self.stop_streaming_btn.clicked.connect(self.stop_eeg)
        self.disconnect_sensor_btn.clicked.connect(self.workers['eeg'].disconnect)

        self.unitylsl_connect_sensor_btn.clicked.connect(self.workers['unityLSL'].connect)
        self.unitylsl_start_streaming_btn.clicked.connect(self.workers['unityLSL'].start_stream)
        self.unitylsl_stop_streaming_btn.clicked.connect(self.stop_unityLSL)
        self.unitylsl_disconnect_sensor_btn.clicked.connect(self.workers['unityLSL'].disconnect)

        # bind visualization
        self.eeg_plots = None
        self.init_visualize_eeg_data()
        self.eeg_num_visualized_sample = int(config.OPENBCI_EEG_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)

        self.unityLSL_plots = None
        self.init_visualize_unityLSL_data()
        self.unityLSL_num_visualized_sample = int(config.UNITY_LSL_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)

        self.workers['eeg'].signal_data.connect(self.visualize_eeg_data)
        self.workers['unityLSL'].signal_data.connect(self.visualize_unityLSL_data)

        # data buffers
        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))
        self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))

    def init_sensor_workers_threads(self, eeg_interface, unityLSL_inferface):
        self.worker_threads = {
            'unityLSL': pg.QtCore.QThread(self),
            'eeg': pg.QtCore.QThread(self),
        }
        [w.start() for w in self.worker_threads.values()]  # start all the worker threads

        self.workers = {
            'eeg': workers.EEGWorker(eeg_interface),
            'unityLSL': workers.UnityLSLWorker(unityLSL_inferface)
        }
        self.workers['eeg'].moveToThread(self.worker_threads['eeg'])
        self.workers['unityLSL'].moveToThread(self.worker_threads['unityLSL'])

    def ticks(self):
        """
        ticks every 'refresh' milliseconds
        """
        [w.tick_signal.emit() for w in self.workers.values()]

    def stop_eeg(self):
        self.workers['eeg'].stop_stream()
        # MUST calculate f sample after stream is stopped, for the end time is recorded when calling worker.stop_stream
        f_sample = self.eeg_data_buffer.shape[-1] / (self.workers['eeg'].end_time - self.workers['eeg'].start_time)
        print('MainWindow: Stopped eeg streaming, sampling rate = ' + str(f_sample) + '; Buffer cleared')
        self.init_eeg_buffer()

    def stop_unityLSL(self):
        self.workers['unityLSL'].stop_stream()
        f_sample = self.unityLSL_data_buffer.shape[-1] / (
                self.workers['unityLSL'].end_time - self.workers['unityLSL'].start_time)
        print('MainWindow: Stopped eeg streaming, sampling rate = ' + str(f_sample) + '; Buffer cleared')
        self.init_unityLSL_buffer()

    def init_visualize_eeg_data(self):
        eeg_plot_widgets = [pg.PlotWidget() for i in range(config.OPENBCI_EEG_USEFUL_CHANNELS_NUM)]
        [self.eeg_widget.layout().addWidget(epw) for epw in eeg_plot_widgets]
        self.eeg_plots = [epw.plot([], [], pen=pg.mkPen(color=(0, 0, 255))) for epw in eeg_plot_widgets]

    def init_visualize_unityLSL_data(self):
        unityLSL_plot_widgets = [pg.PlotWidget() for i in range(config.UNITY_LSL_USEFUL_CHANNELS_NUM)]
        [self.unityLSL_widget.layout().addWidget(upw) for upw in unityLSL_plot_widgets]
        self.unityLSL_plots = [upw.plot([], [], pen=pg.mkPen(color=(0, 0, 255))) for upw in unityLSL_plot_widgets]

    def visualize_eeg_data(self, data_dict):
        self.eeg_data_buffer = np.concatenate((self.eeg_data_buffer, data_dict['data']),
                                              axis=-1)  # get all data and remove it from internal buffer
        if self.eeg_data_buffer.shape[-1] < self.eeg_num_visualized_sample:
            eeg_data_to_plot = np.concatenate((np.zeros(shape=(
                config.OPENBCI_EEG_CHANNEL_SIZE, self.eeg_num_visualized_sample - self.eeg_data_buffer.shape[-1])),
                                               self.eeg_data_buffer), axis=-1)
        else:
            eeg_data_to_plot = self.eeg_data_buffer[:,
                               -self.eeg_num_visualized_sample:]  # plot the most recent 10 seconds
        time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, self.eeg_num_visualized_sample)
        eeg_data_to_plot = eeg_data_to_plot[config.OPENBCI_EEG_USEFUL_CHANNELS]  ## keep only the useful channels
        [ep.setData(time_vector, eeg_data_to_plot[i, :]) for i, ep in enumerate(self.eeg_plots)]
        # print('MainWindow: update eeg graphs, eeg_data_buffer shape is ' + str(self.eeg_data_buffer.shape))

    def visualize_unityLSL_data(self, data_dict):
        if len(data_dict['data']) > 0:
            self.unityLSL_data_buffer = np.concatenate((self.unityLSL_data_buffer, data_dict['data']),
                                                       axis=-1)  # get all data and remove it from internal buffer

            if self.unityLSL_data_buffer.shape[-1] < self.unityLSL_num_visualized_sample:
                unityLSL_data_to_plot = np.concatenate((np.zeros(shape=(
                    config.UNITY_LSL_CHANNEL_SIZE,
                    self.unityLSL_num_visualized_sample - self.unityLSL_data_buffer.shape[-1])),
                                                        self.unityLSL_data_buffer), axis=-1)
            else:
                unityLSL_data_to_plot = self.unityLSL_data_buffer[:,
                                        -self.unityLSL_num_visualized_sample:]  # plot the most recent 10 seconds
            time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, self.unityLSL_num_visualized_sample)
            unityLSL_data_to_plot = unityLSL_data_to_plot[config.UNITY_LSL_USEFUL_CHANNELS]  ## keep only the useful channels
            [up.setData(time_vector, unityLSL_data_to_plot[i, :]) for i, up in enumerate(self.unityLSL_plots)]

    def init_eeg_buffer(self):
        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))

    def init_unityLSL_buffer(self):
        self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))
