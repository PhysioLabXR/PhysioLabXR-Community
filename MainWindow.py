from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QPushButton, QWidget
import numpy as np

import config
import threadings.workers as workers
from interfaces.OpenBCIInterface import OpenBCIInterface
from interfaces.UnityLSLInterface import UnityLSLInterface
from utils.data_utils import window_slice
from utils.ui_utiles import init_sensor_widget


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, eeg_interface, unityLSL_inferface, inference_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)

        # create sensor threads, worker threads for different sensors
        self.worker_threads = {}
        self.sensor_workers = {}
        self.inference_worker = None



        # create workers for different sensors
        # self.init_sensor_workers_threads(eeg_interface, unityLSL_inferface, inference_interface)

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.REFRESH_INTERVAL)  # for 250 KHz refresh rate
        self.timer.timeout.connect(self.ticks)
        self.timer.start()

        # inference timer
        self.inference_timer = QTimer()
        self.inference_timer.setInterval(config.INFERENCE_REFRESH_INTERVAL)  # for 5 KHz refresh rate
        self.inference_timer.timeout.connect(self.inference_ticks)
        self.inference_timer.start()

        # bind buttons
        # self.start_streaming_btn.clicked.connect(self.sensor_workers['eeg'].start_stream)
        # self.stop_streaming_btn.clicked.connect(self.stop_eeg)

        # self.unitylsl_start_streaming_btn.clicked.connect(self.sensor_workers['unityLSL'].start_stream)
        # self.unitylsl_stop_streaming_btn.clicked.connect(self.stop_unityLSL)

        # self.connect_inference_btn.clicked.connect(self.inference_worker.connect)
        # self.disconnect_inference_btn.clicked.connect(self.inference_worker.disconnect)

        # bind visualization
        self.eeg_plots = None
        # self.init_visualize_eeg_data()
        self.eeg_num_visualized_sample = int(config.OPENBCI_EEG_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)
        #
        # self.unityLSL_plots = None
        # self.init_visualize_unityLSL_data()
        self.unityLSL_num_visualized_sample = int(config.UNITY_LSL_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)
        #
        # self.inference_results_plots = None
        # self.init_visualize_inference_results()
        # self.inference_num_visualized_results = int(
        #     config.PLOT_RETAIN_HISTORY * 1 / (1e-3 * config.INFERENCE_REFRESH_INTERVAL))

        # self.sensor_workers['eeg'].signal_data.connect(self.visualize_eeg_data)
        # self.sensor_workers['unityLSL'].signal_data.connect(self.visualize_unityLSL_data)
        # self.inference_worker.signal_inference_results.connect(self.visualize_inference_results)

        # TESTING
        self.init_sensor(sensor_type=config.sensors[0])
        self.init_sensor(sensor_type=config.sensors[1])

        # data buffers
        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))
        self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))

        # inference buffer
        self.inference_buffer = np.empty(shape=(0, config.INFERENCE_CLASS_NUM))  # time axis is the first

    def init_sensor(self, sensor_type):
        sensor_layout, start_stream_btn, stop_stream_btn= init_sensor_widget(parent=self.sensorTabSensorsHorizontalLayout, sensor_type=sensor_type)
        worker_thread = pg.QtCore.QThread(self)
        self.worker_threads[sensor_type] = worker_thread

        if sensor_type == config.sensors[0]:
            interface = OpenBCIInterface()
            self.sensor_workers[sensor_type] = workers.EEGWorker(interface)
            stop_stream_btn.clicked.connect(self.stop_eeg)
            self.init_visualize_eeg_data(parent=sensor_layout)
            self.sensor_workers[sensor_type].signal_data.connect(self.visualize_eeg_data)
        elif sensor_type == config.sensors[1]:
            interface = UnityLSLInterface()
            self.sensor_workers[sensor_type] = workers.UnityLSLWorker(interface)
            stop_stream_btn.clicked.connect(self.stop_unityLSL)
            self.init_visualize_unityLSL_data(parent=sensor_layout)
            self.sensor_workers[sensor_type].signal_data.connect(self.visualize_unityLSL_data)

        self.sensor_workers[sensor_type].moveToThread(self.worker_threads[sensor_type])
        start_stream_btn.clicked.connect(self.sensor_workers[sensor_type].start_stream)

        worker_thread.start()
        pass

    def init_sensor_workers_threads(self, eeg_interface, unityLSL_inferface, inference_interface):
        self.worker_threads = {
            config.sensors[1]: pg.QtCore.QThread(self),
            config.sensors[0]: pg.QtCore.QThread(self),
            'inference': pg.QtCore.QThread(self),
        }
        [w.start() for w in self.worker_threads.values()]  # start all the worker threads

        self.sensor_workers = {
            config.sensors[0]: workers.EEGWorker(eeg_interface),
            config.sensors[1]: workers.UnityLSLWorker(unityLSL_inferface)
        }
        self.sensor_workers[config.sensors[0]].moveToThread(self.worker_threads[config.sensors[0]])
        self.sensor_workers[config.sensors[1]].moveToThread(self.worker_threads[config.sensors[1]])

        self.inference_worker = workers.InferenceWorker(inference_interface)
        self.inference_worker.moveToThread(self.worker_threads['inference'])

    def ticks(self):
        """
        ticks every 'refresh' milliseconds
        """
        # pass
        [w.tick_signal.emit() for w in self.sensor_workers.values()]

    def inference_ticks(self):
        # only ticks if data is streaming
        if config.sensors[1] in self.sensor_workers.keys() and self.inference_worker:
            if self.sensor_workers[config.sensors[1]].is_streaming:
                if self.unityLSL_data_buffer.shape[-1] < config.EYE_INFERENCE_TOTAL_TIMESTEPS:
                    eye_frames = np.concatenate((np.zeros(shape=(
                        2,  # 2 for two eyes' pupil sizes
                        config.EYE_INFERENCE_TOTAL_TIMESTEPS - self.unityLSL_data_buffer.shape[-1])),
                                                 self.unityLSL_data_buffer[1:3, :]), axis=-1)
                else:
                    eye_frames = self.unityLSL_data_buffer[1:3,
                                 -config.EYE_INFERENCE_TOTAL_TIMESTEPS:]
                # make samples out of the most recent data
                eye_samples = window_slice(eye_frames, window_size=config.EYE_INFERENCE_WINDOW_TIMESTEPS,
                                           stride=config.EYE_WINDOW_STRIDE_TIMESTEMPS, channel_mode='channel_first')

                samples_dict = {'eye': eye_samples}
                self.inference_worker.tick_signal.emit(samples_dict)

    def stop_eeg(self):
        self.sensor_workers[config.sensors[0]].stop_stream()
        # MUST calculate f sample after stream is stopped, for the end time is recorded when calling worker.stop_stream
        f_sample = self.eeg_data_buffer.shape[-1] / (self.sensor_workers[config.sensors[0]].end_time - self.sensor_workers[config.sensors[0]].start_time)
        print('MainWindow: Stopped eeg streaming, sampling rate = ' + str(f_sample) + '; Buffer cleared')
        self.init_eeg_buffer()

    def stop_unityLSL(self):
        self.sensor_workers[config.sensors[1]].stop_stream()
        f_sample = self.unityLSL_data_buffer.shape[-1] / (
                self.sensor_workers[config.sensors[1]].end_time - self.sensor_workers[config.sensors[1]].start_time)
        print('MainWindow: Stopped eeg streaming, sampling rate = ' + str(f_sample) + '; Buffer cleared')
        self.init_unityLSL_buffer()

    def init_visualize_eeg_data(self, parent):
        eeg_plot_widgets = [pg.PlotWidget() for i in range(config.OPENBCI_EEG_USEFUL_CHANNELS_NUM)]
        [parent.addWidget(epw) for epw in eeg_plot_widgets]
        self.eeg_plots = [epw.plot([], [], pen=pg.mkPen(color=(0, 0, 255))) for epw in eeg_plot_widgets]

    def init_visualize_unityLSL_data(self, parent):
        unityLSL_plot_widgets = [pg.PlotWidget() for i in range(config.UNITY_LSL_USEFUL_CHANNELS_NUM)]
        [parent.addWidget(upw) for upw in unityLSL_plot_widgets]
        self.unityLSL_plots = [upw.plot([], [], pen=pg.mkPen(color=(255, 0, 0))) for upw in unityLSL_plot_widgets]

    def init_visualize_inference_results(self):
        inference_results_plot_widgets = [pg.PlotWidget() for i in range(config.INFERENCE_CLASS_NUM)]
        [self.inference_widget.layout().addWidget(pw) for pw in inference_results_plot_widgets]
        self.inference_results_plots = [pw.plot([], [], pen=pg.mkPen(color=(0, 255, 255))) for pw in
                                        inference_results_plot_widgets]

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
            unityLSL_data_to_plot = unityLSL_data_to_plot[
                config.UNITY_LSL_USEFUL_CHANNELS]  ## keep only the useful channels
            [up.setData(time_vector, unityLSL_data_to_plot[i, :]) for i, up in enumerate(self.unityLSL_plots)]

    def visualize_inference_results(self, inference_results):
        # results will be -1 if inference is not connected
        if self.inference_worker.is_connected and inference_results[0][0] >= 0:
            self.inference_buffer = np.concatenate([self.inference_buffer, inference_results], axis=0)

            if self.inference_buffer.shape[0] < self.inference_num_visualized_results:
                data_to_plot = np.concatenate((np.zeros(shape=(
                    self.inference_num_visualized_results - self.inference_buffer.shape[0], config.INFERENCE_CLASS_NUM)),
                                               self.inference_buffer), axis=0)  # zero padding
            else:
                # plot the most recent 10 seconds
                data_to_plot = self.inference_buffer[-self.inference_num_visualized_results:, :]
            time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, self.inference_num_visualized_results)
            [p.setData(time_vector, data_to_plot[:, i]) for i, p in enumerate(self.inference_results_plots)]

    def init_eeg_buffer(self):
        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))

    def init_unityLSL_buffer(self):
        self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))
