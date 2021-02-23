import time

from PyQt5 import QtWidgets, uic, sip
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QPushButton, QWidget
import numpy as np

import config
import config_ui
import threadings.workers as workers
from interfaces.LSLInletInterface import LSLInletInterface
from interfaces.OpenBCIInterface import OpenBCIInterface
from interfaces.UnityLSLInterface import UnityLSLInterface
from ui.RecordingsTab import RecordingsTab
from utils.data_utils import window_slice
from utils.ui_utils import init_sensor_or_lsl_widget, init_add_widget, CustomDialog, init_button, dialog_popup


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, inference_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)

        # create sensor threads, worker threads for different sensors
        self.worker_threads = {}
        self.sensor_workers = {}
        self.lsl_workers = {}
        self.inference_worker = None

        # create workers for different sensors
        self.init_inference(inference_interface)

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

        # bind visualization
        self.eeg_num_visualized_sample = int(config.OPENBCI_EEG_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)
        self.unityLSL_num_visualized_sample = int(config.UNITY_LSL_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)

        self.LSLInlet_num_visualized_sample = int(60 * config.PLOT_RETAIN_HISTORY)  # TODO have user input sampling rate

        self.inference_num_visualized_results = int(
            config.PLOT_RETAIN_HISTORY * 1 / (1e-3 * config.INFERENCE_REFRESH_INTERVAL))
        # TESTING
        self.add_sensor_layout, self.sensor_combo_box, self.add_sensor_btn, \
        self.lsl_data_type_input, self.lsl_num_chan_input, self.add_lsl_btn = init_add_widget(
            parent=self.sensorTabSensorsHorizontalLayout)
        self.add_sensor_btn.clicked.connect(self.add_sensor_clicked)
        self.add_lsl_btn.clicked.connect(self.add_lsl_clicked)

        # data buffers
        self.LSL_plots_dict = {}
        self.LSL_data_buffer_dicts = {}

        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))
        self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))

        # inference buffer
        self.inference_buffer = np.empty(shape=(0, config.INFERENCE_CLASS_NUM))  # time axis is the first

        # add other tabs
        self.recordingTab = RecordingsTab(self, self.LSL_data_buffer_dicts)
        self.recordings_tab_vertical_layout.addWidget(self.recordingTab)

    def add_sensor_clicked(self):
        sensor_type = config_ui.sensor_ui_name_type_dict[str(self.sensor_combo_box.currentText())]
        if sensor_type not in self.sensor_workers.keys():
            self.init_sensor(sensor_type=config_ui.sensor_ui_name_type_dict[str(self.sensor_combo_box.currentText())])
        else:
            msg = 'MainWindow: sensor type ' + sensor_type + ' is already added.'
            dlg = CustomDialog(msg)  # If you pass self, the dialog will be centered over the main window as before.
            if dlg.exec_():
                print("Success!")
            else:
                print("Cancel!")

    def add_lsl_clicked(self):
        lsl_data_type = self.lsl_data_type_input.text()
        lsl_num_chan = int(self.lsl_num_chan_input.text())

        if lsl_data_type not in self.lsl_workers.keys():
            self.init_lsl(lsl_data_type, lsl_num_chan)
        else:
            msg = 'MainWindow: LSL inlet with data type ' + lsl_data_type + ' is already added.'
            dlg = CustomDialog(msg)  # If you pass self, the dialog will be centered over the main window as before.
            if dlg.exec_():
                print("Success!")
            else:
                print("Cancel!")

    def init_lsl(self, lsl_data_type, lsl_num_chan):

        try:
            interface = LSLInletInterface(lsl_data_type, lsl_num_chan)
        except AttributeError:
            dialog_popup('Unable to find LSL Stream with given type {0}.'.format(lsl_data_type))
            return
        self.lsl_workers[lsl_data_type] = workers.LSLInletWorker(interface)
        lsl_widget_name = lsl_data_type + '_widget'
        lsl_widget, lsl_layout, start_stream_btn, stop_stream_btn = init_sensor_or_lsl_widget(
            parent=self.sensorTabSensorsHorizontalLayout, label_string=lsl_data_type,
            insert_position=self.sensorTabSensorsHorizontalLayout.count() - 1)
        lsl_widget.setObjectName(lsl_widget_name)
        worker_thread = pg.QtCore.QThread(self)
        self.worker_threads[lsl_data_type] = worker_thread


        stop_stream_btn.clicked.connect(self.lsl_workers[lsl_data_type].stop_stream)
        self.LSL_plots_dict[lsl_data_type] = self.init_visualize_LSLInlet_data(parent=lsl_layout, num_chan=lsl_num_chan)
        self.lsl_workers[lsl_data_type].signal_data.connect(self.process_visualize_LSLInlet_data)
        self.LSL_data_buffer_dicts[lsl_data_type] = np.empty(shape=(lsl_num_chan, 0))

        def remove_lsl():
            # fire stop streaming first
            stop_stream_btn.click()
            worker_thread.exit()
            self.lsl_workers.pop(lsl_data_type)
            self.worker_threads.pop(lsl_data_type)
            self.sensorTabSensorsHorizontalLayout.removeWidget(lsl_widget)
            sip.delete(lsl_widget)
            self.LSL_data_buffer_dicts.pop(lsl_data_type)

        #     worker_thread
        init_button(parent=lsl_layout, label='Remove Inlet',
                    function=remove_lsl)  # add delete sensor button after adding visualization
        self.lsl_workers[lsl_data_type].moveToThread(self.worker_threads[lsl_data_type])
        start_stream_btn.clicked.connect(self.lsl_workers[lsl_data_type].start_stream)
        worker_thread.start()

    def init_sensor(self, sensor_type):
        sensor_widget_name = sensor_type + '_widget'
        sensor_widget, sensor_layout, start_stream_btn, stop_stream_btn = init_sensor_or_lsl_widget(
            parent=self.sensorTabSensorsHorizontalLayout, label_string=sensor_type,
            insert_position=self.sensorTabSensorsHorizontalLayout.count() - 1)
        sensor_widget.setObjectName(sensor_widget_name)
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

        def remove_sensor():
            # fire stop streaming first
            stop_stream_btn.click()
            worker_thread.exit()
            self.sensor_workers.pop(sensor_type)
            self.worker_threads.pop(sensor_type)
            self.sensorTabSensorsHorizontalLayout.removeWidget(sensor_widget)
            sip.delete(sensor_widget)
            # sensor_widget = None

        #     worker_thread
        init_button(parent=sensor_layout, label='Remove Sensor',
                    function=remove_sensor)  # add delete sensor button after adding visualization
        self.sensor_workers[sensor_type].moveToThread(self.worker_threads[sensor_type])
        start_stream_btn.clicked.connect(self.sensor_workers[sensor_type].start_stream)

        worker_thread.start()
        pass

    def init_inference(self, inference_interface):
        inference_thread = pg.QtCore.QThread(self)
        self.worker_threads['inference'] = inference_thread
        self.inference_worker = workers.InferenceWorker(inference_interface)
        self.inference_worker.moveToThread(self.worker_threads['inference'])
        self.init_visualize_inference_results()
        self.inference_worker.signal_inference_results.connect(self.visualize_inference_results)

        self.connect_inference_btn.clicked.connect(self.inference_worker.connect)
        self.disconnect_inference_btn.clicked.connect(self.inference_worker.disconnect)

        inference_thread.start()

    def ticks(self):
        """
        ticks every 'refresh' milliseconds
        """
        # pass
        [w.tick_signal.emit() for w in self.sensor_workers.values()]
        [w.tick_signal.emit() for w in self.lsl_workers.values()]


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
        f_sample = self.eeg_data_buffer.shape[-1] / (
                self.sensor_workers[config.sensors[0]].end_time - self.sensor_workers[config.sensors[0]].start_time)
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
        self.eeg_plots = [epw.plot([], [], pen=pg.mkPen(color=(255, 255, 255))) for epw in eeg_plot_widgets]

    def init_visualize_unityLSL_data(self, parent):
        unityLSL_plot_widgets = [pg.PlotWidget() for i in range(config.UNITY_LSL_USEFUL_CHANNELS_NUM)]
        [parent.addWidget(upw) for upw in unityLSL_plot_widgets]
        self.unityLSL_plots = [upw.plot([], [], pen=pg.mkPen(color=(255, 0, 0))) for upw in unityLSL_plot_widgets]

    def init_visualize_LSLInlet_data(self, parent, num_chan):
        plot_widgets = [pg.PlotWidget() for i in range(num_chan)]
        [parent.addWidget(upw) for upw in plot_widgets]
        return [pw.plot([], [], pen=pg.mkPen(color=(255, 0, 0))) for pw in plot_widgets]

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

    def process_visualize_LSLInlet_data(self, data_dict):
        if data_dict['frames'].shape[-1] > 0:
            buffered_data = self.LSL_data_buffer_dicts[data_dict['lsl_data_type']]
            buffered_data = np.concatenate(
                (buffered_data, data_dict['frames']),
                axis=-1)  # get all data and remove it from internal buffer

            if buffered_data.shape[-1] < self.LSLInlet_num_visualized_sample:
                data_to_plot = np.concatenate((np.zeros(shape=(
                    buffered_data.shape[0],
                    self.LSLInlet_num_visualized_sample -
                    buffered_data.shape[-1])),
                                               buffered_data), axis=-1)
            else:
                data_to_plot = buffered_data[:,
                               - self.LSLInlet_num_visualized_sample:]  # plot the most recent 10 seconds
            time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, self.LSLInlet_num_visualized_sample)
            [up.setData(time_vector, data_to_plot[i, :]) for i, up in enumerate(self.LSL_plots_dict[data_dict['lsl_data_type']])]

            self.LSL_data_buffer_dicts[data_dict['lsl_data_type']] = data_to_plot  # main window only retains the most recent 10 seconds for visualization purposes

            # notify the internal buffer in recordings tab
            self.recordingTab.update_buffers(data_dict)


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
                    self.inference_num_visualized_results - self.inference_buffer.shape[0],
                    config.INFERENCE_CLASS_NUM)),
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
