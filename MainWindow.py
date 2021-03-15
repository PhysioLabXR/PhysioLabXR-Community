import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, sip
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel
from scipy.signal import decimate

import config
import config_ui
import threadings.workers as workers
from interfaces.SimulationInterface import SimulationInterface
from interfaces.LSLInletInterface import LSLInletInterface
from interfaces.OpenBCIInterface import OpenBCIInterface
from interfaces.UnityLSLInterface import UnityLSLInterface
from ui.RecordingsTab import RecordingsTab
from utils.data_utils import window_slice
from utils.general import load_all_LSL_presets
from utils.ui_utils import init_sensor_or_lsl_widget, init_add_widget, CustomDialog, init_button, dialog_popup, \
    get_distinct_colors, init_camera_widget, convert_cv_qt
import numpy as np

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, inference_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)
        self.setWindowTitle('Reality Navigation')
        # create sensor threads, worker threads for different sensors
        self.worker_threads = {}
        self.sensor_workers = {}
        self.lsl_workers = {}
        self.inference_worker = None
        self.cam_workers = {}
        self.cam_displays = {}

        # create workers for different sensors
        self.init_inference(inference_interface)

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.REFRESH_INTERVAL)  # for 1000 Hz refresh rate
        self.timer.timeout.connect(self.ticks)
        self.timer.start()

        # visualization timer
        self.v_timer = QTimer()
        self.v_timer.setInterval(config.VISUALIZATION_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        self.v_timer.timeout.connect(self.visualize_LSLStream_data)
        self.v_timer.start()

        # camera/screen capture timer
        self.c_timer = QTimer()
        self.c_timer.setInterval(config.CAMERA_SCREENCAPTURE_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        self.c_timer.timeout.connect(self.camera_screen_capture_tick)
        self.c_timer.start()

        # inference timer
        self.inference_timer = QTimer()
        self.inference_timer.setInterval(config.INFERENCE_REFRESH_INTERVAL)  # for 5 KHz refresh rate
        self.inference_timer.timeout.connect(self.inference_ticks)
        self.inference_timer.start()

        # bind visualization
        self.eeg_num_visualized_sample = int(config.OPENBCI_EEG_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)
        self.unityLSL_num_visualized_sample = int(config.UNITY_LSL_SAMPLING_RATE * config.PLOT_RETAIN_HISTORY)

        self.inference_num_visualized_results = int(
            config.PLOT_RETAIN_HISTORY * 1 / (1e-3 * config.INFERENCE_REFRESH_INTERVAL))

        self.lsl_presets = load_all_LSL_presets()
        # add camera and add sensor widget initialization
        self.add_sensor_layout, self.camera_combo_box, self.add_camera_btn, self.sensor_combo_box, self.add_sensor_btn, \
        self.lsl_stream_name_input, self.lsl_num_chan_input, self.add_lsl_btn = init_add_widget(
            parent=self.sensorTabSensorsHorizontalLayout, lsl_presets=self.lsl_presets)
        self.add_camera_btn.clicked.connect(self.add_camera_clicked)
        self.add_sensor_btn.clicked.connect(self.add_sensor_clicked)
        self.add_lsl_btn.clicked.connect(self.add_lsl_clicked)

        # data buffers
        self.LSL_plots_fs_label_dict = {}
        self.LSL_data_buffer_dicts = {}

        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))
        self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))

        # inference buffer
        self.inference_buffer = np.empty(shape=(0, config.INFERENCE_CLASS_NUM))  # time axis is the first

        # add other tabs
        self.recordingTab = RecordingsTab(self, self.LSL_data_buffer_dicts)
        self.recordings_tab_vertical_layout.addWidget(self.recordingTab)

    def add_camera_clicked(self):
        selected_camera_id = self.camera_combo_box.currentText()
        self.init_camera(selected_camera_id)

    def init_camera(self, cam_id):
        if cam_id not in self.cam_workers.keys():
            camera_widget_name = ('Webcam ' if cam_id.isnumeric() else 'Screen Capture ') + str(cam_id)
            camera_widget, camera_layout, remove_cam_btn, camera_img_label = init_camera_widget(
                parent=self.camWidgetVerticalLayout, label_string=camera_widget_name,
                insert_position=self.camWidgetVerticalLayout.count() - 1)
            camera_widget.setObjectName(camera_widget_name)

            # create camera worker thread
            worker_thread = pg.QtCore.QThread(self)
            self.worker_threads[cam_id] = worker_thread

            wkr = workers.WebcamWorker(cam_id=cam_id) if cam_id.isnumeric() else workers.ScreenCaptureWorker(cam_id)

            self.cam_workers[cam_id] = wkr
            self.cam_displays[cam_id] = camera_img_label

            wkr.change_pixmap_signal.connect(self.visualize_cam)

            def remove_cam():
                worker_thread.exit()
                self.cam_workers.pop(cam_id)
                self.cam_displays.pop(cam_id)
                self.sensorTabSensorsHorizontalLayout.removeWidget(camera_widget)
                sip.delete(camera_widget)

            remove_cam_btn.clicked.connect(remove_cam)
            self.cam_workers[cam_id].moveToThread(self.worker_threads[cam_id])
            worker_thread.start()
        else:
            dialog_popup('Webcam with ID ' + cam_id + ' is already added.')

    def visualize_cam(self, cam_id_cv_img):
        cam_id, cv_img = cam_id_cv_img
        if cam_id in self.cam_displays.keys():
            qt_img = convert_cv_qt(cv_img)
            self.cam_displays[cam_id].setPixmap(qt_img)

    def add_sensor_clicked(self):
        selected_text = str(self.sensor_combo_box.currentText())
        if selected_text in self.lsl_presets.keys():
            self.init_lsl(selected_text, self.lsl_presets[selected_text]['NumChannels'],
                          self.lsl_presets[selected_text]['ChannelNames'],
                          self.lsl_presets[selected_text]['PlotGroupSlices'])
        else:
            sensor_type = config_ui.sensor_ui_name_type_dict[selected_text]
            if sensor_type not in self.sensor_workers.keys():
                self.init_sensor(
                    sensor_type=config_ui.sensor_ui_name_type_dict[str(self.sensor_combo_box.currentText())])
            else:
                msg = 'MainWindow: sensor type ' + sensor_type + ' is already added.'
                dlg = CustomDialog(msg)  # If you pass self, the dialog will be centered over the main window as before.
                if dlg.exec_():
                    print("Success!")
                else:
                    print("Cancel!")

    def add_lsl_clicked(self):
        lsl_stream_name = self.lsl_stream_name_input.text()
        lsl_num_chan = int(self.lsl_num_chan_input.text())

        self.init_lsl(lsl_stream_name, lsl_num_chan)

    def init_lsl(self, lsl_stream_name, lsl_num_chan, lsl_chan_names=None, plot_group_slices=None):
        if lsl_stream_name not in self.lsl_workers.keys():
            try:
                if lsl_stream_name.lower().endswith('simulation'):
                    interface = SimulationInterface(lsl_stream_name, lsl_num_chan,
                                                    self.lsl_presets[lsl_stream_name]["NominalSamplingRate"])
                else:
                    interface = LSLInletInterface(lsl_stream_name, lsl_num_chan)
            except AttributeError:
                dialog_popup('Unable to find LSL Stream with given type {0}.'.format(lsl_stream_name))
                return

            if lsl_stream_name.lower().startswith('deap'):
                self.lsl_workers[lsl_stream_name] = workers.DEAPWorker(interface)
            else:
                self.lsl_workers[lsl_stream_name] = workers.LSLInletWorker(interface)
            lsl_widget_name = lsl_stream_name + '_widget'
            lsl_widget, lsl_layout, start_stream_btn, stop_stream_btn = init_sensor_or_lsl_widget(
                parent=self.sensorTabSensorsHorizontalLayout, label_string=lsl_stream_name,
                insert_position=self.sensorTabSensorsHorizontalLayout.count() - 1)
            lsl_widget.setObjectName(lsl_widget_name)
            worker_thread = pg.QtCore.QThread(self)
            self.worker_threads[lsl_stream_name] = worker_thread

            stop_stream_btn.clicked.connect(self.lsl_workers[lsl_stream_name].stop_stream)
            self.LSL_plots_fs_label_dict[lsl_stream_name] = self.init_visualize_LSLStream_data(parent=lsl_layout,
                                                                                               num_chan=lsl_num_chan,
                                                                                               chan_names=lsl_chan_names,
                                                                                               plot_group_slices=plot_group_slices)
            self.lsl_workers[lsl_stream_name].signal_data.connect(self.process_LSLStream_data)
            self.LSL_data_buffer_dicts[lsl_stream_name] = np.empty(shape=(lsl_num_chan, 0))
            self.lsl_presets[lsl_stream_name]["num_samples_to_plot"] = int(
                self.lsl_presets[lsl_stream_name]["NominalSamplingRate"] * config.PLOT_RETAIN_HISTORY)
            self.lsl_presets[lsl_stream_name]["ActualSamplingRate"] = self.lsl_presets[lsl_stream_name][
                "NominalSamplingRate"]
            self.lsl_presets[lsl_stream_name]["timevector"] = np.linspace(0., config.PLOT_RETAIN_HISTORY,
                                                                          self.lsl_presets[lsl_stream_name][
                                                                              "num_samples_to_plot"])

            def remove_lsl():
                # fire stop streaming first
                stop_stream_btn.click()
                worker_thread.exit()
                self.lsl_workers.pop(lsl_stream_name)
                self.worker_threads.pop(lsl_stream_name)
                self.sensorTabSensorsHorizontalLayout.removeWidget(lsl_widget)
                sip.delete(lsl_widget)
                self.LSL_data_buffer_dicts.pop(lsl_stream_name)

            #     worker_thread
            init_button(parent=lsl_layout, label='Remove Stream',
                        function=remove_lsl)  # add delete sensor button after adding visualization
            self.lsl_workers[lsl_stream_name].moveToThread(self.worker_threads[lsl_stream_name])
            start_stream_btn.clicked.connect(self.lsl_workers[lsl_stream_name].start_stream)
            worker_thread.start()
        else:
            dialog_popup('LSL Stream with data type ' + lsl_stream_name + ' is already added.')

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
        if 'Unity.ViveSREyeTracking' in self.lsl_workers.keys() and self.inference_worker:
            if self.lsl_workers['Unity.ViveSREyeTracking'].is_streaming:
                buffered_data = self.LSL_data_buffer_dicts['Unity.ViveSREyeTracking']
                if buffered_data.shape[-1] < config.EYE_INFERENCE_TOTAL_TIMESTEPS:
                    eye_frames = np.concatenate((np.zeros(shape=(
                        2,  # 2 for two eyes' pupil sizes
                        config.EYE_INFERENCE_TOTAL_TIMESTEPS - buffered_data.shape[-1])),
                                                 buffered_data[2:4, :]), axis=-1)
                else:
                    eye_frames = buffered_data[1:3,
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

    # def init_visualize_LSLInlet_data(self, parent, num_chan, chan_names):
    #     plot_widgets = [pg.PlotWidget() for i in range(num_chan)]
    #     plot_containing_layouts = [init_container(parent, vertical=False)[1] for i in range(num_chan)]
    #
    #     chan_names = ['Unknown'] * num_chan if chan_names is None else chan_names
    #     [layout.addWidget(QtWidgets.QLineEdit(text=chan_name)) for layout, chan_name in zip(plot_containing_layouts, chan_names)]
    #     [layout.addWidget(upw) for layout, upw in zip(plot_containing_layouts, plot_widgets)]
    #
    #     [pw.addLegend() for pw in plot_widgets]
    #     return [pw.plot([], [], pen=pg.mkPen(color=(255, 255, 255))) for pw in plot_widgets]

    def init_visualize_LSLStream_data(self, parent, num_chan, chan_names, plot_group_slices):
        fs_label = QLabel(text='Sampling rate = ')
        parent.addWidget(fs_label)
        plot_widget = None
        if plot_group_slices:
            plots = []
            for pg_slice in plot_group_slices:  # one plot widget for each group, no need to check chan_names because plot_group_slices only comes with preset
                plot_widget = pg.PlotWidget()
                parent.addWidget(plot_widget)

                distinct_colors = get_distinct_colors(pg_slice[1] - pg_slice[0])
                plot_widget.addLegend()
                plots += [plot_widget.plot([], [], pen=pg.mkPen(color=color), name=c_name) for color, c_name in
                          zip(distinct_colors, chan_names[pg_slice[0]:pg_slice[1]])]
        else:
            plot_widget = pg.PlotWidget()
            parent.addWidget(plot_widget)

            distinct_colors = get_distinct_colors(num_chan)
            plot_widget.addLegend()
            plots = [plot_widget.plot([], [], pen=pg.mkPen(color=color), name=c_name) for color, c_name in
                     zip(distinct_colors, chan_names)]

        [p.setDownsampling(auto=True, method='mean') for p in plots]
        [p.setClipToView(clip=True) for p in plots]

        return plots, plot_widget, fs_label

    def init_visualize_inference_results(self):
        inference_results_plot_widgets = [pg.PlotWidget() for i in range(config.INFERENCE_CLASS_NUM)]
        [self.inference_widget.layout().addWidget(pw) for pw in inference_results_plot_widgets]
        self.inference_results_plots = [pw.plot([], [], pen=pg.mkPen(color=(0, 255, 255))) for pw in
                                        inference_results_plot_widgets]

    def visualize_eeg_data(self, data_dict):
        if config.sensors[0] in self.sensor_workers.keys():
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

    def process_LSLStream_data(self, data_dict):
        samples_to_plot = self.lsl_presets[data_dict['lsl_data_type']]["num_samples_to_plot"]
        if data_dict['frames'].shape[-1] > 0 and data_dict['lsl_data_type'] in self.LSL_data_buffer_dicts.keys():
            buffered_data = self.LSL_data_buffer_dicts[data_dict['lsl_data_type']]
            try:
                buffered_data = np.concatenate(
                    (buffered_data, data_dict['frames']),
                    axis=-1)  # get all data and remove it from internal buffer
            except ValueError:
                raise Exception('The number of channels for stream {0} mismatch from its preset json.'.format(
                    data_dict['lsl_data_type']))
            if buffered_data.shape[-1] < samples_to_plot:
                data_to_plot = np.concatenate((np.zeros(shape=(
                    buffered_data.shape[0],
                    samples_to_plot -
                    buffered_data.shape[-1])),
                                               buffered_data), axis=-1)
            else:
                data_to_plot = buffered_data[:,
                               - samples_to_plot:]  # plot the most recent 10 seconds

            # main window only retains the most recent 10 seconds for visualization purposes
            self.LSL_data_buffer_dicts[data_dict['lsl_data_type']] = data_to_plot
            self.lsl_presets[data_dict['lsl_data_type']]["ActualSamplingRate"] = data_dict['sampling_rate']
            # notify the internal buffer in recordings tab
            self.recordingTab.update_buffers(data_dict)

    def camera_screen_capture_tick(self):
        [w.tick_signal.emit() for w in self.cam_workers.values()]

    def visualize_LSLStream_data(self):

        for lsl_stream_name, data_to_plot in self.LSL_data_buffer_dicts.items():
            time_vector = self.lsl_presets[lsl_stream_name]["timevector"]
            if lsl_stream_name == 'AIYVoice' and len(data_to_plot[0])>0:
                # TODO: move this into a usable structure for the experiment, for now log the relevant speech as detected
                data_to_plot[-1][-1] = 1
                data_to_plot = data_to_plot.astype(np.float)
            elif lsl_stream_name == 'AIYVoice':
                continue
            if data_to_plot.shape[-1] == len(time_vector):
                actual_sampling_rate = self.lsl_presets[lsl_stream_name]["ActualSamplingRate"]
                max_display_datapoint_num = self.LSL_plots_fs_label_dict[lsl_stream_name][1].size().width()

                # reduce the number of points to plot to the number of pixels in the corresponding plot widget
                if data_to_plot.shape[-1] > config.DOWNSAMPLE_MULTIPLY_THRESHOLD * max_display_datapoint_num:
                    data_to_plot = decimate(data_to_plot, q=int(data_to_plot.shape[-1] / max_display_datapoint_num),
                                            axis=1)  # resample to 100 hz with retain history of 10 sec
                    time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, num=data_to_plot.shape[-1])

                [plot.setData(time_vector, data_to_plot[i, :]) for i, plot in
                 enumerate(self.LSL_plots_fs_label_dict[lsl_stream_name][0])]
                self.LSL_plots_fs_label_dict[lsl_stream_name][2].setText(
                    'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))

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
