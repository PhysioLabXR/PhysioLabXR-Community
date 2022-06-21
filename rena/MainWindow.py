import sys
import time
import webbrowser

import pyqtgraph as pg
from PyQt5 import QtWidgets, sip, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QMessageBox, QWidget
from PyQt5.QtWidgets import QLabel, QMessageBox
from pyqtgraph import PlotDataItem
from scipy.signal import decimate
from PyQt5 import QtCore
import config
import config_ui
import threadings.workers as workers
# from interfaces.UnityLSLInterface import UnityLSLInterface
from rena.ui.InferenceTab import InferenceTab
from rena.ui.StreamWidget import StreamWidget
from ui.RecordingsTab import RecordingsTab
from ui.SettingsTab import SettingsTab
from ui.ReplayTab import ReplayTab
from utils.data_utils import window_slice
from utils.general import load_all_lslStream_presets, create_LSL_preset, process_LSL_plot_group, \
    process_preset_create_lsl_interface, load_all_Device_presets, process_preset_create_openBCI_interface_startsensor, \
    load_all_experiment_presets, process_preset_create_TImmWave_interface_startsensor
from utils.ui_utils import init_sensor_or_lsl_widget, init_add_widget, CustomDialog, init_button, dialog_popup, \
    get_distinct_colors, init_camera_widget, convert_cv_qt, AnotherWindow, convert_heatmap_qt
from utils.general import load_all_lslStream_presets, create_LSL_preset, process_preset_create_lsl_interface, \
    load_all_Device_presets, \
    load_all_experiment_presets
from utils.ui_utils import init_sensor_or_lsl_widget, init_add_widget, init_button, dialog_popup, \
    get_distinct_colors, init_camera_widget, convert_cv_qt, AnotherWindow, another_window, stream_stylesheet
import numpy as np
import collections
from ui.SignalSettingsTab import SignalSettingsTab
import os
from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal, pyqtSlot)
from threadings.workers import LSLReplayWorker


# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, inference_interface, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)
        self.setWindowTitle('Reality Navigation')
        self.app = app

        # create sensor threads, worker threads for different sensors
        self.worker_threads = {}
        self.sensor_workers = {}
        self.device_workers = {}
        self.lsl_workers = {}
        self.inference_worker = None
        self.cam_workers = {}
        self.cam_displays = {}
        self.lsl_replay_worker = None
        self.recent_visualization_refresh_timestamps = collections.deque(
            maxlen=config.VISUALIZATION_REFRESH_FREQUENCY_RETAIN_FRAMES)
        self.recent_tick_refresh_timestamps = collections.deque(maxlen=config.REFRESH_FREQUENCY_RETAIN_FRAMES)
        self.visualization_fps = 0
        self.tick_rate = 0

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

        self.lslStream_presets_dict = None
        self.device_presets_dict = None
        self.experiment_presets_dict = None
        self.reload_all_presets()

        # add camera and add sensor widget initialization
        self.add_layout, self.camera_combo_box, self.add_camera_btn, self.preset_LSLStream_combo_box, self.add_preset_lslStream_btn, \
        self.lslStream_name_input, self.add_lslStream_btn, self.reload_presets_btn, self.device_combo_box, self.add_preset_device_btn, \
        self.experiment_combo_box, self.add_experiment_btn = init_add_widget(
            parent=self.sensorTabSensorsHorizontalLayout, lslStream_presets=self.lslStream_presets_dict,
            device_presets=self.device_presets_dict, experiment_presets=self.experiment_presets_dict)
        # add cam
        self.add_camera_btn.clicked.connect(self.add_camera_clicked)
        # add lsl sensor
        self.add_preset_lslStream_btn.clicked.connect(self.add_preset_lslStream_clicked)

        self.add_preset_device_btn.clicked.connect(self.add_preset_device_clicked)  # add serial connection sensor
        self.add_lslStream_btn.clicked.connect(self.add_lslStream_clicked)
        self.add_experiment_btn.clicked.connect(self.add_preset_experiment_clicked)
        # reload all presets
        self.reload_presets_btn.clicked.connect(self.reload_all_presets_btn_clicked)

        self.stream_ui_elements = {}

        # data buffers
        self.LSL_plots_fs_label_dict = {}
        self.LSL_data_buffer_dicts = {}
        self.LSL_current_ts_dict = {}

        self.eeg_data_buffer = np.empty(shape=(config.OPENBCI_EEG_CHANNEL_SIZE, 0))

        # self.unityLSL_data_buffer = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))

        # inference buffer
        self.inference_buffer = np.empty(shape=(0, config.INFERENCE_CLASS_NUM))  # time axis is the first

        # add other tabs
        self.recording_tab = RecordingsTab(self)
        self.recordings_tab_vertical_layout.addWidget(self.recording_tab)

        # self.settingTab = SettingsTab(self)
        # self.settings_tab_vertical_layout.addWidget(self.settingTab)

        self.replay_tab = ReplayTab(self)
        self.replay_tab_vertical_layout.addWidget(self.replay_tab)
        # self.lsl_replay_worker_thread = QThread(self)
        # self.lsl_replay_worker_thread.start()
        # self.lsl_replay_worker = LSLReplayWorker()
        # self.lsl_replay_worker.moveToThread(self.lsl_replay_worker_thread)
        # self.lsl_replay_worker_thread.started.connect(self.parent.lsl_replay_worker.start_stream())

        self.inference_tab = InferenceTab(self)
        self.inference_tab_vertical_layout.addWidget(self.inference_tab)

        # windows
        self.pop_windows = {}
        self.test_ts_buffer = []

        # actions for context menu
        self.actionDocumentation.triggered.connect(self.fire_action_documentation)
        self.actionRepo.triggered.connect(self.fire_action_repo)
        self.actionShow_Recordings.triggered.connect(self.fire_action_show_recordings)
        self.actionExit.triggered.connect(self.fire_action_exit)
        self.actionSettings.triggered.connect(self.fire_action_settings)

        # reserve windows
        self.settings_window = None
        stream_stylesheet('ui/stylesheet/light.qss')

    def add_camera_clicked(self):
        if self.recording_tab.is_recording:
            dialog_popup(msg='Cannot add capture while recording.')
            return
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
                if self.recording_tab.is_recording:
                    dialog_popup(msg='Cannot remove stream while recording.')
                    return False
                worker_thread.exit()
                self.cam_workers.pop(cam_id)
                self.cam_displays.pop(cam_id)
                self.sensorTabSensorsHorizontalLayout.removeWidget(camera_widget)
                sip.delete(camera_widget)
                return True

            remove_cam_btn.clicked.connect(remove_cam)
            self.cam_workers[cam_id].moveToThread(self.worker_threads[cam_id])
            worker_thread.start()
        else:
            dialog_popup('Webcam with ID ' + cam_id + ' is already added.')

    def visualize_cam(self, cam_id_cv_img_timestamp):
        cam_id, cv_img, timestamp = cam_id_cv_img_timestamp
        if cam_id in self.cam_displays.keys():
            qt_img = convert_cv_qt(cv_img)
            self.cam_displays[cam_id].setPixmap(qt_img)
            self.test_ts_buffer.append(time.time())
            self.recording_tab.update_camera_screen_buffer(cam_id, cv_img, timestamp)

    def add_preset_lslStream_clicked(self):
        if self.recording_tab.is_recording:
            dialog_popup(msg='Cannot add stream while recording.')
            return
        selected_text = str(self.preset_LSLStream_combo_box.currentText())
        if selected_text in self.lslStream_presets_dict.keys():
            self.init_lsl(self.lslStream_presets_dict[selected_text])
        else:
            sensor_type = config_ui.sensor_ui_name_type_dict[selected_text]
            if sensor_type not in self.sensor_workers.keys():
                self.init_sensor(
                    sensor_type=config_ui.sensor_ui_name_type_dict[str(self.preset_LSLStream_combo_box.currentText())])
            else:
                msg = 'Sensor type ' + sensor_type + ' is already added.'
                dialog_popup(msg)

    def add_preset_device_clicked(self):
        if self.recording_tab.is_recording:
            dialog_popup(msg='Cannot add device while recording.')
            return
        selected_text = str(self.device_combo_box.currentText())
        if selected_text in self.device_presets_dict.keys():
            print('device found in device preset')
            device_lsl_preset = self.init_device(self.device_presets_dict[selected_text])

    def add_lslStream_clicked(self):
        lsl_stream_name = self.lslStream_name_input.text()
        preset_dict = create_LSL_preset(lsl_stream_name)
        if self.init_lsl(preset_dict):
            self.lslStream_presets_dict[lsl_stream_name] = preset_dict
            self.update_presets_combo_box()  # add the new user-defined stream to presets dropdown

    def add_preset_experiment_clicked(self):
        selected_text = str(self.experiment_combo_box.currentText())
        if selected_text in self.experiment_presets_dict.keys():
            streams_for_experiment = self.experiment_presets_dict[selected_text]
            self.add_streams_to_visulaize(streams_for_experiment)

    def add_streams_to_visulaize(self, stream_names):
        try:
            assert np.all([x in self.lslStream_presets_dict.keys() or x in self.device_presets_dict.keys() for x in
                           stream_names])
        except AssertionError:
            dialog_popup(
                msg="One or more stream name(s) in the experiment preset is not defined in LSLPreset or DevicePreset",
                title="Error")
            return
        loading_dlg = dialog_popup(
            msg="Please wait while streams are being added...",
            title="Info")
        for stream_name in stream_names:
            isLSL = stream_name in self.lslStream_presets_dict.keys()
            if isLSL:
                index = self.preset_LSLStream_combo_box.findText(stream_name, pg.QtCore.Qt.MatchFixedString)
                self.preset_LSLStream_combo_box.setCurrentIndex(index)
                self.add_preset_lslStream_clicked()
            else:
                index = self.device_combo_box.findText(stream_name, pg.QtCore.Qt.MatchFixedString)
                self.device_combo_box.setCurrentIndex(index)
                self.add_preset_device_clicked()
        loading_dlg.close()

    def add_streams_from_replay(self, stream_names):
        # switch tab to visulalization
        self.ui.tabWidget.setCurrentWidget(self.ui.tabWidget.findChild(QWidget, 'visualization_tab'))
        self.add_streams_to_visulaize(stream_names)
        for stream_name in stream_names:
            if stream_name in self.lsl_workers.keys() and not self.lsl_workers[stream_name].is_streaming:
                self.stream_ui_elements[stream_name]['start_stop_stream_btn'].click()

    def init_lsl(self, preset):
        error_initialization = False
        lsl_stream_name = preset['StreamName']
        if lsl_stream_name not in self.lsl_workers.keys():  # if this inlet hasn't been already added
            try:
                preset, interface = process_preset_create_lsl_interface(preset)
            except AssertionError as e:
                dialog_popup(str(e))
                return None
            lsl_num_chan, lsl_chan_names, plot_group_slices = preset['NumChannels'], \
                                                              preset['ChannelNames'], \
                                                              preset['PlotGroupSlices']

            if 'GroupFormat' in preset.keys():
                plot_group_format = preset['GroupFormat']
            elif plot_group_slices is not None:
                plot_group_format = ['time_series'] * (len(plot_group_slices))
            else:
                plot_group_format = ['time_series']

            # set up UI elements
            lsl_widget_name = lsl_stream_name + '_widget'
            lsl_stream_widget = StreamWidget(main_parent=self, parent=self.sensorTabSensorsHorizontalLayout, stream_name=lsl_stream_name, insert_position=self.sensorTabSensorsHorizontalLayout.count() - 1)
            start_stop_stream_btn, remove_stream_btn, pop_window_btn = lsl_stream_widget.StartStopStreamBtn, lsl_stream_widget.RemoveStreamBtn, lsl_stream_widget.PopWindowBtn
            lsl_layout = lsl_stream_widget.TopLayout
            lsl_stream_widget.setObjectName(lsl_widget_name)

            # set up workers
            self.lsl_workers[lsl_stream_name] = workers.LSLInletWorker(interface)
            worker_thread = lsl_stream_widget.worker_thread
            # worker_thread = pg.QtCore.QThread(self)
            self.worker_threads[lsl_stream_name] = worker_thread

            # stop_stream_btn.clicked.connect(self.lsl_workers[lsl_stream_name].stop_stream)

            try:
                self.LSL_plots_fs_label_dict[lsl_stream_name] = self.init_visualize_LSLStream_data(
                    parent=lsl_layout,
                    num_chan=lsl_num_chan,
                    chan_names=lsl_chan_names,
                    plot_group_slices=plot_group_slices,
                    plot_group_format=plot_group_format
                )

            except AssertionError as e:
                error_initialization = True
                dialog_popup(str(e))

            self.lsl_workers[lsl_stream_name].signal_data.connect(self.process_LSLStream_data)
            self.LSL_data_buffer_dicts[lsl_stream_name] = np.empty(shape=(lsl_num_chan, 0))
            preset["num_samples_to_plot"] = int(
                preset["NominalSamplingRate"] * config.PLOT_RETAIN_HISTORY)
            preset["ActualSamplingRate"] = preset[
                "NominalSamplingRate"]  # actual sampling rate is updated during runtime
            preset["timevector"] = np.linspace(0., config.PLOT_RETAIN_HISTORY,
                                               preset[
                                                   "num_samples_to_plot"])

            def signal_settings_window():
                print("signal settings btn clicked")
                signal_settings_window = SignalSettingsTab()
                if signal_settings_window.exec_():
                    print("signal setting window open")
                else:
                    print("Cancel!")




            self.stream_ui_elements[lsl_stream_name] = {'lsl_widget': lsl_stream_widget,
                                                        'start_stop_stream_btn': start_stop_stream_btn,
                                                        'remove_stream_btn': remove_stream_btn}

            self.lsl_workers[lsl_stream_name].moveToThread(self.worker_threads[lsl_stream_name])
            # start_stream_btn.clicked.connect(self.lsl_workers[lsl_stream_name].start_stream)
            worker_thread.start()
            # if error initialization
            if error_initialization:
                remove_stream_btn.click()
            return preset
        else:
            dialog_popup('LSL Stream with data type ' + lsl_stream_name + ' is already added.')
            return None

    def init_device(self, device_presets):
        device_name = device_presets['StreamName']
        device_type = device_presets['DeviceType']
        if device_name not in self.device_workers.keys() and device_type == 'OpenBCI':
            try:
                openBCI_lsl_presets, OpenBCILSLInterface = process_preset_create_openBCI_interface_startsensor(
                    device_presets)
            except AssertionError as e:
                dialog_popup(str(e))
                return None
            self.lslStream_presets_dict[device_name] = openBCI_lsl_presets
            self.device_workers[device_name] = workers.TimeSeriesDeviceWorker(OpenBCILSLInterface)
            worker_thread = pg.QtCore.QThread(self)
            self.worker_threads[device_name] = worker_thread
            self.device_workers[device_name].moveToThread(self.worker_threads[device_name])
            worker_thread.start()
            self.init_lsl(openBCI_lsl_presets)

        # TI mmWave connection
        if device_name not in self.device_workers.keys() and device_type == 'TImmWave_6843AOP':
            print('mmWave test')
            try:
                # mmWave connect, send config, start sensor
                mmWave_lsl_presets, MmWaveSensorLSLInterface = process_preset_create_TImmWave_interface_startsensor(
                    device_presets)
            except AssertionError as e:
                dialog_popup(str(e))
                return None
            self.lslStream_presets_dict[device_name] = mmWave_lsl_presets
            self.device_workers[device_name] = workers.MmwWorker(mmw_interface=MmWaveSensorLSLInterface)
            worker_thread = pg.QtCore.QThread(self)
            self.worker_threads[device_name] = worker_thread
            self.device_workers[device_name].moveToThread(self.worker_threads[device_name])
            worker_thread.start()
            self.init_lsl(mmWave_lsl_presets)

        else:
            dialog_popup('We are not supporting this Device or the Device has been added')
            return None

    def init_inference(self, inference_interface):
        inference_thread = pg.QtCore.QThread(self)
        self.worker_threads['inference'] = inference_thread
        self.inference_worker = workers.InferenceWorker(inference_interface)
        self.inference_worker.moveToThread(self.worker_threads['inference'])
        self.init_visualize_inference_results()
        self.inference_worker.signal_inference_results.connect(self.visualize_inference_results)

        self.connect_inference_btn.clicked.connect(self.inference_worker.connect)
        self.disconnect_inference_btn.clicked.connect(self.inference_worker.disconnect)

        # self.connect_inference_btn.setStyleSheet(config_ui.inference_button_style)
        inference_thread.start()
        self.inference_widget.hide()

    def ticks(self):
        """
        ticks every 'refresh' milliseconds
        """
        # pass
        [w.tick_signal.emit() for w in self.sensor_workers.values()]
        [w.tick_signal.emit() for w in self.lsl_workers.values()]
        [w.tick_signal.emit() for w in self.device_workers.values()]

        self.recent_tick_refresh_timestamps.append(time.time())
        if len(self.recent_tick_refresh_timestamps) > 2:
            self.tick_rate = 1 / ((self.recent_tick_refresh_timestamps[-1] - self.recent_tick_refresh_timestamps[0]) / (
                        len(self.recent_tick_refresh_timestamps) - 1))

        self.tickFrequencyLabel.setText(
            'Pull Data Frequency: {0}'.format(round(self.tick_rate, config_ui.tick_frequency_decimal_places)))

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

    def init_visualize_eeg_data(self, parent):
        eeg_plot_widgets = [pg.PlotWidget() for i in range(config.OPENBCI_EEG_USEFUL_CHANNELS_NUM)]
        [parent.addWidget(epw) for epw in eeg_plot_widgets]
        self.eeg_plots = [epw.plot([], [], pen=pg.mkPen(color=(255, 255, 255))) for epw in eeg_plot_widgets]

    def init_visualize_LSLStream_data(self, parent, num_chan, chan_names, plot_group_slices, plot_group_format):
        fs_label = QLabel(text='Sampling rate = ')
        ts_label = QLabel(text='Current Time Stamp = ')
        parent.addWidget(fs_label)
        parent.addWidget(ts_label)
        plot_widget = pg.PlotWidget()
        if plot_group_slices:
            plots = []
            plot_formats = []
            for pg_slice, channel_format in zip(plot_group_slices, plot_group_format):
                plot_group_format_info = channel_format.split('_')
                # one plot widget for each group, no need to check chan_names because plot_group_slices only comes with preset
                if plot_group_format_info[0] == 'time' and plot_group_format_info[1] == 'series':  # time_series plot
                    plot_formats.append(plot_group_format_info)
                    plot_widget = pg.PlotWidget()
                    parent.addWidget(plot_widget)

                    distinct_colors = get_distinct_colors(pg_slice[1] - pg_slice[0])
                    plot_widget.addLegend()
                    plots += [plot_widget.plot([], [], pen=pg.mkPen(color=color), name=c_name) for color, c_name in
                              zip(distinct_colors, chan_names[pg_slice[0]:pg_slice[1]])]

                elif plot_group_format_info[0] == 'image':
                    plot_group_format_info[1] = tuple(eval(plot_group_format_info[1]))

                    # check if the channel num matches:
                    if pg_slice[1] - pg_slice[0] != np.prod(np.array(plot_group_format_info[1])):
                        raise AssertionError(
                            'The number of channel in this slice does not match with the number of image pixels.'
                            'The image format is {0} but channel slice format is {1}'.format(plot_group_format_info,
                                                                                             pg_slice))
                    plot_formats.append(plot_group_format_info)
                    image_label = QLabel('Image_Label')
                    image_label.setAlignment(QtCore.Qt.AlignCenter)
                    parent.addWidget(image_label)
                    plots.append(image_label)
                else:
                    raise AssertionError('Unknown plotting group format. We only support: time_series, image_(a,b,c)')

        else:
            plot_widget = pg.PlotWidget()
            parent.addWidget(plot_widget)

            distinct_colors = get_distinct_colors(num_chan)
            plot_widget.addLegend()
            plots = [plot_widget.plot([], [], pen=pg.mkPen(color=color), name=c_name) for color, c_name in
                     zip(distinct_colors, chan_names)]
            plot_formats = [['time_series']]

        [p.setDownsampling(auto=True, method='mean') for p in plots if p == PlotDataItem]
        [p.setClipToView(clip=True) for p in plots for p in plots if p == PlotDataItem]

        return plots, plot_widget, fs_label, plot_group_slices, plot_formats, ts_label

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
        samples_to_plot = self.lslStream_presets_dict[data_dict['lsl_data_type']]["num_samples_to_plot"]
        if data_dict['frames'].shape[-1] > 0 and data_dict['lsl_data_type'] in self.LSL_data_buffer_dicts.keys():
            buffered_data = self.LSL_data_buffer_dicts[data_dict['lsl_data_type']]
            try:
                buffered_data = np.concatenate(
                    (buffered_data, data_dict['frames']),
                    axis=-1)  # get all data and remove it from internal buffer
                self.LSL_current_ts_dict[data_dict['lsl_data_type']] = data_dict['timestamps'][-1]
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
                               - samples_to_plot:]  # plot the most recent few seconds

            # main window only retains the most recent 10 seconds for visualization purposes
            self.LSL_data_buffer_dicts[data_dict['lsl_data_type']] = data_to_plot
            self.lslStream_presets_dict[data_dict['lsl_data_type']]["ActualSamplingRate"] = data_dict['sampling_rate']
            # notify the internal buffer in recordings tab

            # reshape data_dict based on sensor interface
            self.recording_tab.update_buffers(data_dict)

            # inference tab
            self.inference_tab.update_buffers(data_dict)

    def camera_screen_capture_tick(self):
        [w.tick_signal.emit() for w in self.cam_workers.values()]

    def visualize_LSLStream_data(self):

        for lsl_stream_name, data_to_plot in self.LSL_data_buffer_dicts.items():
            time_vector = self.lslStream_presets_dict[lsl_stream_name]["timevector"]

            if data_to_plot.shape[-1] == len(time_vector):
                actual_sampling_rate = self.lslStream_presets_dict[lsl_stream_name]["ActualSamplingRate"]
                max_display_datapoint_num = self.LSL_plots_fs_label_dict[lsl_stream_name][1].size().width()

                # reduce the number of points to plot to the number of pixels in the corresponding plot widget
                if data_to_plot.shape[-1] > config.DOWNSAMPLE_MULTIPLY_THRESHOLD * max_display_datapoint_num:
                    data_to_plot = np.nan_to_num(data_to_plot, nan=0)
                    # start = time.time()
                    # data_to_plot = data_to_plot[:, ::int(data_to_plot.shape[-1] / max_display_datapoint_num)]
                    # data_to_plot = signal.resample(data_to_plot, int(data_to_plot.shape[-1] / max_display_datapoint_num), axis=1)
                    data_to_plot = decimate(data_to_plot, q=int(data_to_plot.shape[-1] / max_display_datapoint_num),
                                            axis=1)  # resample to 100 hz with retain history of 10 sec
                    # print(time.time()-start)
                    time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, num=data_to_plot.shape[-1])

                # self.LSL_plots_fs_label_dict[lsl_stream_name][2].setText(
                #     'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
                #
                # [plot.setData(time_vector, data_to_plot[i, :]) for i, plot in
                #  enumerate(self.LSL_plots_fs_label_dict[lsl_stream_name][0])]
                # change to loop with type condition plot time_series and image
                if self.LSL_plots_fs_label_dict[lsl_stream_name][3]:
                    plot_channel_num_offset = 0
                    for plot_group_index, (plot_group, plot_format) in enumerate(
                            zip(self.LSL_plots_fs_label_dict[lsl_stream_name][3],
                                self.LSL_plots_fs_label_dict[lsl_stream_name][4])):
                        if plot_format[0] == 'time' and plot_format[1] == 'series':
                            # plot corresponding time series data, range (a,b) is time series
                            plot_group_channel_num = plot_group[1] - plot_group[0]
                            for i in range(plot_channel_num_offset, plot_channel_num_offset + plot_group_channel_num):
                                self.LSL_plots_fs_label_dict[lsl_stream_name][0][i].setData(time_vector,
                                                                                            data_to_plot[i, :])
                            plot_channel_num_offset += plot_group_channel_num
                        elif plot_format[0] == 'image':
                            image_shape = plot_format[1]
                            channel_num = image_shape[2]
                            plot_array = data_to_plot[plot_group[0]: plot_group[1], -1]

                            img = plot_array.reshape(image_shape)
                            # display openCV image if channel_num = 3
                            # display heat map if channel_num = 1
                            if channel_num == 3:
                                img = convert_cv_qt(img)
                            if channel_num == 1:
                                img = np.squeeze(img, axis=-1)
                                img = convert_heatmap_qt(img)

                            self.LSL_plots_fs_label_dict[lsl_stream_name][0][plot_channel_num_offset].setPixmap(img)
                            plot_channel_num_offset += 1

                else:
                    [plot.setData(time_vector, data_to_plot[i, :]) for i, plot in
                     enumerate(self.LSL_plots_fs_label_dict[lsl_stream_name][0])]

                self.LSL_plots_fs_label_dict[lsl_stream_name][2].setText(
                    'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
                self.LSL_plots_fs_label_dict[lsl_stream_name][-1].setText(
                    'Current Time Stamp = {0}'.format(self.LSL_current_ts_dict[lsl_stream_name]))

        # calculate and update the frame rate
        self.recent_visualization_refresh_timestamps.append(time.time())
        if len(self.recent_visualization_refresh_timestamps) > 2:
            self.visualization_fps = 1 / ((self.recent_visualization_refresh_timestamps[-1] -
                                           self.recent_visualization_refresh_timestamps[0]) / (
                                                      len(self.recent_visualization_refresh_timestamps) - 1))
        # print("visualization refresh frequency: "+ str(self.visualization_refresh_frequency))
        # print("John")
        self.visualizationFPSLabel.setText(
            'Visualization FPS: {0}'.format(round(self.visualization_fps, config_ui.visualization_fps_decimal_places)))

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

    def reload_all_presets_btn_clicked(self):
        if self.reload_all_presets():
            self.update_presets_combo_box()
            dialog_popup('Reloaded all presets', title='Info')

    def reload_all_presets(self):
        if len(self.lsl_workers) > 0 or len(self.device_workers) > 0:
            dialog_popup('Remove all streams before reloading presets!', title='Warning')
            return False
        else:
            try:
                self.lslStream_presets_dict = load_all_lslStream_presets()
                self.device_presets_dict = load_all_Device_presets()
                self.experiment_presets_dict = load_all_experiment_presets()
            except KeyError as e:
                dialog_popup(
                    msg='Unknown preset specifier, {0}\n Please check the example presets for list of valid specifiers: '.format(
                        e), title='Error')
                return False
        return True

    def update_presets_combo_box(self):
        self.preset_LSLStream_combo_box.clear()
        self.preset_LSLStream_combo_box.addItems(self.lslStream_presets_dict.keys())
        self.device_combo_box.clear()
        self.device_combo_box.addItems(self.device_presets_dict.keys())
        self.experiment_combo_box.clear()
        self.experiment_combo_box.addItems(self.experiment_presets_dict.keys())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Exit Application?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.settings_window is not None:
                self.settings_window.close()
            remove_btns = [x['remove_stream_btn'] for x in self.stream_ui_elements.values()]
            [x.click() for x in remove_btns]
            event.accept()
            self.app.quit()
        else:
            event.ignore()

    def fire_action_documentation(self):
        webbrowser.open("https://realitynavigationdocs.readthedocs.io/")

    def fire_action_repo(self):
        webbrowser.open("https://github.com/ApocalyVec/RealityNavigation")

    def fire_action_show_recordings(self):
        self.recording_tab.open_recording_directory()

    def fire_action_exit(self):
        self.close()

    def fire_action_settings(self):
        if self.settings_window is not None:
            self.settings_window.close()
        settings_tab = SettingsTab(self)
        self.settings_window = another_window('Settings')
        self.settings_window.get_layout().addWidget(settings_tab)
        self.settings_window.show()
