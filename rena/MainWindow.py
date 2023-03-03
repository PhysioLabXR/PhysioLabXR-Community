import os
import random
import sys
import time
import webbrowser

import pyqtgraph as pg
from PyQt5 import QtWidgets, sip, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget

from exceptions.exceptions import RenaError
from rena import config
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.ui.AddWiget import AddStreamWidget
from rena.ui.ScriptingTab import ScriptingTab
from rena.ui.VideoDeviceWidget import VideoDeviceWidget
from rena.ui_shared import num_active_streams_label_text
from rena.utils.settings_utils import get_presets_by_category, get_childKeys_for_group, create_default_preset, \
    check_preset_exists, get_experiment_preset_streams

try:
    import rena.config
except ModuleNotFoundError as e:
    print('Make sure you set the working directory to ../RealityNavigation/rena, cwd is ' + os.getcwd())
    raise e
import rena.threadings.workers as workers
from rena.ui.StreamWidget import StreamWidget
from rena.ui.RecordingsTab import RecordingsTab
from rena.ui.SettingsTab import SettingsTab
from rena.ui.ReplayTab import ReplayTab
from rena.utils.data_utils import window_slice
from rena.utils.general import process_preset_create_openBCI_interface_startsensor, \
    process_preset_create_TImmWave_interface_startsensor
from rena.utils.ui_utils import dialog_popup, \
    init_camera_widget, convert_rgb_to_qt_image, another_window

import numpy as np
import collections


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

    def __init__(self, app, ask_to_close=True, *args, **kwargs):
        """
        This is the main entry point to RenaLabApp
        :param app: the main QT app
        :param ask_to_close: whether to show a 'confirm exit' dialog and ask for
         user's confirmation in a close event
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.ui = uic.loadUi("ui/mainwindow.ui", self)
        self.setWindowTitle('Reality Navigation')
        self.app = app
        self.ask_to_close = ask_to_close

        ############
        self.stream_widgets = {}  # key: stream -> value: stream_widget
        self.video_device_widgets = {}  # key: stream -> value: stream_widget
        ############

        # create sensor threads, worker threads for different sensors
        self.device_workers = {}
        self.lsl_workers = {}

        ######### init server
        print('Creating Rena Client')
        self.rena_dsp_client = RenaTCPInterface(stream_name=config.rena_server_name,
                                                port_id=config.rena_server_port,
                                                identity='client')

        #########
        # meta data udpate timer
        self.meta_data_update_timer = QTimer()
        self.meta_data_update_timer.setInterval(config.MAIN_WINDOW_META_DATA_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        self.meta_data_update_timer.timeout.connect(self.update_meta_data)
        self.meta_data_update_timer.start()

        self.addStreamWidget = AddStreamWidget(self)
        self.MainTabVerticalLayout.insertWidget(0, self.addStreamWidget)  # add the add widget to visualization tab's
        self.addStreamWidget.add_btn.clicked.connect(self.add_btn_clicked)

        self.start_all_btn.setEnabled(False)
        self.stop_all_btn.setEnabled(False)

        self.start_all_btn.clicked.connect(self.on_start_all_btn_clicked)
        self.stop_all_btn.clicked.connect(self.on_stop_all_btn_clicked)
        # scripting buffer
        self.inference_buffer = np.empty(shape=(0, config.INFERENCE_CLASS_NUM))  # time axis is the first

        # add other tabs
        self.recording_tab = RecordingsTab(self)
        self.recordings_tab_vertical_layout.addWidget(self.recording_tab)

        self.replay_tab = ReplayTab(self)
        self.replay_tab_vertical_layout.addWidget(self.replay_tab)

        self.scripting_tab = ScriptingTab(self)
        self.scripting_tab_vertical_layout.addWidget(self.scripting_tab)

        # windows
        self.pop_windows = {}

        # actions for context menu
        self.actionDocumentation.triggered.connect(self.fire_action_documentation)
        self.actionRepo.triggered.connect(self.fire_action_repo)
        self.actionShow_Recordings.triggered.connect(self.fire_action_show_recordings)
        self.actionExit.triggered.connect(self.fire_action_exit)
        self.actionSettings.triggered.connect(self.fire_action_settings)

        # create the settings window
        self.settings_tab = SettingsTab(self)
        self.settings_window = another_window('Settings')
        self.settings_window.get_layout().addWidget(self.settings_tab)
        self.settings_window.hide()


    def add_btn_clicked(self):
        """
        This should be the only entry point to adding a stream widget
        :return:
        """
        if self.recording_tab.is_recording:
            dialog_popup(msg='Cannot add while recording.')
            return
        selected_text, data_type, port, networking_interface = self.addStreamWidget.get_selected_stream_name(), \
                                                               self.addStreamWidget.get_data_type(), \
                                                               self.addStreamWidget.get_port_number(), \
                                                               self.addStreamWidget.get_networking_interface()
        if len(selected_text) == 0:
            return
        try:
            if selected_text in self.stream_widgets.keys():  # if this inlet hasn't been already added
                dialog_popup('Nothing is done for: {0}. This stream is already added.'.format(selected_text),title='Warning')
                return
            selected_type = self.addStreamWidget.get_current_selected_type()
            if selected_type == 'video':  # add video device
                self.init_video_device(selected_text)
            elif selected_type == 'Device':  # if this is a device preset
                self.init_device(selected_text)  # add device stream
            elif selected_type == 'LSL' or selected_type == 'ZMQ':
                self.init_network_streaming(selected_text, networking_interface, data_type, port)  # add lsl stream
            elif selected_type == 'exp':  # add multiple streams from an experiment preset
                streams_for_experiment = get_experiment_preset_streams(selected_text)
                self.add_streams_to_visualize(streams_for_experiment)
            elif selected_type == 'other':  # add a previous unknown lsl stream
                self.create_preset(selected_text, data_type, port, networking_interface)
                self.scripting_tab.update_script_widget_input_combobox()  # add thew new preset to the combo box
                self.init_network_streaming(selected_text, data_type=data_type, port_number=port)  # TODO this can also be a device or experiment preset
            else:
                raise Exception("Unknow preset type {}".format(selected_type))
            self.update_active_streams()
        except RenaError as error:
            dialog_popup('Failed to add: {0}. {1}'.format(selected_text, str(error)), title='Error')
        self.addStreamWidget.check_can_add_input()

    def create_preset(self, stream_name, data_type, port, networking_interface, num_channels=1):
        create_default_preset(stream_name, data_type, port, networking_interface, num_channels)  # create the preset
        self.addStreamWidget.update_combobox_presets()  # add thew new preset to the combo box

    def remove_stream_widget(self, target):
        self.sensorTabSensorsHorizontalLayout.removeWidget(target)
        self.update_active_streams()
        self.addStreamWidget.check_can_add_input()  # check if the current selected preset has already been added

    def update_active_streams(self):
        available_widget_count = len([x for x in self.stream_widgets.values() if x.is_stream_available])
        streaming_widget_count = len([x for x in self.stream_widgets.values() if x.is_widget_streaming()])
        self.numActiveStreamsLabel.setText(
            num_active_streams_label_text.format(len(self.stream_widgets), available_widget_count,
                                                 streaming_widget_count, self.replay_tab.get_num_replay_channels()))
        # enable/disable the start/stop all buttons
        self.start_all_btn.setEnabled(available_widget_count > streaming_widget_count)
        self.stop_all_btn.setEnabled(streaming_widget_count > 0)

    def on_start_all_btn_clicked(self):
        [x.start_stop_stream_btn_clicked() for x in self.stream_widgets.values() if x.is_stream_available and not x.is_widget_streaming()]

    def on_stop_all_btn_clicked(self):
        [x.start_stop_stream_btn_clicked() for x in self.stream_widgets.values() if x.is_widget_streaming and x.is_widget_streaming()]

    def init_video_device(self, video_device_name):
        widget_name = video_device_name + '_widget'
        widget = VideoDeviceWidget(main_parent=self,
                                   parent_layout=self.sensorTabSensorsHorizontalLayout,
                                   video_device_name=video_device_name,
                                   insert_position=self.sensorTabSensorsHorizontalLayout.count() - 1)
        widget.setObjectName(widget_name)
        self.video_device_widgets[video_device_name] = widget

    def add_streams_to_visualize(self, stream_names):

        for stream_name in stream_names:
            # check if the stream in setting's preset
            if check_preset_exists(stream_name):
                self.addStreamWidget.select_by_stream_name(stream_name)
                self.addStreamWidget.add_btn.click()
            else:  # add a new preset if the stream name is not defined
                self.addStreamWidget.set_selection_text(stream_name)
                self.addStreamWidget.add_btn.click()

        # loading_dlg.close()

    def add_streams_from_replay(self, stream_names):
        # switch tab to visulalization
        self.ui.tabWidget.setCurrentWidget(self.ui.tabWidget.findChild(QWidget, 'visualization_tab'))
        self.add_streams_to_visualize(stream_names)
        for stream_name in stream_names:
            if self.stream_widgets[stream_name].is_streaming():  # if not running click start stream
                self.stream_widgets[stream_name].StartStopStreamBtn.click()

    def init_network_streaming(self, networking_stream_name, networking_interface='LSL', data_type=None, port_number=None, worker=None):
        error_initialization = False

        # set up UI elements
        widget_name = networking_stream_name + '_widget'
        stream_widget = StreamWidget(main_parent=self,
                                     parent=self.sensorTabSensorsHorizontalLayout,
                                     stream_name=networking_stream_name,
                                     data_type=data_type,
                                     worker = worker,
                                     networking_interface=networking_interface,
                                     port_number=port_number,
                                     insert_position=self.sensorTabSensorsHorizontalLayout.count() - 1)
        start_stop_stream_btn, remove_stream_btn, pop_window_btn = stream_widget.StartStopStreamBtn, stream_widget.RemoveStreamBtn, stream_widget.PopWindowBtn
        stream_widget.setObjectName(widget_name)

        self.stream_widgets[networking_stream_name] = stream_widget

        if error_initialization:
            remove_stream_btn.click()
        config.settings.endGroup()

    def update_meta_data(self):
        # get the stream viz fps
        fps_list = np.array([[s.get_fps() for s in self.stream_widgets.values()] + [v.get_fps() for v in self.video_device_widgets.values()]])
        pull_data_delay_list = np.array([[s.get_pull_data_delay() for s in self.stream_widgets.values()] + [v.get_pull_data_delay() for v in self.video_device_widgets.values()]])
        if len(fps_list) == 0:
            return
        if np.all(fps_list == 0):
            self.visualizationFPSLabel.setText("0")
        else:
            self.visualizationFPSLabel.setText("%.2f" % np.mean(fps_list))

        if len(pull_data_delay_list) == 0:
            return
        if np.all(pull_data_delay_list == 0):
            self.pull_data_delay_label.setText("0")
        else:
            self.pull_data_delay_label.setText("%.5f ms" % (1e3 * np.mean(pull_data_delay_list)))

    def init_device(self, device_name):
        config.settings.beginGroup('presets/streampresets/{0}'.format(device_name))
        device_type = config.settings.value('DeviceType')

        if device_name not in self.device_workers.keys() and device_type == 'OpenBCI':
            serial_port = config.settings.value('_SerialPort')
            board_id = config.settings.value('_Board_id')
            # create and start this device's worker thread
            worker = workers.OpenBCIDeviceWorker(device_name, serial_port, board_id)
            config.settings.endGroup()
            self.init_network_streaming(device_name, networking_interface='Device', worker=worker)
        # TI mmWave connection

        # elif device_name not in self.device_workers.keys() and device_type == 'TImmWave_6843AOP':
        #     print('mmWave test')
        #     try:
        #         # mmWave connect, send config, start sensor
        #         num_range_bin = config.settings.value('NumRangeBin')
        #         Dport = config.settings.value['Dport(Standard)']
        #         Uport = config.settings.value['Uport(Enhanced)']
        #         config_path = config.settings.value['ConfigPath']
        #
        #         MmWaveSensorLSLInterface = process_preset_create_TImmWave_interface_startsensor(
        #             num_range_bin, Dport, Uport, config_path)
        #     except AssertionError as e:
        #         dialog_popup(str(e))
        #         config.settings.endGroup()
        #         return None
        #     self.device_workers[device_name] = workers.MmwWorker(mmw_interface=MmWaveSensorLSLInterface)
        #     worker_thread = pg.QtCore.QThread(self)
        #     self.worker_threads[device_name] = worker_thread
        #     self.device_workers[device_name].moveToThread(self.worker_threads[device_name])
        #     worker_thread.start()
        #     self.init_network_streaming(device_name)  # TODO test needed
        else:
            dialog_popup('We are not supporting this Device or the Device has been added')
        config.settings.endGroup()

    def reload_all_presets_btn_clicked(self):
        if self.reload_all_presets():
            self.update_presets_combo_box()
            dialog_popup('Reloaded all presets', title='Info')

    def update_presets_combo_box(self):
        self.preset_LSLStream_combo_box.clear()
        self.preset_LSLStream_combo_box.addItems(self.lslStream_presets_dict.keys())
        self.device_combo_box.clear()
        self.device_combo_box.addItems(self.device_presets_dict.keys())
        self.experiment_combo_box.clear()
        self.experiment_combo_box.addItems(self.experiment_presets_dict.keys())

    def closeEvent(self, event):
        if self.ask_to_close:
            reply = QMessageBox.question(self, 'Window Close', 'Exit Application?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            reply = QMessageBox.Yes
        if reply == QMessageBox.Yes:
            if self.settings_window is not None:
                self.settings_window.close()
            remove_btns = [x.RemoveStreamBtn for x in self.stream_widgets.values()]
            [x.click() for x in remove_btns]

            # close other tabs
            self.scripting_tab.try_close()
            self.replay_tab.try_close()

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
        self.settings_window.show()
        self.settings_window.activateWindow()

    def get_added_stream_names(self):
        return list(self.stream_widgets.keys()) + list(self.video_device_widgets.keys())
