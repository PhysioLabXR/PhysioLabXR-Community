import os
import sys
import webbrowser
from typing import Dict

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox, QDialogButtonBox

from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.exceptions.exceptions import RenaError, InvalidStreamMetaInfoError
from physiolabxr.configs import config
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.Presets import Presets
from physiolabxr.presets.PresetEnums import PresetType, DataType
from physiolabxr.ui.AddWiget import AddStreamWidget
from physiolabxr.ui.AudioInputDeviceWidget import AudioInputDeviceWidget
from physiolabxr.ui.BaseStreamWidget import BaseStreamWidget
from physiolabxr.ui.CloseDialog import CloseDialog
from physiolabxr.ui.LSLWidget import LSLWidget
from physiolabxr.ui.ScriptingTab import ScriptingTab
from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier
from physiolabxr.ui.VideoWidget import VideoWidget
from physiolabxr.ui.ZMQWidget import ZMQWidget
from physiolabxr.ui.ui_shared import num_active_streams_label_text
from physiolabxr.presets.presets_utils import get_experiment_preset_streams, is_stream_name_in_presets, \
    create_default_lsl_preset, \
    create_default_zmq_preset, verify_stream_meta_info, get_stream_meta_info, is_name_in_preset, \
    change_stream_preset_type, change_stream_preset_data_type, change_stream_preset_port_number

try:
    import physiolabxr.configs.config
except ModuleNotFoundError as e:
    print("Make sure you set the working directory to PhysioLabXR's root, cwd is " + os.getcwd())
    raise e
import physiolabxr.threadings.workers as workers
from physiolabxr.ui.RecordingsTab import RecordingsTab
from physiolabxr.ui.SettingsWidget import SettingsWidget
from physiolabxr.ui.ReplayTab import ReplayTab
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.utils.ui_utils import dialog_popup, \
    another_window

import numpy as np


# Define function to import external files when using PyInstaller.
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath("..")

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
        SplashLoadingTextNotifier().set_loading_text('Creating main window...')
        self.ui = uic.loadUi(AppConfigs()._ui_mainwindow, self)
        self.setWindowTitle('PhysioLabXR')
        self.app = app
        self.ask_to_close = ask_to_close

        ############
        self.stream_widgets: Dict[str, BaseStreamWidget] = {}
        ############

        # create sensor threads, worker threads for different sensors
        self.device_workers = {}
        self.lsl_workers = {}

        ######### init server
        print('Creating Rena Client')
        # self.rena_dsp_client = RenaTCPInterface(stream_name=config.rena_server_name,
        #                                         port_id=config.rena_server_port,
        #                                         identity='client')

        #########
        # meta data update timer
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
        self.current_dialog = None

        # actions for context menu
        self.actionDocumentation.triggered.connect(self.fire_action_documentation)
        self.actionRepo.triggered.connect(self.fire_action_repo)
        self.actionShow_Recordings.triggered.connect(self.fire_action_show_recordings)
        self.actionExit.triggered.connect(self.fire_action_exit)
        self.actionSettings.triggered.connect(self.fire_action_settings)

        # create the settings window
        self.settings_widget = SettingsWidget(self)
        self.settings_window = another_window('Settings')
        self.settings_window.get_layout().addWidget(self.settings_widget)
        self.settings_window.hide()

        # global buffer object for visualization, recording, and scripting
        self.global_stream_buffer = DataBuffer()


        # close dialog
        self.close_dialog = None
        self.close_event = None
        self.is_already_closed = False

        # # fmri widget
        # # TODO: FMRI WIDGET
        # fmri_preset = FMRIPreset(stream_name='Siemens Prisma 3T', preset_type=PresetType.FMRI, data_type=DataType.float64, num_channels=10713600,
        #                          data_shape=(240,240,186),
        #                          normalize=True, alignment=True, threshold=0.5, nominal_sampling_rate=2, mri_file_path='')
        # _presets().stream_presets[fmri_preset.stream_name] = fmri_preset
        # self.fmri_widget = FMRIWidget(parent_widget=self, parent_layout=self.streamsHorizontalLayout,
        #                               stream_name=fmri_preset.stream_name, data_type=fmri_preset.data_type, worker=None,
        #                               insert_position=None)
        #
        # self.fmri_widget.setObjectName("FMRIWidget")
        # self.fmri_widget.show()

    def add_btn_clicked(self):
        """
        This is the only entry point to adding a stream widget
        :return:
        """
        # self.addStreamWidget.add_btn.setEnabled(False)
        # self.loading_dialog = LoadingDialog(self, message=f"Adding stream {self.addStreamWidget.get_selected_stream_name()}")
        # self.loading_dialog.show()
        # task_thread = LongTaskThread(self, "process_add")
        # task_thread.completed.connect(self.add_completed)
        # task_thread.start()
        selected_text, preset_type, data_type, port = self.addStreamWidget.get_selected_stream_name(), \
                                                      self.addStreamWidget.get_selected_preset_type(), \
                                                       self.addStreamWidget.get_data_type(), \
                                                       self.addStreamWidget.get_port_number()
        self.process_add(selected_text, preset_type, data_type, port)

    # def add_completed(self):
    #     self.addStreamWidget.add_btn.setEnabled(True)
    #     self.loading_dialog.close()

    def process_add(self, stream_name, preset_type, data_type, port):
        if self.recording_tab.is_recording:
            dialog_popup(msg='Cannot add while recording.')
            return

        if len(stream_name) == 0:
            return
        try:
            verify_stream_meta_info(preset_type=preset_type, data_type=data_type, port_number=port)
            if stream_name in self.stream_widgets.keys():  # if this inlet hasn't been already added
                dialog_popup('Nothing is done for: {0}. This stream is already added.'.format(stream_name),title='Warning')
                return
            if not is_name_in_preset(stream_name):
                if not PresetType.is_lsl_zmq_custom_preset(preset_type):
                    dialog_popup("New stream must be either LSL or ZMQ", title="Error", buttons=QDialogButtonBox.StandardButton.Ok)
                    return
                self.create_preset(stream_name, preset_type=preset_type, data_type=data_type, port=port)
                GlobalSignals().stream_presets_entry_changed_signal.emit()  # add the new preset to the combo box

            if preset_type == PresetType.WEBCAM:  # add video device
                self.init_video_device(stream_name, video_preset_type=preset_type)
            elif preset_type == PresetType.AUDIO:
                self.init_audio_input_device(stream_name)
            elif preset_type == PresetType.MONITOR:
                self.init_video_device(stream_name, video_preset_type=preset_type)
            elif preset_type == PresetType.CUSTOM:  # if this is a device preset
                raise NotImplementedError
                # self.init_device(selected_text)  # add device stream
            elif preset_type == PresetType.LSL:
                self.init_LSL_streaming(stream_name)  # add lsl stream
            elif preset_type == PresetType.ZMQ:
                self.init_ZMQ_streaming(stream_name, port, data_type)  # add lsl stream
            elif preset_type == PresetType.EXPERIMENT:  # add multiple streams from an experiment preset
                streams_for_experiment = get_experiment_preset_streams(stream_name)
                self.add_streams_from_experiment_preset(streams_for_experiment)
            else:
                raise Exception("Unknow preset type {}".format(preset_type))
            self.update_active_streams()
        except RenaError as error:
            dialog_popup(f'Failed to add: {stream_name}. {error}', title='Error')
        self.addStreamWidget.check_can_add_input()

    def add_streams_from_experiment_preset(self, stream_names):
        for stream_name in stream_names:
            if stream_name not in self.stream_widgets.keys():
                assert is_stream_name_in_presets(stream_name), InvalidStreamMetaInfoError(f"Adding multiple streams must use streams already defined in presets. Undefined stream: {stream_name}")
                self.process_add(stream_name, *get_stream_meta_info(stream_name))

    def add_streams_from_replay(self, stream_infos):
        # switch tab to stream
        is_new_preset_added = False
        self.ui.tabWidget.setCurrentWidget(self.visualization_tab)
        for stream_name, info in stream_infos.items():
            if stream_name not in self.stream_widgets.keys():
                if not is_stream_name_in_presets(stream_name):
                    sampling_rate = max(0, int(info['srate']))
                    self.create_preset(stream_name, preset_type=info['preset_type'], num_channels=info['n_channels'], data_type=info['data_type'], port=info['port_number'], nominal_sample_rate=sampling_rate)
                    is_new_preset_added = True
                else:
                    stream_meta_info = get_stream_meta_info(stream_name)

                    if stream_meta_info[0].value != info['preset_type']:
                        print(f"Warning: stream {stream_name} has different preset type {stream_meta_info[0].value} from the one in the replay file {info['preset_type']}.")
                        change_stream_preset_type(stream_name, info['preset_type'])
                    if stream_meta_info[1].value != info['data_type']:
                        print(f"Warning: stream {stream_name} has different data type {stream_meta_info[1].value} from the one in the replay file {info['data_type']}.")
                        change_stream_preset_data_type(stream_name, info['data_type'])
                    if info['preset_type'] == PresetType.ZMQ:
                        change_stream_preset_port_number(stream_name, info['port_number'])
                    # n channels won't be dealt here, leave that to starting the stream, handled by BaseStreamWidget
                self.process_add(stream_name, *get_stream_meta_info(stream_name))

        if is_new_preset_added:
            GlobalSignals().stream_presets_entry_changed_signal.emit()

    def create_preset(self, stream_name, preset_type, data_type=DataType.float32, num_channels=1, nominal_sample_rate=None, **kwargs):
        if preset_type == PresetType.LSL:
            create_default_lsl_preset(stream_name, num_channels, nominal_sample_rate, data_type=data_type)  # create the preset
        elif preset_type == PresetType.ZMQ:
            try:
                assert 'port' in kwargs.keys()
            except AssertionError:
                raise ValueError("Port number must be specified for ZMQ preset")
            create_default_zmq_preset(stream_name, kwargs['port'], num_channels, nominal_sample_rate, data_type=data_type)  # create the preset
        elif preset_type == PresetType.CUSTOM:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown preset type {preset_type}")

    def remove_stream_widget(self, target):
        self.streamsHorizontalLayout.removeWidget(target)
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
        for x in self.stream_widgets.values():
            if (not x.add_stream_availability or x.is_stream_available) and not x.is_widget_streaming():
                x.start_stop_stream_btn_clicked()

    def on_stop_all_btn_clicked(self):
        [x.start_stop_stream_btn_clicked() for x in self.stream_widgets.values() if x.is_widget_streaming and x.is_widget_streaming()]

    def init_video_device(self, video_device_name, video_preset_type):
        widget_name = video_device_name + '_widget'
        widget = VideoWidget(parent_widget=self,
                           parent_layout=self.camHorizontalLayout,
                             video_preset_type=video_preset_type,
                           video_device_name=video_device_name,
                           insert_position=self.camHorizontalLayout.count() - 1)
        widget.setObjectName(widget_name)
        self.stream_widgets[video_device_name] = widget

    def init_LSL_streaming(self, stream_name):
        widget_name = stream_name + '_widget'
        stream_widget = LSLWidget(parent_widget=self,
                                 parent_layout=self.streamsHorizontalLayout,
                                 stream_name=stream_name,
                                 insert_position=self.streamsHorizontalLayout.count() - 1)
        stream_widget.setObjectName(widget_name)
        self.stream_widgets[stream_name] = stream_widget

    def init_ZMQ_streaming(self, topic_name, port_number, data_type):
        widget_name = topic_name + '_widget'
        stream_widget = ZMQWidget(parent_widget=self,
                                 parent_layout=self.streamsHorizontalLayout,
                                 topic_name=topic_name,
                                  port_number=port_number,
                                 data_type=data_type,
                                 insert_position=self.streamsHorizontalLayout.count() - 1)
        stream_widget.setObjectName(widget_name)
        self.stream_widgets[topic_name] = stream_widget

    def init_audio_input_device(self, stream_name):
        widget_name = stream_name + '_widget'
        stream_widget = AudioInputDeviceWidget(parent_widget=self,
                                 parent_layout=self.streamsHorizontalLayout,
                                 stream_name=stream_name,
                                 insert_position=self.streamsHorizontalLayout.count() - 1)
        stream_widget.setObjectName(widget_name)
        self.stream_widgets[stream_name] = stream_widget

    def update_meta_data(self):
        # get the stream viz fps
        fps_list = np.array([[s.get_fps() for s in self.stream_widgets.values()]])
        pull_data_delay_list = np.array([[s.get_pull_data_delay() for s in self.stream_widgets.values()]])
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
        if self.is_already_closed:
            event.accept()
            return
        if self.ask_to_close:
            reply = QMessageBox.question(self, 'Confirm Exit', 'Are you sure you want to exit?',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        else:
            reply = QMessageBox.StandardButton.Yes
        if reply == QMessageBox.StandardButton.Yes:
            self.meta_data_update_timer.stop()
            self.meta_data_update_timer.timeout.disconnect()
            if self.settings_window is not None:
                self.settings_window.close()

            # close other tabs
            stream_close_calls = [s_widgets.try_close for s_widgets in self.stream_widgets.values()]
            [c() for c in stream_close_calls]

            self.close_event = event
            if self.scripting_tab.need_to_wait_to_close():
                def abort_script_finish_close():
                    self.scripting_tab.kill_all_scripts()
                    self.finish_close()
                self.close_dialog = CloseDialog(abort_script_finish_close)
                self.scripting_tab.try_close(self.close_dialog.close_success_signal)
                event.ignore()
            else:
                self.scripting_tab.try_close()
                self.finish_close()
        else:
            event.ignore()

    def finish_close(self):
        print('MainWindow: closing replay')
        self.replay_tab.try_close()
        print('MainWindow: closing replay')
        self.settings_widget.try_close()

        Presets().__del__()
        AppConfigs().__del__()

        self.is_already_closed = True  # set this to true so we don't through another close event
        self.close()  # fire another close event

    def fire_action_documentation(self):
        webbrowser.open("https://realitynavigationdocs.readthedocs.io/")

    def fire_action_repo(self):
        webbrowser.open("https://github.com/ApocalyVec/RealityNavigation")

    def fire_action_show_recordings(self):
        self.recording_tab.open_recording_directory()

    def fire_action_exit(self):
        self.close()

    def fire_action_settings(self):
        self.open_settings_tab()

    def open_settings_tab(self, tab_name: str='Streams'):
        self.settings_window.show()
        self.settings_window.activateWindow()
        if tab_name is not None:
            self.settings_widget.switch_to_tab(tab_name)

    def get_added_stream_names(self):
        return list(self.stream_widgets.keys())

    def is_any_stream_widget_added(self):
        return len(self.stream_widgets) > 0

    def is_any_streaming(self):
        """
        Check if any stream is streaming. Checks if any stream widget or video device widget is streaming.
        @return: return True if any network streams or video device is streaming, False otherwise
        """
        is_stream_widgets_streaming = np.any([x.is_widget_streaming() for x in self.stream_widgets.values()])
        return np.any(is_stream_widgets_streaming)

    def remove_stream_widget_with_preset_type(self, preset_type, remove_warning=True):
        # pass
        # # if any stream is streaming, warn the user
        target_widget_names = [s_name for s_name, s_widget in self.stream_widgets.items() if s_widget.preset_type == preset_type]
        if remove_warning and len(target_widget_names) > 0:
            reply = dialog_popup(
                msg=f"The following streams are active {target_widget_names}?.\n"
                    f"Do you want to remove these streams from visualization for now (you can always add them back)?",
                # f'There\'s another stream source with the name {target_widget_names} on the network.\n'
                # f'Are you sure you want to proceed with replaying this file? \n'
                # f'Proceeding may result in unpredictable streaming behavior.\n'
                # f'It is recommended to remove the other data stream with the same name.',
                title='Stream Added Warning', mode='modal',
                buttons=QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
            if reply.result():
                for stream_name in target_widget_names:
                    self.stream_widgets[stream_name].try_close()
            else:
                return

        # for stream_name, stream_widget in self.stream_widgets.items():
        #     if stream_widget.preset_type == preset_type:
        #         stream_widget.try_close()

