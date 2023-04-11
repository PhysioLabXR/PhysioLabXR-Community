import abc
import time
import time
from collections import deque

import cv2
import numpy as np
import psutil as psutil
import pyautogui
import pyqtgraph as pg
import zmq
from PyQt5 import QtCore
from PyQt5.QtCore import QMutex
from PyQt5.QtCore import (QObject, pyqtSignal)
from pylsl import local_clock

from exceptions.exceptions import DataPortNotOpenError
from rena import config_ui, config_signal, shared, config
from rena.config import STOP_PROCESS_KILL_TIMEOUT, REQUEST_REALTIME_INFO_TIMEOUT
from rena.interfaces import InferenceInterface, LSLInletInterface
from rena.shared import SCRIPT_STDOUT_MSG_PREFIX, SCRIPT_STOP_REQUEST, SCRIPT_STOP_SUCCESS, SCRIPT_INFO_REQUEST, \
    STOP_COMMAND, STOP_SUCCESS_INFO, TERMINATE_COMMAND, TERMINATE_SUCCESS_COMMAND, PLAY_PAUSE_SUCCESS_INFO, PLAY_PAUSE_COMMAND
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.utils.general import process_preset_create_openBCI_interface_startsensor, create_lsl_interface
from rena.utils.networking_utils import recv_string
from rena.utils.sim import sim_imp, sim_heatmap, sim_detected_points
from rena.utils.sim import sim_openBCI_eeg, sim_unityLSL, sim_inference

class RenaWorkerMeta(type(QtCore.QObject), abc.ABCMeta):
    pass

class RenaWorker(metaclass=RenaWorkerMeta):
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()
    # def __init__(self):
    #     super().__init__()
        # self.dsp_on = True
        # self.dsp_processor = None
        # self.dsp_server_process = None
        # self.dsp_client = None
        # self.init_dsp_client_server('John')
    pull_data_times = deque(maxlen=100 * config.pull_data_interval)

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        pass

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    # def init_client(self, rena_tcp_request_object:RenaTCPRequestObject):
    #     print('creating client')
    #     self.rena_tcp_client_interface = RenaTCPInterface(stream_name=rena_tcp_request_object.stream_name,
    #                                                  port_id=rena_tcp_request_object.port_id,
    #                                                  identity='client')


        # self.rena_tcp_client_interface = RenaTCPInterface(stream_name=, port_id=, identity=)

    # def init_dsp_client_server(self, stream_name):
    #
    #     self.dsp_server_process = Process(target=dsp_processor,
    #                                       args=(stream_name,))
    #     # mp.set_start_method(method='spawn')
    #     self.dsp_server_process.start()
    #     print('dsp_server_process pid: ', str(self.dsp_server_process.pid))
    #
    #     dsp_client_interface = RenaTCPInterface(stream_name=stream_name,
    #                                             port_id=self.dsp_server_process.pid,
    #                                             identity='client')
    #     self.dsp_client = RenaTCPClient(RENATCPInterface=dsp_client_interface)
        # create a server and get it's pid
        # server_interface = RENATCPInterface()
        # clint_interface = RENATCPInterface()
        # tcp_client = RENATCP
    def get_pull_data_delay(self):
        if len(self.pull_data_times) == 0:
            return 0
        return np.mean(self.pull_data_times)
"""
Deprecated software/device specific workers
class EEGWorker(QObject):
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, eeg_interface=None, *args, **kwargs):
        super(EEGWorker, self).__init__()
        self.tick_signal.connect(self.eeg_process_on_tick)
        if not eeg_interface:
            print('None type eeg_interface, starting in simulation mode')

        self._eeg_interface = eeg_interface
        self._is_streaming = False

        self.start_time = time.time()
        self.end_time = time.time()

    @pg.QtCore.pyqtSlot()
    def eeg_process_on_tick(self):
        if self._is_streaming:
            if self._eeg_interface:
                data = self._eeg_interface.process_frames()  # get all data and remove it from internal buffer
            else:  # this is in simulation mode
                # assume we only working with OpenBCI eeg
                data = sim_openBCI_eeg()

            # notify the eeg data for the radar tab
            data_dict = {'data': data}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        if self._eeg_interface:  # if the sensor interfaces is established
            self._eeg_interface.start_sensor()
        else:
            print('EEGWorker: Start Simulating EEG data')
        self._is_streaming = True
        self.start_time = time.time()

    def stop_stream(self):
        if self._eeg_interface:
            self._eeg_interface.stop_sensor()
        else:
            print('EEGWorker: Stop Simulating eeg data')
            print('EEGWorker: frame rate calculation is not enabled in simulation mode')
        self._is_streaming = False
        self.end_time = time.time()

    def is_streaming(self):
        return self._is_streaming


class UnityLSLWorker(QObject):
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, unityLSL_interface=None, *args, **kwargs):
        super(UnityLSLWorker, self).__init__()
        self.tick_signal.connect(self.unityLSL_process_on_tick)
        if not unityLSL_interface:
            print('None type unityLSL_interface, starting in simulation mode')

        self._unityLSL_interface = unityLSL_interface
        self.is_streaming = False

        self.start_time = time.time()
        self.end_time = time.time()

    @pg.QtCore.pyqtSlot()
    def unityLSL_process_on_tick(self):
        if self.is_streaming:
            if self._unityLSL_interface:
                data, _ = self._unityLSL_interface.process_frames()  # get all data and remove it from internal buffer
            else:  # this is in simulation mode
                data = sim_unityLSL()

            data_dict = {'data': data}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        if self._unityLSL_interface:  # if the sensor interfaces is established
            self._unityLSL_interface.start_sensor()
        else:
            print('UnityLSLWorker: Start Simulating Unity LSL data')
        self.is_streaming = True
        self.start_time = time.time()

    def stop_stream(self):
        if self._unityLSL_interface:
            self._unityLSL_interface.stop_sensor()
        else:
            print('UnityLSLWorker: Stop Simulating Unity LSL data')
            print('UnityLSLWorker: frame rate calculation is not enabled in simulation mode')
        self.is_streaming = False
        self.end_time = time.time()
        
class InferenceWorker(QObject):
    # for passing data to the gesture tab
    # signal_inference_results = pyqtSignal(np.ndarray)
    signal_inference_results = pyqtSignal(list)
    tick_signal = pyqtSignal(dict)

    def __init__(self, inference_interface: InferenceInterface=None, *args, **kwargs):
        super(InferenceWorker, self).__init__()
        self.tick_signal.connect(self.inference_process_on_tick)
        if not inference_interface:
            print('None type unityLSL_interface, starting in simulation mode')

        self.inference_interface = inference_interface
        self._is_streaming = True
        self.is_connected = False

        self.start_time = time.time()
        self.end_time = time.time()

    def connect(self):
        if self.inference_interface:
            self.inference_interface.connect_inference_result_stream()
            self.is_connected = True

    def disconnect(self):
        if self.inference_interface:
            self.inference_interface.disconnect_inference_result_stream()
            self.is_connected = False

    def inference_process_on_tick(self, samples_dict):
        if self._is_streaming:
            if self.inference_interface:
                inference_results = self.inference_interface.send_samples_receive_inference(samples_dict)  # get all data and remove it from internal buffer
            else:  # this is in simulation mode
                inference_results = sim_inference()  # TODO implement simulation mode
            if len(inference_results) > 0:
                self.signal_inference_results.emit(inference_results)
"""



class LSLInletWorker(QObject, RenaWorker):

    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    # signal_stream_num_channels = pyqtSignal(int)

    def __init__(self, stream_name, channel_names, data_type, RenaTCPInterface=None, *args, **kwargs):
        super(LSLInletWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self.data_type = data_type

        self._lslInlet_interface = create_lsl_interface(stream_name, channel_names)
        self._rena_tcp_interface = RenaTCPInterface
        self.is_streaming = False
        # self.dsp_on = True

        self.start_time = time.time()
        self.num_samples = 0

        self.previous_availability = None

        # self.init_dsp_client_server(self._lslInlet_interface.lsl_stream_name)
        self.interface_mutex = QMutex()

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            self.interface_mutex.lock()
            frames, timestamps = self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer
            self.interface_mutex.unlock()

            if frames.shape[-1] == 0:
                return

            self.num_samples += len(timestamps)
            try:
                sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0
            except ZeroDivisionError:
                sampling_rate = 0
            # if self.dsp_on:
            #     current_time = time.time()
            #     self._rena_tcp_interface.send_array(frames)
            #     # self._rena_tcp_interface.send_obj(RenaTCPObject(data=frames))
            #     # send the data
            #     frames = self._rena_tcp_interface.recv_array()
            #     print('time: ', time.time()-current_time)

                # receive the data
                # frames = rena_tcp_object.data
                # print(frames)

            # if self.dsp_on:
            #     receive_obj = self.dsp_client.process_data(data=RenaTCPObject(data=frames))
            #     print(receive_obj.data)
            # insert professor
            # insert dsp processor
            # if self.dsp_on:
            #     self

            data_dict = {'stream_name': self._lslInlet_interface.lsl_stream_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

    @pg.QtCore.pyqtSlot()
    def process_stream_availability(self):
        """
        only emit when the stream is not available
        """
        is_stream_availability = self._lslInlet_interface.is_stream_available()
        if self.previous_availability is None:  # first time running
            self.previous_availability = is_stream_availability
            self.signal_stream_availability.emit(self._lslInlet_interface.is_stream_available())
        else:
            if is_stream_availability != self.previous_availability:
                self.previous_availability = is_stream_availability
                self.signal_stream_availability.emit(is_stream_availability)

    def reset_interface(self, stream_name, channel_names):
        self.interface_mutex.lock()
        self._lslInlet_interface = create_lsl_interface(stream_name, channel_names)
        self.interface_mutex.unlock()

    def start_stream(self):
        self._lslInlet_interface.start_sensor()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()
        self.signal_stream_availability.emit(self._lslInlet_interface.is_stream_available())  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self._lslInlet_interface.stop_sensor()
        self.is_streaming = False

    def is_stream_available(self):
        return self._lslInlet_interface.is_stream_available()


    # def remove_stream(self):
    #     # self.stop_stream()
    #     # kill server
    #     if self.dsp_server_process:
    #         self.dsp_client.tcp_interface.send_obj(RenaTCPObject(data=None, exit_process=True))
    #         self.dsp_server_process.join()
            # self.dsp_server_process.terminate()
            # while self.dsp_server_process.exitcode is None:
            #     self.dsp_server_process.close()
            #     break
            # self.dsp_server_process.close()


class WebcamWorker(QObject, RenaWorker):
    tick_signal = pyqtSignal()
    change_pixmap_signal = pyqtSignal(tuple)

    def __init__(self, cam_id):
        super().__init__()
        self.cap = None
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(int(self.cam_id))
        self.tick_signal.connect(self.process_on_tick)
        self.is_streaming = True

    def stop_stream(self):
        self.is_streaming = False
        if self.cap is not None:
            self.cap.release()

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            ret, cv_img = self.cap.read()
            if ret:
                cv_img = cv_img.astype(np.uint8)
                cv_img = cv2.resize(cv_img, (config_ui.cam_display_width, config_ui.cam_display_height), interpolation=cv2.INTER_NEAREST)
                self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
                self.change_pixmap_signal.emit((self.cam_id, cv_img, local_clock()))  # uses lsl local clock for syncing


class ScreenCaptureWorker(QObject, RenaWorker):
    tick_signal = pyqtSignal()  # note that the screen capture follows visualization refresh rate
    change_pixmap_signal = pyqtSignal(tuple)

    def __init__(self, screen_label):
        super().__init__()
        self.tick_signal.connect(self.process_on_tick)
        self.screen_label = screen_label
        self.is_streaming = True

    def stop_stream(self):
        self.is_streaming = False

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.uint8)
            frame = cv2.resize(frame, (config_ui.cam_display_width, config_ui.cam_display_height), interpolation=cv2.INTER_NEAREST)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)
            self.change_pixmap_signal.emit((self.screen_label, frame, local_clock()))  # uses lsl local clock for syncing


class OpenBCIDeviceWorker(QObject, RenaWorker):
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()
    def __init__(self, stream_name, serial_port, board_id, *args, **kwargs):
        super(OpenBCIDeviceWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)

        self.interface = process_preset_create_openBCI_interface_startsensor(stream_name, serial_port, board_id)
        self.is_streaming = False

        self.start_time = time.time()

        self.timestamps_queue = deque(maxlen=self.interface.get_sampling_rate() * 10)  # TODO move this number to settings

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            data = self.interface.process_frames()  # get all data and remove it from internal buffer
            # notify the eeg data for the radar tab
            # to_local_clock = time.time() - local_clock()
            # timestamps = data[-2,:] - to_local_clock

            timestamps = data[-2, :] - data[-2, :][-1] + local_clock() if len(data[-2, :]) > 0 else []
            #print("samplee num = ", len(data[-2,:]))

            self.timestamps_queue.extend(timestamps)

            # sampling_rate = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
            sampling_rate = len(self.timestamps_queue) / (self.timestamps_queue[-1] - self.timestamps_queue[0]) if len(self.timestamps_queue) > 1 else 0
            # print("openbci sampling rate:", sampling_rate)
            data_dict = {'stream_name': 'openbci', 'timestamps': timestamps, 'frames': data, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        self.interface.start_sensor()
        self.is_streaming = True
        self.start_time = time.time()

    def stop_stream(self):
        self.interface.stop_sensor()
        self.is_streaming = False
        self.end_time = time.time()

    def is_streaming(self):
        return self.is_streaming

    @pg.QtCore.pyqtSlot()
    def process_stream_availability(self):
        return self.is_stream_available()

    def is_stream_available(self):
        return True




class MmwWorker(QObject):
    """
    mmw data package (dict):
        'range_doppler': ndarray
        'range_azi': ndarray
        'pts': ndarray
        'range_amplitude' ndarray
    """
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)

    tick_signal = pyqtSignal()
    timing_list = []  # TODO refactor timing calculation

    def __init__(self, mmw_interface=None, *args, **kwargs):
        super(MmwWorker, self).__init__()
        self.tick_signal.connect(self.mmw_process_on_tick)
        if not mmw_interface:
            print('None type mmw_interface, starting in simulation mode')

        self._mmw_interface = mmw_interface
        self._is_streaming = True

    @pg.QtCore.pyqtSlot()
    def mmw_process_on_tick(self):
        if self._is_streaming:
            if self._mmw_interface:
                try:
                    start = time.time()
                    pts_array, range_amplitude, rd_heatmap, azi_heatmap, rd_heatmap_clutter_removed, azi_heatmap_clutter_removed = self._mmw_interface.process_frame()
                except DataPortNotOpenError:  # happens when the emitted signal accumulates
                    return
                if range_amplitude is None:  # replace with simulated data if not enabled
                    range_amplitude = sim_imp()
                if rd_heatmap is None:
                    rd_heatmap = rd_heatmap_clutter_removed = sim_heatmap(config_signal.rd_shape)
                if azi_heatmap is None:
                    azi_heatmap = azi_heatmap_clutter_removed = sim_heatmap(config_signal.ra_shape)
                self.timing_list.append(time.time() - start)  # TODO refactor timing calculation

            else:  # this is in simulation mode
                pts_array = sim_detected_points()
                range_amplitude = sim_imp()
                rd_heatmap = rd_heatmap_clutter_removed = sim_heatmap(config_signal.rd_shape)
                azi_heatmap = azi_heatmap_clutter_removed = sim_heatmap(config_signal.ra_shape)

            # notify the mmw data for the radar tab
            data_dict = {'range_doppler': rd_heatmap,
                         'range_azi': azi_heatmap,
                         'range_doppler_rc': rd_heatmap_clutter_removed,
                         'range_azi_rc': azi_heatmap_clutter_removed,
                         'pts': pts_array,
                         'range_amplitude': range_amplitude}
            self.signal_data.emit(data_dict)

    def stop_stream(self):
        if self._mmw_interface:
            self._is_streaming = False
            time.sleep(0.1)
            self._mmw_interface.close_connection()
        else:
            print('EEGWorker: Stop Simulating eeg data')
            print('EEGWorker: frame rate calculation is not enabled in simulation mode')
        self.end_time = time.time()
    # def start_mmw(self):
    #     if self._mmw_interface:  # if the sensor interface is established
    #         try:
    #             self._mmw_interface.start_sensor()
    #         except exceptions.PortsNotSetUpError:
    #             print('Radar COM ports are not set up, connect to the sensor prior to start the sensor')
    #     else:
    #         print('Start Simulating mmW data')
    #         # raise exceptions.InterfaceNotExistError
    #     self._is_running = True
    #
    # def stop_mmw(self):
    #     self._is_running = False
    #     time.sleep(0.1)  # wait 100ms for the previous frames to finish process
    #     if self._mmw_interface:
    #         self._mmw_interface.stop_sensor()
    #         print('frame rate is ' + str(1 / np.mean(self.timing_list)))  # TODO refactor timing calculation
    #     else:
    #         print('Stop Simulating mmW data')
    #         print('frame rate calculation is not enabled in simulation mode')
    #
    # def is_mmw_running(self):
    #     return self._is_running
    #
    # def is_connected(self):
    #     if self._mmw_interface:
    #         return self._mmw_interface.is_connected()
    #     else:
    #         print('No Radar Interface Connected, ignored.')
    #         # raise exceptions.InterfaceNotExistError
    #
    # def set_rd_csr(self, value):
    #     if self._mmw_interface:
    #         self._mmw_interface.set_rd_csr(value)
    #
    # def set_ra_csr(self, value):
    #     if self._mmw_interface:
    #         self._mmw_interface.set_ra_csr(value)
    #
    # # def connect_mmw(self, uport_name, dport_name):
    # #     """
    # #     check if _mmw_interface exists before connecting.
    # #     """
    # #     if self._mmw_interface:
    # #         self._mmw_interface.connect(uport_name, dport_name)
    # #     else:
    # #         print('No Radar Interface Connected, ignored.')
    # #         # raise exceptions.InterfaceNotExistError
    #
    # def disconnect_mmw(self):
    #     """
    #     check if _mmw_interface exists before disconnecting.
    #     """
    #     self.stop_mmw()
    #     if self._mmw_interface:
    #         self._mmw_interface.close_connection()
    #     else:
    #         print('No Radar Interface Connected, ignored.')
    #         # raise exceptions.InterfaceNotExistError
    #
    # # def send_config(self, config_path):
    # #     """
    # #     check if _mmw_interface exists before sending the config path.
    # #     """
    # #     if self._mmw_interface:
    # #         self._mmw_interface.send_config(config_path)
    # #         self.start_mmw()
    # #     else:
    # #         print('No Radar Interface Connected, ignored.')
    # #         # raise exceptions.InterfaceNotExistError

class PlaybackWorker(QObject):
    """
    The playback worker listens from the replay process and emit the playback position
    """
    playback_tick_signal = pyqtSignal()
    replay_play_pause_signal = pyqtSignal(str)
    replay_progress_signal = pyqtSignal(float)
    replay_stopped_signal = pyqtSignal()
    replay_terminated_signal = pyqtSignal()

    def __init__(self, command_info_interface):
        super(PlaybackWorker, self).__init__()
        self.command_info_interface: RenaTCPInterface = command_info_interface
        self.playback_tick_signal.connect(self.run)
        self.send_command_mutex = QMutex()
        self.command_queue = deque()
        self.is_running = False
        # initialize pause/resume status
        self.is_paused = False

    @pg.QtCore.pyqtSlot()
    def run(self):
        if self.is_running:
            self.send_command_mutex.lock()
            if len(self.command_queue) > 0:
                self.command_info_interface.send_string(self.command_queue.pop())
                reply = self.command_info_interface.socket.recv()
                reply = reply.decode('utf-8')
                if reply == STOP_SUCCESS_INFO:
                    self.is_running = False
                    self.is_paused = False # reset is_paused in case is_paused had been set to True
                    self.replay_stopped_signal.emit()
                    self.send_command_mutex.unlock()
                    return
                elif reply == PLAY_PAUSE_SUCCESS_INFO:
                    if self.is_paused:
                        self.replay_play_pause_signal.emit('resume')
                    else:
                        self.replay_play_pause_signal.emit('pause')
                    self.is_paused = not self.is_paused
                    self.send_command_mutex.unlock()
                    return
                # elif reply == TERMINATE_SUCCESS_COMMAND:
                #     self.is_running = False
                #     self.replay_terminated_signal.emit()
                #     self.send_command_mutex.unlock()
                #     return
                else:
                    raise NotImplementedError
            self.command_info_interface.send_string(shared.VIRTUAL_CLOCK_REQUEST)
            virtual_clock = self.command_info_interface.socket.recv()  # this is blocking, but replay should respond fast
            virtual_clock = np.frombuffer(virtual_clock)[0]
            if virtual_clock == -1:  # replay has finished
                self.is_running = False
                self.replay_stopped_signal.emit()
                self.send_command_mutex.unlock()
                return
            self.replay_progress_signal.emit(virtual_clock)
            self.send_command_mutex.unlock()

    def start_run(self):
        self.is_running = True

    def queue_play_pause_command(self):
        self.send_command_mutex.lock()
        self.command_queue.append(PLAY_PAUSE_COMMAND)
        self.send_command_mutex.unlock()

    def queue_stop_command(self):
        self.send_command_mutex.lock()
        self.command_queue.append(STOP_COMMAND)
        self.send_command_mutex.unlock()

    def queue_terminate_command(self):
        self.send_command_mutex.lock()
        self.command_info_interface.send_string(TERMINATE_COMMAND)
        reply = self.command_info_interface.socket.recv()
        reply = reply.decode('utf-8')

        if reply == TERMINATE_SUCCESS_COMMAND:
            self.is_running = False
            self.replay_terminated_signal.emit()
        else:
            raise NotImplementedError
        self.send_command_mutex.unlock()

class ScriptingStdoutWorker(QObject):
    stdout_signal = pyqtSignal(str)
    tick_signal = pyqtSignal()

    def __init__(self, stdout_socket_interface):
        super().__init__()
        self.tick_signal.connect(self.process_stdout)
        self.stdout_socket_interface = stdout_socket_interface

    @pg.QtCore.pyqtSlot()
    def process_stdout(self):
        msg: str = recv_string(self.stdout_socket_interface, is_block=False)  # this must not block otherwise check_pid won't get to run because they are on the same thread, cannot block otherwise the thread cannot exit
        if msg:
            if msg.startswith(SCRIPT_STDOUT_MSG_PREFIX):  # if received is a message
                msg = msg[len(SCRIPT_STDOUT_MSG_PREFIX):]
                self.stdout_signal.emit(msg)  # send message if it's not None


class ScriptInfoWorker(QObject):
    abnormal_termination_signal = pyqtSignal()
    tick_signal = pyqtSignal()
    realtime_info_signal = pyqtSignal(list)

    def __init__(self, info_socket_interface, script_pid):
        super().__init__()
        self.tick_signal.connect(self.check_pid)
        self.tick_signal.connect(self.request_get_info)
        self.info_socket_interface = info_socket_interface
        self.script_pid = script_pid
        self.script_process_active = True
        self.send_info_request = False

    @pg.QtCore.pyqtSlot()
    def check_pid(self):
        """
        check if the script process is still running
        """
        if not psutil.pid_exists(self.script_pid) and self.script_process_active:
            self.abnormal_termination_signal.emit()
            self.deactivate()

    @pg.QtCore.pyqtSlot()
    def request_get_info(self):
        if self.script_process_active:
            if not self.send_info_request:  # should not duplicate a request if the last request hasn't been answered yet
                self.info_socket_interface.send_string(SCRIPT_INFO_REQUEST)
                self.send_info_request = True

            events = self.info_socket_interface.poller.poll(REQUEST_REALTIME_INFO_TIMEOUT)
            if len(events):
                self.send_info_request = False
                msg = self.info_socket_interface.socket.recv()
                realtime_info = np.frombuffer(msg)
                self.realtime_info_signal.emit(list(realtime_info))

    def deactivate(self):
        self.script_process_active = False


# class ScriptCommandWorker(QObject):
#     command_signal = pyqtSignal(str)
#     command_return_signal = pyqtSignal(tuple)
#
#     def __init__(self):
#         super().__init__()
#         self.command_signal.connect(self.process_command)
#
#     @pg.QtCore.pyqtSlot()
#     def process_command(self, command):
#         self.command_info_mutex.lock()
#         if command == SCRIPT_STOP_REQUEST:
#             is_success = self.notify_script_to_stop()
#         else:
#             raise NotImplementedError
#         self.command_return_signal.emit((command, is_success))
#         self.command_info_mutex.unlock()
#
#     def notify_script_to_stop(self):
#         self.info_socket_interface.send_string(SCRIPT_STOP_REQUEST)
#         events = self.info_socket_interface.poller.poll(STOP_PROCESS_KILL_TIMEOUT)
#         if len(events) > 0:
#             msg = self.info_socket_interface.socket.recv().decode('utf-8')
#         else:
#             msg = None
#         if msg == SCRIPT_STOP_SUCCESS:
#             return True
#         else:
#             return False

class ZMQWorker(QObject, RenaWorker):
    """
    Rena's implementation of working with ZMQ's tcp interfaces
    """
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, port_number, subtopic, data_type, *args, **kwargs):
        super(ZMQWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self.data_type = data_type
        # networking parameters
        self.sub_address = "tcp://localhost:%s" % port_number
        self.subtopic = subtopic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.sub_address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.subtopic)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        self.ZQMSocket = RenaTCPInterface
        self.is_streaming = False
        self.timestamp_queue = deque(maxlen=1024)

        self.previous_availability = None
        self.last_poll_time = None
        self.is_stream_available()

    def __del__(self):
        self.socket.close()
        self.context.term()
        print('In ZMQWorker.__dell__(): Socket closed and context terminated')

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            try:
                pull_data_start_time = time.perf_counter()
                _, timestamp, data = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                np.frombuffer(timestamp)
            except zmq.error.Again:
                return None
            timestamp = np.frombuffer(timestamp, dtype=np.float64)
            self.timestamp_queue.append(timestamp)
            if len(self.timestamp_queue) > 1:
                sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
            else:
                sampling_rate = np.nan
            data = np.expand_dims(np.frombuffer(data, dtype=self.data_type), axis=-1)
            data_dict = {'stream_name': self.subtopic, 'frames': data, 'timestamps': timestamp, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

    @pg.QtCore.pyqtSlot()
    def process_stream_availability(self):
        is_stream_availability = self.is_stream_available()
        if self.previous_availability is None:  # first time running
            self.previous_availability = is_stream_availability
            self.signal_stream_availability.emit(self.is_stream_available())
        else:
            if is_stream_availability != self.previous_availability:
                self.previous_availability = is_stream_availability
                self.signal_stream_availability.emit(is_stream_availability)

    def start_stream(self):
        self.is_streaming = True

    def stop_stream(self):
        self.is_streaming = False

    def is_stream_available(self):
        poll_results = dict(self.poller.poll(timeout=1000))
        return len(poll_results) > 0

    def reset_interface(self, stream_name, channel_names):
        pass
