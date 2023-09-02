import abc
import time
from collections import deque

import numpy as np
import psutil as psutil
import pyqtgraph as pg
import zmq
from PyQt6 import QtCore
from PyQt6.QtCore import QMutex, QThread
from PyQt6.QtCore import (QObject, pyqtSignal)
from pylsl import local_clock

from physiolabxr.exceptions.exceptions import DataPortNotOpenError
from physiolabxr.configs import config_signal, shared
from physiolabxr.configs.config import REQUEST_REALTIME_INFO_TIMEOUT
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.interfaces.AudioInputInterface import AudioInputInterface
from physiolabxr.configs.shared import SCRIPT_STDOUT_MSG_PREFIX, SCRIPT_INFO_REQUEST, \
    STOP_COMMAND, STOP_SUCCESS_INFO, TERMINATE_COMMAND, TERMINATE_SUCCESS_COMMAND, PLAY_PAUSE_SUCCESS_INFO, \
    PLAY_PAUSE_COMMAND, SLIDER_MOVED_COMMAND, SLIDER_MOVED_SUCCESS_INFO, SCRIPT_STDERR_MSG_PREFIX
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.utils.buffers import process_preset_create_openBCI_interface_startsensor, create_lsl_interface, \
    create_audio_input_interface
from physiolabxr.utils.networking_utils import recv_string
from physiolabxr.utils.sim import sim_imp, sim_heatmap, sim_detected_points


class RenaWorkerMeta(type(QtCore.QObject), abc.ABCMeta):
    pass

class RenaWorker(metaclass=RenaWorkerMeta):
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()
    pull_data_times = deque(maxlen=100 * AppConfigs().pull_data_interval)

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        pass

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    def get_pull_data_delay(self):
        if len(self.pull_data_times) == 0:
            return 0
        return np.mean(self.pull_data_times)


class LSLInletWorker(QObject, RenaWorker):
    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, stream_name, num_channels, RenaTCPInterface=None, *args, **kwargs):
        super(LSLInletWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self._lslInlet_interface = create_lsl_interface(stream_name, num_channels)
        self._rena_tcp_interface = RenaTCPInterface
        self.is_streaming = False
        self.timestamp_queue = deque(maxlen=1024)

        self.start_time = time.time()
        self.num_samples = 0

        self.previous_availability = None

        # self.init_dsp_client_server(self._lslInlet_interface.lsl_stream_name)
        self.interface_mutex = QMutex()

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if QThread.currentThread().isInterruptionRequested():
            return
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            self.interface_mutex.lock()
            frames, timestamps = self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer
            self.timestamp_queue.extend(timestamps)
            if len(self.timestamp_queue) > 1:
                sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
            else:
                sampling_rate = np.nan

            self.interface_mutex.unlock()

            if frames.shape[-1] == 0:
                return

            self.num_samples += len(timestamps)

            data_dict = {'stream_name': self._lslInlet_interface.lsl_stream_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

    @QtCore.pyqtSlot()
    def process_stream_availability(self):
        """
        only emit when the stream is not available
        """
        if QThread.currentThread().isInterruptionRequested():
            return
        is_stream_availability = self._lslInlet_interface.is_stream_available()
        if self.previous_availability is None:  # first time running
            self.previous_availability = is_stream_availability
            self.signal_stream_availability.emit(self._lslInlet_interface.is_stream_available())
        else:
            if is_stream_availability != self.previous_availability:
                self.previous_availability = is_stream_availability
                self.signal_stream_availability.emit(is_stream_availability)

    def reset_interface(self, stream_name, num_channels):
        self.interface_mutex.lock()
        self._lslInlet_interface = create_lsl_interface(stream_name, num_channels)
        self.interface_mutex.unlock()

    def start_stream(self):
        self._lslInlet_interface.start_stream()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()
        self.signal_stream_availability.emit(self._lslInlet_interface.is_stream_available())  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self._lslInlet_interface.stop_stream()
        self.is_streaming = False

    def is_stream_available(self):
        return self._lslInlet_interface.is_stream_available()

class AudioInputDeviceWorker(QObject, RenaWorker):
    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, stream_name, *args, **kwargs):
        super(AudioInputDeviceWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self._audio_device_interface: AudioInputInterface = create_audio_input_interface(stream_name)
        # self._lslInlet_interface = create_lsl_interface(stream_name, num_channels)
        self.is_streaming = False
        self.timestamp_queue = deque(maxlen=1024)

        self.start_time = time.time()
        self.num_samples = 0

        self.previous_availability = None

        # self.init_dsp_client_server(self._lslInlet_interface.lsl_stream_name)
        self.interface_mutex = QMutex()

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if QThread.currentThread().isInterruptionRequested():
            return
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            self.interface_mutex.lock()
            frames, timestamps = self._audio_device_interface.process_frames()  # get all data and remove it from internal buffer
            self.timestamp_queue.extend(timestamps)
            if len(self.timestamp_queue) > 1:
                sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
            else:
                sampling_rate = np.nan

            self.interface_mutex.unlock()

            if frames.shape[-1] == 0:
                return

            self.num_samples += len(timestamps)

            data_dict = {'stream_name': self._audio_device_interface._device_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
            self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

    @QtCore.pyqtSlot()
    def process_stream_availability(self):
        """
        only emit when the stream is not available
        """
        if QThread.currentThread().isInterruptionRequested():
            return
        is_stream_availability = self._audio_device_interface.is_stream_available()
        if self.previous_availability is None:  # first time running
            self.previous_availability = is_stream_availability
            self.signal_stream_availability.emit(self._audio_device_interface.is_stream_available())
        else:
            if is_stream_availability != self.previous_availability:
                self.previous_availability = is_stream_availability
                self.signal_stream_availability.emit(is_stream_availability)

    def reset_interface(self, stream_name, num_channels):
        self.interface_mutex.lock()
        self._audio_device_interface = create_audio_input_interface(stream_name)
        self.interface_mutex.unlock()

    def start_stream(self):
        self._audio_device_interface.start_stream()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()
        self.signal_stream_availability.emit(self._audio_device_interface.is_stream_available())  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self._audio_device_interface.stop_stream()
        self.is_streaming = False

    def is_stream_available(self):
        return self._audio_device_interface.is_stream_available()



class OpenBCIDeviceWorker(QObject):
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

    @QtCore.pyqtSlot()
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
        self.interface.start_stream()
        self.is_streaming = True
        self.start_time = time.time()

    def stop_stream(self):
        self.interface.stop_stream()
        self.is_streaming = False
        self.end_time = time.time()

    def is_streaming(self):
        return self.is_streaming

    @QtCore.pyqtSlot()
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

    @QtCore.pyqtSlot()
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

    @QtCore.pyqtSlot()
    def run(self):
        if self.is_running:
            self.send_command_mutex.lock()
            if len(self.command_queue) > 0:
                to_send = self.command_queue.pop()
                if type(to_send) is str:
                    self.command_info_interface.send_string(to_send)
                elif type(to_send) is np.ndarray:
                    self.command_info_interface.send(to_send)
                elif type(to_send) is list:
                    for s in to_send:
                        if type(s) is str:
                            self.command_info_interface.send_string(s)
                        elif type(s) is np.ndarray:
                            self.command_info_interface.send(s)
                else:
                    raise NotImplementedError
                # a command has been sent, wait for reply
                print('PlaybackWorker: waiting for reply')
                reply = self.command_info_interface.socket.recv()
                print('PlaybackWorker: reply received')

                reply = reply.decode('utf-8')

                if reply == STOP_SUCCESS_INFO:
                    self.replay_stopped()
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
                elif reply == SLIDER_MOVED_SUCCESS_INFO:
                    self.send_command_mutex.unlock()
                    return
                elif reply == TERMINATE_SUCCESS_COMMAND:
                    self.is_running = False
                    self.replay_terminated_signal.emit()
                    self.send_command_mutex.unlock()
                    return
                # elif reply == TERMINATE_SUCCESS_COMMAND:
                #     self.is_running = False
                #     self.replay_terminated_signal.emit()
                #     self.send_command_mutex.unlock()
                #     return'
                else:
                    raise NotImplementedError
            self.command_info_interface.send_string(shared.VIRTUAL_CLOCK_REQUEST)
            virtual_clock = self.command_info_interface.socket.recv()  # this is blocking, but replay should respond fast
            virtual_clock = np.frombuffer(virtual_clock)[0]
            if virtual_clock == -1:  #  important, receiving a virtual clock of value -1 means that the replay has finished from the server end
                self.replay_stopped()
                self.send_command_mutex.unlock()
                return
            self.replay_progress_signal.emit(virtual_clock)
            self.send_command_mutex.unlock()

    def replay_stopped(self):
        self.is_running = False
        self.is_paused = False  # reset is_paused in case is_paused had been set to True
        self.replay_stopped_signal.emit()

    def start_run(self):
        self.is_running = True

    def queue_play_pause_command(self):
        self.send_command_mutex.lock()
        self.command_queue.append(PLAY_PAUSE_COMMAND)
        self.send_command_mutex.unlock()

    def queue_slider_moved_command(self, command):
        self.send_command_mutex.lock()
        self.command_queue.append([SLIDER_MOVED_COMMAND, command])
        self.send_command_mutex.unlock()

    def queue_stop_command(self):
        self.send_command_mutex.lock()
        self.command_queue.append(STOP_COMMAND)
        self.send_command_mutex.unlock()

    def queue_terminate_command(self):

        self.send_command_mutex.lock()
        self.command_queue.append(TERMINATE_COMMAND)
        self.send_command_mutex.unlock()
        self.terminated = True

        # self.send_command_mutex.lock()
        # self.command_info_interface.send_string(TERMINATE_COMMAND)
        # print("PlaybackWorker issued terminate command to replay server.")
        # reply = self.command_info_interface.socket.recv()
        # print("PlaybackWorker received reply from replay server.")
        # reply = reply.decode('utf-8')
        #
        # if reply == TERMINATE_SUCCESS_COMMAND:
        #     self.is_running = False
        #     self.replay_terminated_signal.emit()
        # else:
        #     raise NotImplementedError(f'response for reply {reply} is not implemented')
        # self.send_command_mutex.unlock()

class ScriptingStdoutWorker(QObject):
    std_signal = pyqtSignal(tuple)
    tick_signal = pyqtSignal()

    def __init__(self, stdout_socket_interface):
        super().__init__()
        self.tick_signal.connect(self.process_std)
        self.stdout_socket_interface = stdout_socket_interface

    @QtCore.pyqtSlot()
    def process_std(self):
        msg: str = recv_string(self.stdout_socket_interface, is_block=False)  # this must not block otherwise check_pid won't get to run because they are on the same thread, cannot block otherwise the thread cannot exit
        if msg:  # if received is a message
            prefix = msg[:len(SCRIPT_STDOUT_MSG_PREFIX)]
            msg = msg[len(SCRIPT_STDOUT_MSG_PREFIX):]
            if prefix == SCRIPT_STDOUT_MSG_PREFIX:  # if received is a message
                self.std_signal.emit(('out', msg))  # send message if it's not None
            elif prefix == SCRIPT_STDERR_MSG_PREFIX:
                self.std_signal.emit(('error', msg))

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

    @QtCore.pyqtSlot()
    def check_pid(self):
        """
        check if the script process is still running
        """
        if not psutil.pid_exists(self.script_pid) and self.script_process_active:
            self.abnormal_termination_signal.emit()
            self.deactivate()

    @QtCore.pyqtSlot()
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
#     @QtCore.pyqtSlot()
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

    The supported socket patterns is SUB/PUB
    """

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, port_number, subtopic, data_type, poll_stream_availability=False, *args, **kwargs):
        super(ZMQWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        if poll_stream_availability:
            self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self.data_type = data_type if isinstance(data_type, str) else data_type.value
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
        print('In ZMQWorker.__del__(): Socket closed and context terminated')

    @QtCore.pyqtSlot()
    def process_on_tick(self):
        if QThread.currentThread().isInterruptionRequested():
            return
        if self.is_streaming:
            pull_data_start_time = time.perf_counter()
            timestamp_list, data_list = [], []
            while True:
                try:
                    _, timestamp, data = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                    timestamp_list.append(timestamp:=np.frombuffer(timestamp, dtype=np.float64))
                    data_list.append(np.expand_dims(np.frombuffer(data, dtype=self.data_type), axis=-1))
                    self.timestamp_queue.append(timestamp)
                except zmq.error.Again:
                    break
            if len(timestamp_list) > 0:
                if len(self.timestamp_queue) > 1:
                    sampling_rate = len(self.timestamp_queue) / (np.max(self.timestamp_queue) - np.min(self.timestamp_queue))
                else:
                    sampling_rate = np.nan
                data_dict = {'stream_name': self.subtopic, 'frames': np.concatenate(data_list, axis=1), 'timestamps': np.concatenate(timestamp_list), 'sampling_rate': sampling_rate}
                self.signal_data.emit(data_dict)
                self.pull_data_times.append(time.perf_counter() - pull_data_start_time)

    @QtCore.pyqtSlot()
    def process_stream_availability(self):
        if QThread.currentThread().isInterruptionRequested():
            return
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
        self.signal_stream_availability.emit(True)  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self.is_streaming = False

    def is_stream_available(self):
        poll_results = dict(self.poller.poll(timeout=AppConfigs().zmq_lost_connection_timeout))
        # print(f"pulled stream availability: {len(poll_results)}, at {time.time()}" )
        return len(poll_results) > 0

    def reset_interface(self, stream_name, channel_names):
        pass
