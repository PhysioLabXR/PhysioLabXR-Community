import time
import time

import cv2
import numpy as np
import psutil as psutil
import pyautogui
import pyqtgraph as pg
from PyQt5.QtCore import QMutex
from PyQt5.QtCore import (QObject, pyqtSignal)
from pylsl import local_clock

from exceptions.exceptions import DataPortNotOpenError
from rena import config_ui, config_signal, shared
from rena.config import STOP_PROCESS_KILL_TIMEOUT, REQUEST_REALTIME_INFO_TIMEOUT
from rena.interfaces import InferenceInterface, LSLInletInterface
from rena.shared import SCRIPT_STDOUT_MSG_PREFIX, SCRIPT_STOP_REQUEST, SCRIPT_STOP_SUCCESS, SCRIPT_INFO_REQUEST
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.utils.networking_utils import recv_string
from rena.utils.sim import sim_imp, sim_heatmap, sim_detected_points
from rena.utils.sim import sim_openBCI_eeg, sim_unityLSL, sim_inference


class RENAWorker(QObject):
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()
    def __init__(self):
        super().__init__()
        # self.dsp_on = True
        # self.dsp_processor = None
        # self.dsp_server_process = None
        # self.dsp_client = None
        # self.init_dsp_client_server('John')

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

class EEGWorker(QObject):
    """

    """
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
    """

    """
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
    """

    """
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


class LSLInletWorker(RENAWorker):

    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    signal_data_tick = pyqtSignal()

    signal_stream_availability = pyqtSignal(bool)
    signal_stream_availability_tick = pyqtSignal()

    def __init__(self, LSLInlet_interface: LSLInletInterface, RenaTCPInterface=None, *args, **kwargs):
        super(LSLInletWorker, self).__init__()
        self.signal_data_tick.connect(self.process_on_tick)
        self.signal_stream_availability_tick.connect(self.process_stream_availability)

        self._lslInlet_interface = LSLInlet_interface
        self._rena_tcp_interface = RenaTCPInterface
        self.is_streaming = False
        # self.dsp_on = True

        self.start_time = time.time()
        self.num_samples = 0

        self.previous_availability = None

        # self.init_dsp_client_server(self._lslInlet_interface.lsl_stream_name)

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            frames, timestamps = self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer
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

            data_dict = {'lsl_data_type': self._lslInlet_interface.lsl_stream_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)

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

    def start_stream(self):
        self._lslInlet_interface.start_sensor()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()
        self.signal_stream_availability.emit(self._lslInlet_interface.is_stream_available())  # extra emit because the signal availability does not change on this call, but stream widget needs update

    def stop_stream(self):
        self._lslInlet_interface.stop_sensor()
        self.is_streaming = False


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


class WebcamWorker(QObject):
    tick_signal = pyqtSignal()
    change_pixmap_signal = pyqtSignal(tuple)

    def __init__(self, cam_id):
        super().__init__()
        self.cap = None
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(int(self.cam_id))
        self.tick_signal.connect(self.process_on_tick)

    def release_webcam(self):
        if self.cap is not None:
            self.cap.release()

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        ret, cv_img = self.cap.read()
        if ret:
            cv_img = cv_img.astype(np.uint8)
            cv_img = cv2.resize(cv_img, (config_ui.cam_display_width, config_ui.cam_display_height), interpolation=cv2.INTER_NEAREST)
            self.change_pixmap_signal.emit((self.cam_id, cv_img, local_clock()))  # uses lsl local clock for syncing

class ScreenCaptureWorker(QObject):
    tick_signal = pyqtSignal()  # note that the screen capture follows visualization refresh rate
    change_pixmap_signal = pyqtSignal(tuple)

    def __init__(self, screen_label):
        super().__init__()
        self.tick_signal.connect(self.process_on_tick)
        self.screen_label = screen_label

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (config_ui.cam_display_width, config_ui.cam_display_height), interpolation=cv2.INTER_NEAREST)
        self.change_pixmap_signal.emit((self.screen_label, frame, local_clock()))  # uses lsl local clock for syncing


class TimeSeriesDeviceWorker(QObject):
    """

    """
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, eeg_interface=None, *args, **kwargs):
        super(TimeSeriesDeviceWorker, self).__init__()
        self.tick_signal.connect(self.eeg_process_on_tick)
        if not eeg_interface:
            print('None type eeg_interface, starting in simulation mode')

        self._eeg_interface = eeg_interface
        self.is_streaming = True

        self.start_time = time.time()
        self.end_time = time.time()

    @pg.QtCore.pyqtSlot()
    def eeg_process_on_tick(self):
        if self.is_streaming:
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
        self.is_streaming = True
        self.start_time = time.time()

    def stop_stream(self):
        if self._eeg_interface:
            self._eeg_interface.stop_sensor()
        else:
            print('EEGWorker: Stop Simulating eeg data')
            print('EEGWorker: frame rate calculation is not enabled in simulation mode')
        self.is_streaming = False
        self.end_time = time.time()

    def is_streaming(self):
        return self.is_streaming



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
    replay_progress_signal = pyqtSignal(float)

    def __init__(self, command_info_interface):
        super(PlaybackWorker, self).__init__()
        self.command_info_interface: RenaTCPInterface = command_info_interface
        self.playback_tick_signal.connect(self.run)

    @pg.QtCore.pyqtSlot()
    def run(self):
        self.command_info_interface.send_string(shared.VIRTUAL_CLOCK_REQUEST)
        virtual_clock = self.command_info_interface.socket.recv()
        virtual_clock = np.frombuffer(virtual_clock)[0]
        self.replay_progress_signal.emit(virtual_clock)


# class LSLReplayWorker(QObject):
#     replay_progress_signal = pyqtSignal(float)
#
#     def __init__(self, parent, playback_position_signal, play_pause_signal):
#         super(LSLReplayWorker, self).__init__()
#         playback_position_signal.connect(self.on_playback_position_changed)
#         # play_pause_signal.connect(self.on_play_pause_toggle)
#         # self.is_playing = False
#         self.start_time, self.end_time = math.inf, -math.inf
#         self.total_time = None
#
#         # rns_stream = RNStream('C:/Recordings/03_22_2021_16_43_45-Exp_realitynavigation-Sbj_0-Ssn_0 CLEANED.dats')
#         self.stream_data = None # rns_stream.stream_in(ignore_stream=['0', 'monitor1'])
#         # self.tick_signal.connect(self.start_stream)
#         self.stop_signal = False
#
#         # stream related initializations
#         self.stream_names = None
#         self.virtual_clock = None
#         self.virtual_clock_offset = None
#         self.outlets = []
#         self.next_sample_of_stream = [] # index of the next sample of each stream that will be send
#         self.chunk_sizes = [] # how many samples should be published at once
#         self.selected_stream_indices = None
#
#         # fps counter
#         self.tick_times = deque(maxlen=50)
#
#     def virtual_time_to_playback_position_value(self):
#         # TODO: do not hardcode playback range (100)
#         return (self.virtual_clock - self.start_time) * 100 / self.total_time
#
#     def playback_position_value_to_virtual_time(self, position_value):
#         # TODO: do not hardcode playback range (100)
#         print("on_playback_position_changed", position_value)
#         return (self.total_time * (position_value / 100)) + self.start_time
#
#     def on_playback_position_changed(self, new_position):
#         # set the virtual clock according to the new playback position
#         self.virtual_clock = self.playback_position_value_to_virtual_time(new_position)
#
#     # def on_play_pause_toggle(self): # is_playing
#     #     print("on play pause toggle from workers.py - next status is ")
#     #     # play and pause accordingly
#     #     self.is_playing = not self.is_playing
#
#     def setup_stream(self):
#         # setup the streams
#         self.stream_names = list(self.stream_data)
#
#         for i in range(0, len(self.stream_names)):
#             self.outlets.append(None)
#             self.next_sample_of_stream.append(0)
#             self.chunk_sizes.append(1)
#
#         print("Creating outlets")
#         print("\t[index]\t[name]")
#
#         def isStreamVideo(stream):
#             if stream.isdigit():
#                 return True
#             if ("monitor" in stream) or ("video" in stream):
#                 return True
#             return False
#
#         self.selected_stream_indices = list(range(0, len(self.stream_names)))
#
#         for streamIndex, stream_name in enumerate(self.stream_names):
#             if not isStreamVideo(stream_name):
#                 stream_channel_count = self.stream_data[stream_name][0].shape[0]
#                 stream_channel_format = 'double64'
#                 stream_source_id = 'Replay Stream - ' + stream_name
#
#                 outletInfo = pylsl.StreamInfo(stream_name, '', stream_channel_count, 0.0, stream_channel_format,
#                                               stream_source_id)
#
#                 self.outlets[streamIndex] = pylsl.StreamOutlet(outletInfo)
#                 print("\t" + str(streamIndex) + "\t" + stream_name)
#
#         self.virtual_clock_offset = 0
#
#         for stream in self.stream_names:
#             # find the start time
#             if self.virtual_clock is None or self.stream_data[stream][1][0] < self.virtual_clock:
#                 # virtual clock will be set to the timestamp of the first received stream data
#                 self.virtual_clock = self.stream_data[stream][1][0]
#                 self.start_time = self.virtual_clock
#
#             # find the end time
#             if self.stream_data[stream][1][-1] > self.end_time:
#                 self.end_time = self.stream_data[stream][1][-1]
#
#             self.total_time = self.end_time - self.start_time
#
#         self.virtual_clock_offset = pylsl.local_clock() - self.virtual_clock
#         print("Offsetting replayed timestamps by " + str(self.virtual_clock_offset))
#
#         print("start time and end time ", self.start_time, self.end_time)
#
#     @pg.QtCore.pyqtSlot()
#     def replay(self):
#         self.tick_times.append(time.time())
#         print("is_playing process: FPS is {0}".format(self.get_fps()))
#         if self.is_playing:
#             print("replay ticking in progress")
#             # run the stream
#             nextStreamIndex = None
#             nextBlockingTimestamp = None
#
#             if self.stop_signal:
#                 return
#
#             # determine which stream to send next
#             for i, stream_name in enumerate(self.stream_names):
#                 stream = self.stream_data[stream_name]
#                 # when a chunk can be send depends on it's last sample's timestamp
#                 blockingElementIdx = self.next_sample_of_stream[i] + self.chunk_sizes[i] - 1
#                 try:
#                     blockingTimestamp = stream[1][blockingElementIdx]
#                 except Exception as e:
#                     print(e)
#                 if nextBlockingTimestamp is None or blockingTimestamp <= nextBlockingTimestamp:
#                     nextStreamIndex = i
#                     nextBlockingTimestamp = blockingTimestamp
#
#             # retrieve the data and timestamps to be send
#             nextStream = self.stream_data[self.stream_names[nextStreamIndex]]
#             print("chunk sizes: ", self.chunk_sizes)
#             chunkSize = self.chunk_sizes[nextStreamIndex]
#
#             nextChunkRangeStart = self.next_sample_of_stream[nextStreamIndex]
#             nextChunkRangeEnd = nextChunkRangeStart + chunkSize
#
#             nextChunkTimestamps = nextStream[1][nextChunkRangeStart: nextChunkRangeEnd]
#             nextChunkValues = (nextStream[0][:, nextChunkRangeStart: nextChunkRangeEnd]).transpose()
#
#             # prepare the data (if necessary)
#             if isinstance(nextChunkValues, np.ndarray):
#                 # load_xdf loads numbers into numpy arrays (strings will be put into lists). however, LSL doesn't seem to
#                 # handle them properly as providing data in numpy arrays leads to falsified data being sent. therefore the data
#                 # are converted to lists
#                 nextChunkValues = nextChunkValues.tolist()
#             self.next_sample_of_stream[nextStreamIndex] += chunkSize
#
#             stream_length = nextStream[0].shape[-1]
#             # calculates a lower chunk_size if there are not enough samples left for a "complete" chunk
#             if stream_length < self.next_sample_of_stream[nextStreamIndex] + chunkSize:
#                 print("CHUNK UPDATE")
#                 self.chunk_sizes[nextStreamIndex] = stream_length - self.next_sample_of_stream[nextStreamIndex]
#
#             self.virtual_clock = pylsl.local_clock() - self.virtual_clock_offset
#             # TODO: fix this
#             sleepDuration = nextBlockingTimestamp - self.virtual_clock
#             if sleepDuration > 0:
#                 time.sleep(sleepDuration)
#
#             outlet = self.outlets[nextStreamIndex]
#             print("outlet for this replay is: ", outlet)
#             nextStreamName = self.stream_names[nextStreamIndex]
#             if chunkSize == 1:
#                 # print(str(nextChunkTimestamps[0] + virtualTimeOffset) + "\t" + nextStreamName + "\t" + str(nextChunkValues[0]))
#                 outlet.push_sample(nextChunkValues[0], nextChunkTimestamps[0] + self.virtual_clock_offset)
#                 print("pushed, chunk size 1")
#                 print(nextChunkValues)
#             else:
#                 # according to the documentation push_chunk can only be invoked with exactly one (the last) time stamp
#                 outlet.push_chunk(nextChunkValues, nextChunkTimestamps[-1] + self.virtual_clock_offset)
#                 print("pushed else")
#                 # chunks are not printed to the terminal because they happen hundreds of times per second and therefore
#                 # would make the terminal output unreadable
#
#             # remove this stream from the list if there are no remaining samples
#             if self.next_sample_of_stream[nextStreamIndex] >= stream_length:
#                 self.selected_stream_indices.remove(self.selected_stream_indices[nextStreamIndex])
#                 self.outlets.remove(self.outlets[nextStreamIndex])
#                 self.next_sample_of_stream.remove(self.next_sample_of_stream[nextStreamIndex])
#                 self.chunk_sizes.remove(self.chunk_sizes[nextStreamIndex])
#                 self.stream_names.remove(self.stream_names[nextStreamIndex])
#
#             playback_position = self.virtual_time_to_playback_position_value()
#             self.replay_progress_signal.emit(playback_position)
#             print("virtual clock time: ", self.virtual_clock)
#
#         # print(datetime.now())
#
#     @pg.QtCore.pyqtSlot()
#     def start_stream(self):
#         # self.stream_data = stream_data
#         stream_names = list(self.stream_data)
#
#         outlets = []
#         nextSampleOfStream = []  # index of the next sample of each stream that will be send
#         chunk_sizes = []  # how many samples should be published at once
#         for i in range(0, len(stream_names)):
#             outlets.append(None)
#             nextSampleOfStream.append(0)
#             chunk_sizes.append(1)
#
#         print("Creating outlets")
#         print("\t[index]\t[name]")
#
#         def isStreamVideo(stream):
#             if stream.isdigit():
#                 return True
#             if ("monitor" in stream) or ("video" in stream):
#                 return True
#             return False
#
#         selectedStreamIndices = list(range(0, len(stream_names)))
#
#         for streamIndex, stream_name in enumerate(stream_names):
#             if not isStreamVideo(stream_name):
#                 stream_channel_count = self.stream_data[stream_name][0].shape[0]
#                 stream_channel_format = 'double64'
#                 stream_source_id = 'Replay Stream - ' + stream_name
#
#                 outletInfo = pylsl.StreamInfo(stream_name, '', stream_channel_count, 0.0, stream_channel_format,
#                                               stream_source_id)
#
#                 outlets[streamIndex] = pylsl.StreamOutlet(outletInfo)
#                 print("\t" + str(streamIndex) + "\t" + stream_name)
#
#         virtualTimeOffset = 0
#         virtualTime = None
#
#         for stream in stream_names:
#             if virtualTime is None or self.stream_data[stream][1][0] < virtualTime:
#                 # determine when the recording started
#                 virtualTime = self.stream_data[stream][1][0]
#
#         # temp
#         # if virtualTime is None:
#         #     virtualTime = 0
#
#         virtualTimeOffset = pylsl.local_clock() - virtualTime
#         print("Offsetting replayed timestamps by " + str(virtualTimeOffset))
#
#         print(datetime.now())
#         # replay the recording
#         while len(selectedStreamIndices) > 0:  # streams get removed from the list if there are no samples left to play
#
#             nextStreamIndex = None
#             nextBlockingTimestamp = None
#
#             if self.stop_signal:
#                 break
#
#             # determine which stream to send next
#             for i, stream_name in enumerate(stream_names):
#                 stream = self.stream_data[stream_name]
#                 # when a chunk can be send depends on it's last sample's timestamp
#                 blockingElementIdx = nextSampleOfStream[i] + chunk_sizes[i] - 1
#                 try:
#                     blockingTimestamp = stream[1][blockingElementIdx]
#                 except Exception as e:
#                     print(e)
#                 if nextBlockingTimestamp is None or blockingTimestamp <= nextBlockingTimestamp:
#                     nextStreamIndex = i
#                     nextBlockingTimestamp = blockingTimestamp
#
#             # retrieve the data and timestamps to be send
#             nextStream = self.stream_data[stream_names[nextStreamIndex]]
#             chunkSize = chunk_sizes[nextStreamIndex]
#
#             nextChunkRangeStart = nextSampleOfStream[nextStreamIndex]
#             nextChunkRangeEnd = nextChunkRangeStart + chunkSize
#
#             nextChunkTimestamps = nextStream[1][nextChunkRangeStart: nextChunkRangeEnd]
#             nextChunkValues = (nextStream[0][:, nextChunkRangeStart: nextChunkRangeEnd]).transpose()
#
#             # prepare the data (if necessary)
#             if isinstance(nextChunkValues, np.ndarray):
#                 # load_xdf loads numbers into numpy arrays (strings will be put into lists). however, LSL doesn't seem to
#                 # handle them properly as providing data in numpy arrays leads to falsified data being sent. therefore the data
#                 # are converted to lists
#                 nextChunkValues = nextChunkValues.tolist()
#             nextSampleOfStream[nextStreamIndex] += chunkSize
#
#             stream_length = nextStream[0].shape[-1]
#             # calculates a lower chunk_size if there are not enough samples left for a "complete" chunk
#             if stream_length < nextSampleOfStream[nextStreamIndex] + chunkSize:
#                 chunk_sizes[nextStreamIndex] = stream_length - nextSampleOfStream[nextStreamIndex]
#
#             virtualTime = pylsl.local_clock() - virtualTimeOffset
#             # TODO: fix this
#             sleepDuration = nextBlockingTimestamp - virtualTime
#             if sleepDuration > 0:
#                 time.sleep(sleepDuration)
#
#             outlet = outlets[nextStreamIndex]
#             nextStreamName = stream_names[nextStreamIndex]
#             if chunkSize == 1:
#                 # print(str(nextChunkTimestamps[0] + virtualTimeOffset) + "\t" + nextStreamName + "\t" + str(nextChunkValues[0]))
#                 outlet.push_sample(nextChunkValues[0], nextChunkTimestamps[0] + virtualTimeOffset)
#             else:
#                 # according to the documentation push_chunk can only be invoked with exactly one (the last) time stamp
#                 outlet.push_chunk(nextChunkValues, nextChunkTimestamps[-1] + virtualTimeOffset)
#                 # chunks are not printed to the terminal because they happen hundreds of times per second and therefore
#                 # would make the terminal output unreadable
#
#             # remove this stream from the list if there are no remaining samples
#             if nextSampleOfStream[nextStreamIndex] >= stream_length:
#                 selectedStreamIndices.remove(selectedStreamIndices[nextStreamIndex])
#                 outlets.remove(outlets[nextStreamIndex])
#                 nextSampleOfStream.remove(nextSampleOfStream[nextStreamIndex])
#                 chunk_sizes.remove(chunk_sizes[nextStreamIndex])
#                 stream_names.remove(stream_names[nextStreamIndex])
#
#         # self.stop_replay_btn_pressed()
#
#     def get_fps(self):
#         try:
#             return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
#         except ZeroDivisionError:
#             return 0


class ScriptingStdoutWorker(QObject):
    stdout_signal = pyqtSignal(str)
    tick_signal = pyqtSignal()

    def __init__(self, stdout_socket_interface):
        super().__init__()
        self.tick_signal.connect(self.process_stdout)
        self.stdout_socket_interface = stdout_socket_interface

    @pg.QtCore.pyqtSlot()
    def process_stdout(self):
        msg: str = recv_string(self.stdout_socket_interface, is_block=True)  # this must not block otherwise check_pid won't get to run because they are on the same thread
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


class ScriptCommandWorker(QObject):
    command_signal = pyqtSignal(str)
    command_return_signal = pyqtSignal(tuple)

    def __init__(self, command_socket_interface):
        super().__init__()
        self.command_signal.connect(self.process_command)

    @pg.QtCore.pyqtSlot()
    def process_command(self, command):
        self.command_info_mutex.lock()
        if command == SCRIPT_STOP_REQUEST:
            is_success = self.notify_script_to_stop()
        else:
            raise NotImplementedError
        self.command_return_signal.emit((command, is_success))
        self.command_info_mutex.unlock()

    def notify_script_to_stop(self):
        self.info_socket_interface.send_string(SCRIPT_STOP_REQUEST)
        events = self.info_socket_interface.poller.poll(STOP_PROCESS_KILL_TIMEOUT)
        if len(events) > 0:
            msg = self.info_socket_interface.socket.recv().decode('utf-8')
        else:
            msg = None
        if msg == SCRIPT_STOP_SUCCESS:
            return True
        else:
            return False
