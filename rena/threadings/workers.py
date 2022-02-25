import time

import cv2
import pyqtgraph as pg
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
from pylsl import local_clock

import rena.config_signal
import rena.config_ui
from exceptions.exceptions import DataPortNotOpenError
from rena.interfaces.InferenceInterface import InferenceInterface
from rena.interfaces.LSLInletInterface import LSLInletInterface
from rena.utils.sim import sim_openBCI_eeg, sim_unityLSL, sim_inference, sim_imp, sim_heatmap, sim_detected_points
from rena import config_ui, config_signal
from rena.interfaces import InferenceInterface, LSLInletInterface
from rena.utils.sim import sim_openBCI_eeg, sim_unityLSL, sim_inference

import pyautogui

import numpy as np

from rena.utils.ui_utils import dialog_popup


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


class LSLInletWorker(QObject):

    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, LSLInlet_interface: LSLInletInterface,  *args, **kwargs):
        super(LSLInletWorker, self).__init__()
        self.tick_signal.connect(self.process_on_tick)

        self._lslInlet_interface = LSLInlet_interface
        self.is_streaming = False

        self.start_time = time.time()
        self.num_samples = 0

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            t = time.time()
            frames, timestamps= self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer
            print("LSL interface process frame took {0} seconds".format(time.time() - t))

            self.num_samples += len(timestamps)
            try:
                sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0
            except ZeroDivisionError:
                sampling_rate = 0
            data_dict = {'lsl_data_type': self._lslInlet_interface.lsl_stream_name, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)
    def start_stream(self):
        try:
            self._lslInlet_interface.start_sensor()
        except AttributeError as e:
            dialog_popup(e)
            return
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()

    def stop_stream(self):
        self._lslInlet_interface.stop_sensor()
        self.is_streaming = False

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
