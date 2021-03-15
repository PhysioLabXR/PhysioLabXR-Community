import time

#import cv2
import pyqtgraph as pg
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal
from interfaces.SimulationInterface import SimulationInterface
from interfaces.AIYVoiceInterface import AIYVoiceInterface
from interfaces.InferenceInterface import InferenceInterface
from interfaces.LSLInletInterface import LSLInletInterface
from utils.sim import sim_openBCI_eeg, sim_unityLSL, sim_inference
from pylsl import local_clock
import pyautogui

import numpy as np

class EEGWorker(QObject):
    """

    """
    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, eeg_interface=None, *args, **kwargs):
        super(EEGWorker, self).__init__()
        self.tick_signal.connect(self.eeg_process_on_tick)
        self.tick_signal.connect(self.test_google)

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

    @pg.QtCore.pyqtSlot()
    def test_google(self):
        if self._is_streaming:
            print('is streaming')

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

    def __init__(self, LSLInlet_interface: LSLInletInterface, *args, **kwargs):
        super(LSLInletWorker, self).__init__()
        self.tick_signal.connect(self.process_on_tick)

        self._lslInlet_interface = LSLInlet_interface
        self.is_streaming = False
        self.start_time = time.time()
        self.num_samples = 0

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            frames, timestamps= self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer

            self.num_samples += len(timestamps)
            sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0

            data_dict = {'lsl_data_type': self._lslInlet_interface.lsl_data_type, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        self._lslInlet_interface.start_sensor()
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
            self.change_pixmap_signal.emit((self.cam_id, cv_img))


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

        self.change_pixmap_signal.emit((self.screen_label, frame))


class AIYWorker(QObject):

    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, LSLInlet_interface: AIYVoiceInterface, *args, **kwargs):
        super(AIYWorker, self).__init__()
        self.tick_signal.connect(self.process_on_tick)
        self._lslInlet_interface = LSLInlet_interface
        self.is_streaming = False
        self.start_time = time.time()
        self.num_samples = 0

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            frames, timestamps= self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer
            if len(frames) == 0: # if no signal is received from AIY
                frames, timestamps = np.array([[0]]), np.array([local_clock()])
            else:
                print('received confidence rating', frames)
                frames = np.array([[1]]) # just visualize as trigger for now
            self.num_samples += len(timestamps)
            sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0

            data_dict = {'lsl_data_type': self._lslInlet_interface.lsl_data_type, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        self._lslInlet_interface.start_sensor()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()

    def stop_stream(self):
        self._lslInlet_interface.stop_sensor()
        self.is_streaming = False


class DEAPWorker(QObject):

    # for passing data to the gesture tab
    signal_data = pyqtSignal(dict)
    tick_signal = pyqtSignal()

    def __init__(self, LSLInlet_interface: SimulationInterface, *args, **kwargs):
        super(DEAPWorker, self).__init__()
        self.tick_signal.connect(self.process_on_tick)
        self._lslInlet_interface = LSLInlet_interface
        self.is_streaming = False
        self.start_time = time.time()
        self.num_samples = 0

    @pg.QtCore.pyqtSlot()
    def process_on_tick(self):
        if self.is_streaming:
            frames, timestamps= self._lslInlet_interface.process_frames()  # get all data and remove it from internal buffer

            self.num_samples += len(timestamps)
            sampling_rate = self.num_samples / (time.time() - self.start_time) if self.num_samples > 0 else 0

            data_dict = {'lsl_data_type': self._lslInlet_interface.lsl_data_type, 'frames': frames, 'timestamps': timestamps, 'sampling_rate': sampling_rate}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        self._lslInlet_interface.start_sensor()
        self.is_streaming = True

        self.num_samples = 0
        self.start_time = time.time()

    def stop_stream(self):
        self._lslInlet_interface.stop_sensor()
        self.is_streaming = False