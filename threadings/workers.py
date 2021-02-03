import base64
import time
from collections import deque
from io import BytesIO

import PIL
import numpy as np
from PIL import Image

from PyQt5.QtCore import pyqtSignal, QObject
import pyqtgraph as pg

import matplotlib.pyplot as plt

import numpy as np

from utils.sim import sim_openBCI_eeg, sim_unityLSL


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
        self._is_connected = False

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

    def connect(self, params):
        """
        check if _eeg_interface exists before connecting.
        """
        if self._eeg_interface:
            self._eeg_interface.connect_sensor()
        else:
            print('EEGWorker: No EEG Interface defined, ignored.')
        self._is_connected = True

    def disconnect(self, params):
        """
        check if _eeg_interface exists before connecting.
        """
        if self._eeg_interface:
            self._eeg_interface.disconnect_sensor()
        else:
            print('EEGWorker: No EEG Interface defined, ignored.')
        self._is_connected = False

    def is_streaming(self):
        return self._is_streaming

    def is_connected(self):
        if self._eeg_interface:
            return self._is_connected
        else:
            print('EEGWorker: No Radar Interface Connected, ignored.')


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
        self._is_streaming = False
        self._is_connected = False

        self.start_time = time.time()
        self.end_time = time.time()

    @pg.QtCore.pyqtSlot()
    def unityLSL_process_on_tick(self):
        if self._is_streaming:
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
        self._is_streaming = True
        self.start_time = time.time()

    def stop_stream(self):
        if self._unityLSL_interface:
            self._unityLSL_interface.stop_sensor()
        else:
            print('UnityLSLWorker: Stop Simulating Unity LSL data')
            print('UnityLSLWorker: frame rate calculation is not enabled in simulation mode')
        self._is_streaming = False
        self.end_time = time.time()

    def connect(self, params):
        """
        check if _unityLSL_interface exists before connecting.
        """
        if self._unityLSL_interface:
            self._unityLSL_interface.connect_sensor()
        else:
            print('UnityLSLWorker: No Unity LSL Interface defined, ignored.')
        self._is_connected = True

    def disconnect(self, params):
        """
        check if _unityLSL_interface exists before connecting.
        """
        if self._unityLSL_interface:
            self._unityLSL_interface.disconnect_sensor()
        else:
            print('UnityLSLWorker: No Unity LSL Interface defined, ignored.')
        self._is_connected = False

    def is_streaming(self):
        return self._is_streaming

    def is_connected(self):
        if self._unityLSL_interface:
            return self._is_connected
        else:
            print('UnityLSLWorker: No Radar Interface Connected, ignored.')