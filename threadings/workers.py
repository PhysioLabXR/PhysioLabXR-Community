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

from utils.sim import sim_openBCI_eeg


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

        self._start_time = time.time()
        self._end_time = time.time()

    @pg.QtCore.pyqtSlot()
    def eeg_process_on_tick(self):
        if self._is_streaming:
            if self._mmw_interface:
                data = self._eeg_interface.process_frames()  # get all data and remove it from internal buffer
            else:  # this is in simulation mode
                # assume we only working with OpenBCI eeg
                data = sim_openBCI_eeg()

            # notify the mmw data for the radar tab
            data_dict = {'data': data}
            self.signal_data.emit(data_dict)

    def start_stream(self):
        if self._eeg_interface:  # if the sensor interfaces is established
            self._eeg_interface.start_sensor()
        else:
            print('EEGWorker: Start Simulating EEG data')
        self._is_streaming = True
        self._start_time = time.time()

    def stop_stream(self):
        self._is_streaming = False
        if self._eeg_interface:
            self._end_time = time.time()
            self._eeg_interface.stop_sensor()
        else:
            print('EEGWorker: Stop Simulating mmW data')
            print('EEGWorker: frame rate calculation is not enabled in simulation mode')

    def connect(self, params):
        """
        check if _mmw_interface exists before connecting.
        """
        if self._eeg_interface:
            self._eeg_interface.connect_sensor()
        else:
            print('EEGWorker: No EEG Interface defined, ignored.')
        self._is_connected = True

    def disconnect(self, params):
        """
        check if _mmw_interface exists before connecting.
        """
        if self._eeg_interface:
            self._eeg_interface.disconnect_sensor()
        else:
            print('EEGWorker: No EEG Interface defined, ignored.')
        self._is_connected = False

    def is_streaming(self):
        return self._is_streaming

    def is_connected(self):
        if self._mmw_interface:
            return self._is_connected
        else:
            print('EEGWorker: No Radar Interface Connected, ignored.')

