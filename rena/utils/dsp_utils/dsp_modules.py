from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt
from scipy.sparse.linalg import spsolve
from enum import Enum
from PyQt5.QtCore import (QObject, pyqtSignal)


class DataProcessorType(Enum):
    NotchFilter = 'NotchFilter'
    ButterworthBandpassFilter = 'ButterworthBandpassFilter'
    RealtimeVrms = 'RealtimeVrms'


class DataProcessor(QObject):
    data_processor_valid_signal = pyqtSignal()
    data_processor_activated_signal = pyqtSignal()

    def __init__(self, data_processor_type: DataProcessorType = None):
        super().__init__()
        self.data_processor_type = data_processor_type
        self.data_processor_activated = False
        self.data_processor_valid = False
        self.channel_num = 0

    def process_sample(self, data):
        return data

    def process_buffer(self, data):
        if self.data_processor_valid and self.data_processor_activated:
            output_buffer = np.empty(shape=data.shape)
            for index in range(0, data.shape[1]):
                output_buffer[:, index] = self.process_sample(data[:, index])
            return output_buffer
        else:
            return data

    def reset_data_processor(self):
        pass

    def activate_data_processor(self):
        self.data_processor_activated = True

    def deactivate_data_processor(self):
        self.data_processor_activated = False

    def evoke_data_processor(self):
        # set data_processor_valid
        pass

    def set_data_processor_params(self, **params):
        pass

    def set_channel_num(self, channel_num):
        self.channel_num = channel_num
        self.evoke_data_processor()

    def set_data_processor_activated(self, data_processor_activated):
        self.data_processor_activated = data_processor_activated
        self.data_processor_activated_signal.emit()

    def set_data_processor_valid(self, data_processor_valid):
        self.data_processor_valid = data_processor_valid
        self.data_processor_valid_signal.emit()


class IIRFilter(DataProcessor):

    def __init__(self, data_processor_type: DataProcessorType):
        super().__init__(data_processor_type)
        self._a = None
        self._b = None
        self._x_tap = None
        self._y_tap = None

    def process_sample(self, data):
        # perform realtime filter with tap

        # push x
        self._x_tap[:, 1:] = self._x_tap[:, : -1]
        self._x_tap[:, 0] = data
        # push y
        self._y_tap[:, 1:] = self._y_tap[:, : -1]
        # calculate new y
        self._y_tap[:, 0] = np.sum(np.multiply(self._x_tap, self._b), axis=1) - \
                            np.sum(np.multiply(self._y_tap[:, 1:], self._a[1:]), axis=1)

        data = self._y_tap[:, 0]
        return data

    def reset_data_processor(self):
        self._x_tap.fill(0)
        self._y_tap.fill(0)


class NotchFilter(IIRFilter):
    def __init__(self, w0: float = 0, Q: float = 0, fs: float = 0):
        super().__init__(data_processor_type=DataProcessorType.NotchFilter)
        self.w0 = w0
        self.Q = Q
        self.fs = fs

    def evoke_data_processor(self):
        try:
            self._b, self._a = iirnotch(w0=self.w0, Q=self.Q, fs=self.fs)
            self._x_tap = np.zeros((self.channel_num, len(self._b)))
            self._y_tap = np.zeros((self.channel_num, len(self._a)))
            self.set_data_processor_valid(True)
            self.reset_data_processor()
        except (ValueError, ZeroDivisionError, TypeError) as e:
            self.data_processor_valid = False
            print(e)
            print('Data Processor Evoke Failed Error')

    def set_data_processor_params(self, w0, Q, fs):
        self.w0 = w0
        self.Q = Q
        self.fs = fs

        self.evoke_data_processor()


class ButterworthBandpassFilter(IIRFilter):
    def __init__(self, lowcut: float = 0, highcut: float = 0, fs: float = 0, order: int = 0):
        super().__init__(data_processor_type=DataProcessorType.ButterworthBandpassFilter)
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def evoke_data_processor(self):
        try:
            self._b, self._a = self.butter_bandpass(lowcut=self.lowcut,
                                                    highcut=self.highcut,
                                                    fs=self.fs,
                                                    order=self.order)
            self._x_tap = np.zeros((self.channel_num, len(self._b)))
            self._y_tap = np.zeros((self.channel_num, len(self._a)))
            self.set_data_processor_valid(True)
            self.reset_data_processor()
            print("data_processor_valid")
        except (ValueError, ZeroDivisionError, TypeError) as e:
            self.set_data_processor_valid(False)
            print(e)
            print('Data Processor Evoke Failed Error')

    def set_data_processor_params(self, lowcut, highcut, fs, order):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        self.evoke_data_processor()

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # def process_sample(self, data):
    #     # perform realtime filter with tap
    #
    #     # push x
    #     self._x_tap[:, 1:] = self._x_tap[:, : -1]
    #     self._x_tap[:, 0] = data
    #     # push y
    #     self._y_tap[:, 1:] = self._y_tap[:, : -1]
    #     # calculate new y
    #     self._y_tap[:, 0] = np.sum(np.multiply(self._x_tap, self._b), axis=1) - \
    #                         np.sum(np.multiply(self._y_tap[:, 1:], self._a[1:]), axis=1)
    #
    #     data = self._y_tap[:, 0]
    #     return data
    #
    # def reset_data_processor(self):
    #     self._x_tap.fill(0)
    #     self._y_tap.fill(0)
    #


# class RealtimeVrms(DataProcessor):
#     def __init__(self, fs=250, channel_num=8, interval_ms=250, offset_ms=0):  # interval in ms
#         super().__init__(data_processor_type=DataProcessorType.RealtimeVrms)
#         self.fs = fs
#         self.channel_num = channel_num
#         self.interval_ms = interval_ms
#         self.offset_ms = offset_ms
#         self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
#         self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))
#
#     # def init_buffer(self):
#     #     self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
#     #     self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))
#
#     def process_sample(self, data):
#         self.data_buffer[:, 1:] = self.data_buffer[:, : -1]
#         self.data_buffer[:, 0] = data
#         vrms = np.sqrt(1 / self.data_buffer_size * np.sum(np.square(self.data_buffer), axis=1))
#         # vrms = np.mean(self.data_buffer, axis=1)
#         # print(vrms)
#         return vrms
#
#     def reset_data_processor(self):
#         self.data_buffer.fill(0)


# class DataProcessorType(Enum):
#     NotchFilter = NotchFilter
#     RealtimeButterBandpass = RealtimeButterBandpass
#     RealtimeVrms = RealtimeVrms


# def get_processor_class(data_processor_type):
#     if data_processor_type == DataProcessorType.NotchFilter:
#         return NotchFilter
#     elif data_processor_type == DataProcessorType.RealtimeButterBandpass:
#         return RealtimeButterBandpass
#     elif data_processor_type == DataProcessorType.RealtimeVrms:
#         return RealtimeVrms


def run_data_processors(data, data_processor_pipeline: list[DataProcessor]):
    for data_processor in data_processor_pipeline:
        data = data_processor.process_buffer(data)

    return data


# if __name__ == '__main__':
#     pass
#     a = ButterworthBandpassFilter()
#     a.process_sample([1, 2])
#     print(a)
