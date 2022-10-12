from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt
from scipy.sparse.linalg import spsolve
import warnings
warnings.simplefilter("error")


class DataProcessor:
    def __init__(self):
        pass

    def process_sample(self, sample):
        return sample

    def process_buffer(self, data):
        output_buffer = np.empty(shape=data.shape)
        return output_buffer

    def reset_tap(self):
        pass


class RenaFilter(DataProcessor):

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.x_tap = None
        self.y_tap = None

    def process_sample(self, sample):
        # perform realtime filter with tap
        # push x
        self.x_tap[:, 1:] = self.x_tap[:, : -1]
        self.x_tap[:, 0] = sample
        # push y
        self.y_tap[:, 1:] = self.y_tap[:, : -1]
        # calculate new y

        try:
            self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
                               np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)
        except RuntimeWarning as e:
            # handel the overflow error in numpy
            print(e)
            self.reset_tap()

        sample = self.y_tap[:, 0]
        return sample

    def process_buffer(self, input_buffer):
        output_buffer = np.empty(shape=input_buffer.shape)
        for index in range(0, input_buffer.shape[1]):
            output_buffer[:, index] = self.process_sample(input_buffer[:, index])
        return output_buffer

    def reset_tap(self):
        self.x_tap.fill(0)
        self.y_tap.fill(0)


class RealtimeNotch(RenaFilter):
    def __init__(self, w0=60, Q=20, fs=250, channel_num=8):
        super().__init__()
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.channel_num = channel_num
        self.b, self.a = iirnotch(w0=w0, Q=self.Q, fs=self.fs)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

class RealtimeButter(RenaFilter):
    def __init__(self, fs=250, order=5, channel_num=8):
        super().__init__()
        self.fs = fs
        self.order = order
        self.channel_num = channel_num
        self.b = None
        self.a = None
        self.x_tap = None
        self.y_tap = None


class RealtimeButterBandpass(RealtimeButter):
    def __init__(self, lowcut=5, highcut=50, fs=250, order=5, channel_num=8):
        super().__init__(fs, order, channel_num)
        self.lowcut = lowcut
        self.highcut = highcut
        self.b, self.a = self.butter_bandpass(lowcut=self.lowcut, highcut=self.highcut, fs=self.fs, order=self.order)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a


class RealtimeButterLowpass(RealtimeButter):
    def __init__(self, highcut=5, fs=250, order=5, channel_num=8):
        super().__init__(fs, order, channel_num)
        self.highcut = highcut
        self.b, self.a = self.butter_lowpass(highcut=self.highcut, fs=self.fs, order=self.order)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def butter_lowpass(self, highcut, fs, order=5):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, btype='lowpass')
        return b, a


class RealtimeButterHighpass(RealtimeButter):
    def __init__(self, lowcut=5, fs=250, order=5, channel_num=8):
        super().__init__(fs, order, channel_num)
        self.lowcut = lowcut
        self.b, self.a = self.butter_highpass(lowcut=self.lowcut, fs=self.fs, order=self.order)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def butter_highpass(self, lowcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, low, btype='highpass')
        return b, a


# class RealtimeVrms(DataProcessor):
#     def __init__(self, fs=250, channel_num=8, interval_ms=250, offset_ms=0):  # interval in ms
#         super().__init__()
#         self.fs = fs
#         self.channel_num = channel_num
#         self.interval_ms = interval_ms
#         self.offset_ms = offset_ms
#         self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
#         self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))
#
#     def process_sample(self, data):
#         self.data_buffer[:, 1:] = self.data_buffer[:, : -1]
#         self.data_buffer[:, 0] = data
#         vrms = np.sqrt(1 / self.data_buffer_size * np.sum(np.square(self.data_buffer), axis=1))
#         # vrms = np.mean(self.data_buffer, axis=1)
#         # print(vrms)
#         return vrms
#
#     def reset_tap(self):
#         self.data_buffer.fill(0)
