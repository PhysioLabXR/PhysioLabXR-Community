import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt



class DataProcessor:
    def __init__(self):
        pass

    def process_sample(self, data):

        return data

    def process_buffer(self, data):
        output_buffer = np.empty(shape=data.shape)
        for index in range(0, data.shape[1]):
            output_buffer[:, index] = self.process_sample(data[:, index])
        return output_buffer

    def reset_tap(self):
        pass


class IIRFilter(DataProcessor):

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.x_tap = None
        self.y_tap = None

    def process_sample(self, data):
        # perform realtime filter with tap

        # push x
        self.x_tap[:, 1:] = self.x_tap[:, : -1]
        self.x_tap[:, 0] = data
        # push y
        self.y_tap[:, 1:] = self.y_tap[:, : -1]
        # calculate new y
        self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
                           np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)

        data = self.y_tap[:, 0]
        return data

    def reset_tap(self):
        self.x_tap.fill(0)
        self.y_tap.fill(0)

class RealtimeNotch(DataProcessor):
    def __init__(self, w0=60, Q=20, fs=250, channel_num=8):
        super().__init__()
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.channel_num = channel_num
        self.b, self.a = iirnotch(w0=w0, Q=self.Q, fs=self.fs)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def process_sample(self, data):
        # perform realtime filter with tap

        # push x
        self.x_tap[:, 1:] = self.x_tap[:, : -1]
        self.x_tap[:, 0] = data
        # push y
        self.y_tap[:, 1:] = self.y_tap[:, : -1]
        # calculate new y
        self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
                           np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)

        data = self.y_tap[:, 0]
        return data

    def reset_tap(self):
        self.x_tap.fill(0)
        self.y_tap.fill(0)


class RealtimeButterBandpass(DataProcessor):
    def __init__(self, lowcut=5.0, highcut=50, fs=250, order=5, channel_num=8):
        super().__init__()
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.channel_num = channel_num
        self.b, self.a = self.butter_bandpass(lowcut=self.lowcut, highcut=self.highcut, fs=self.fs, order=self.order)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def process_sample(self, data):
        # perform realtime filter with tap

        # push x
        self.x_tap[:, 1:] = self.x_tap[:, : -1]
        self.x_tap[:, 0] = data
        # push y
        self.y_tap[:, 1:] = self.y_tap[:, : -1]
        # calculate new y
        self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
                           np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)

        data = self.y_tap[:, 0]
        return data

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def reset_tap(self):
        self.x_tap.fill(0)
        self.y_tap.fill(0)


class RealtimeVrms(DataProcessor):
    def __init__(self, fs=250, channel_num=8, interval_ms=250, offset_ms=0):  # interval in ms
        super().__init__()
        self.fs = fs
        self.channel_num = channel_num
        self.interval_ms = interval_ms
        self.offset_ms = offset_ms
        self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
        self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))

    # def init_buffer(self):
    #     self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
    #     self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))

    def process_sample(self, data):
        self.data_buffer[:, 1:] = self.data_buffer[:, : -1]
        self.data_buffer[:, 0] = data
        vrms = np.sqrt(1 / self.data_buffer_size * np.sum(np.square(self.data_buffer), axis=1))
        # vrms = np.mean(self.data_buffer, axis=1)
        # print(vrms)
        return vrms

    def reset_tap(self):
        self.data_buffer.fill(0)


if __name__ == '__main__':
    a = RealtimeButterBandpass(lowcut=0.5, highcut=50, fs=250, order=4, channel_num=8)

    print("John")