import numpy as np
from scipy.signal import butter, freqz, iirnotch, filtfilt
from enum import Enum

from physiolabxr.exceptions.exceptions import UnsupportedErrorTypeError, DataProcessorEvokeFailedError, \
    DataProcessorInvalidBufferSizeError, DataProcessorInvalidFrequencyError, DaProcessorNotchFilterInvalidQError


class DataProcessorType(Enum):
    NotchFilter = 'NotchFilter'
    ButterworthLowpassFilter = 'ButterworthLowpassFilter'
    ButterworthHighpassFilter = 'ButterworthHighpassFilter'
    ButterworthBandpassFilter = 'ButterworthBandpassFilter'
    RootMeanSquare = 'RootMeanSquare'
    ClutterRemoval = 'ClutterRemoval'


# class SubDataProcessor(type):
#     pass

class DataProcessor:  # (QObject):
    # data_processor_valid_signal = pyqtSignal()
    # data_processor_activated_signal = pyqtSignal()

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

    def param_check(self):
        pass  # catch the known exception raised by the data processor manually

    def evoke_function(self):
        pass

    def evoke_data_processor(self):
        try:
            self.param_check()
            self.evoke_function()
            self.set_data_processor_valid(True)
            self.reset_data_processor()
        except Exception as e:
            self.set_data_processor_valid(False)
            print('Data Processor Evoke Failed Error: ' + str(e))
            raise DataProcessorEvokeFailedError(str(e))

    def set_data_processor_params(self, **params):
        pass

    def set_channel_num(self, channel_num):
        self.channel_num = channel_num

    def set_data_processor_activated(self, data_processor_activated):
        self.data_processor_activated = data_processor_activated
        # self.data_processor_activated_signal.emit()

    def set_data_processor_valid(self, data_processor_valid):
        self.data_processor_valid = data_processor_valid
        # self.data_processor_valid_signal.emit()

    def serialize_data_processor_params(self):
        return {key: value for key, value in vars(self).items() if not key.startswith('_')}


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

    # def evoke_data_processor(self):
    #     try:
    #         self.evoke_function()
    #         self.set_data_processor_valid(True)
    #         self.reset_data_processor()
    #     except Exception as e:
    #         self.set_data_processor_valid(False)
    #         print('Data Processor Evoke Failed Error: ' + str(e))
    #         raise DataProcessorEvokeFailedError(str(e))

    def evoke_function(self):
        pass

    def reset_data_processor(self):
        self._x_tap.fill(0)
        self._y_tap.fill(0)


class NotchFilter(IIRFilter):
    def __init__(self, w0: float = 10, Q: float = 20, fs: float = 0):
        super().__init__(data_processor_type=DataProcessorType.NotchFilter)
        self.w0 = w0
        self.Q = Q
        self.fs = fs

    def param_check(self):
        if self.Q <= 0:
            raise DaProcessorNotchFilterInvalidQError('Q must be greater than 0')
        if self.w0 <= 0 or self.w0 >= self.fs / 2:
            raise DataProcessorInvalidFrequencyError('w0 must be greater than 0 and less than fs/2 (Niquest Frequency)')

    def evoke_function(self):
        self._b, self._a = iirnotch(w0=self.w0, Q=self.Q, fs=self.fs)
        self._x_tap = np.zeros((self.channel_num, len(self._b)))
        self._y_tap = np.zeros((self.channel_num, len(self._a)))

    def set_data_processor_params(self, w0, Q, fs):
        self.w0 = w0
        self.Q = Q
        self.fs = fs


class ButterworthBandpassFilter(IIRFilter):
    def __init__(self, lowcut: float = 5, highcut: float = 10, fs: float = 0, order: int = 1):
        super().__init__(data_processor_type=DataProcessorType.ButterworthBandpassFilter)
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def param_check(self):
        if self.lowcut < 0 or self.highcut < 0:
            raise DataProcessorInvalidFrequencyError('lowcut and highcut must be greater than 0')
        if self.lowcut >= self.highcut:
            raise DataProcessorInvalidFrequencyError('lowcut must be less than highcut')
        if self.fs <= 0:
            raise DataProcessorInvalidFrequencyError('fs must be greater than 0')
        if self.order <= 0:
            raise DataProcessorInvalidFrequencyError('order must be greater than 0')
        if self.order > 10:
            raise DataProcessorInvalidFrequencyError('order must be <=10 (for stability reason)')
        if self.highcut >= self.fs / 2:
            raise DataProcessorInvalidFrequencyError('cutoff must be less than fs/2 (Niquest Frequency)')

    def evoke_function(self):
        self._b, self._a = self.butter_bandpass(lowcut=self.lowcut,
                                                highcut=self.highcut,
                                                fs=self.fs,
                                                order=self.order)
        self._x_tap = np.zeros((self.channel_num, len(self._b)))
        self._y_tap = np.zeros((self.channel_num, len(self._a)))

    def set_data_processor_params(self, lowcut, highcut, fs, order):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # self.evoke_data_processor()

    def butter_bandpass(self, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


class ButterworthLowpassFilter(IIRFilter):
    def __init__(self, cutoff: float = 0, fs: float = 0, order: int = 1):
        super().__init__(data_processor_type=DataProcessorType.ButterworthLowpassFilter)
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def param_check(self):
        if self.fs <= 0:
            raise DataProcessorInvalidFrequencyError('fs must be greater than 0 ')
        if self.order <= 0:
            raise DataProcessorInvalidFrequencyError('order must be greater than 0 ')
        if self.order > 10:
            raise DataProcessorInvalidFrequencyError('order must be <=10 (for stability reason) ')
        if self.cutoff <= 0:
            raise DataProcessorInvalidFrequencyError('cutoff must be greater than 0 ')
        if self.cutoff >= self.fs / 2:
            raise DataProcessorInvalidFrequencyError('cutoff must be less than fs/2 (Niquest Frequency) ')

    def evoke_function(self):
        self._b, self._a = self.butter_lowpass(cutoff=self.cutoff, fs=self.fs, order=self.order)
        self._x_tap = np.zeros((self.channel_num, len(self._b)))
        self._y_tap = np.zeros((self.channel_num, len(self._a)))

    def set_data_processor_params(self, cutoff, fs, order):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def butter_lowpass(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
        return b, a


class ButterworthHighpassFilter(IIRFilter):
    def __init__(self, cutoff: float = 0, fs: float = 0, order: int = 1):
        super().__init__(data_processor_type=DataProcessorType.ButterworthHighpassFilter)
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def param_check(self):
        if self.fs <= 0:
            raise DataProcessorInvalidFrequencyError('fs must be greater than 0 ')
        if self.order <= 0:
            raise DataProcessorInvalidFrequencyError('order must be greater than 0 ')
        if self.order > 10:
            raise DataProcessorInvalidFrequencyError('order must be <=10 (for stability reason) ')
        if self.cutoff <= 0:
            raise DataProcessorInvalidFrequencyError('cutoff must be greater than 0 ')
        if self.cutoff >= self.fs / 2:
            raise DataProcessorInvalidFrequencyError('cutoff must be less than fs/2 (Niquest Frequency) ')

    def evoke_function(self):
        self._b, self._a = self.butter_highpass(cutoff=self.cutoff, fs=self.fs, order=self.order)
        self._x_tap = np.zeros((self.channel_num, len(self._b)))
        self._y_tap = np.zeros((self.channel_num, len(self._a)))

    def set_data_processor_params(self, cutoff, fs, order):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    def butter_highpass(self, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
        return b, a


class RootMeanSquare(DataProcessor):

    def __init__(self, fs=0, window=0):  # interval in ms
        super().__init__(data_processor_type=DataProcessorType.RootMeanSquare)
        self.fs = fs
        self.window = window

        self._data_buffer_size = 0
        self._data_buffer = np.empty(0)

    def param_check(self):
        data_buffer_size = round(self.fs * self.window * 0.001)

        if data_buffer_size <= 0:
            raise DataProcessorInvalidBufferSizeError('data_buffer_size must be greater than 0 ')

    def evoke_function(self):
        self._data_buffer_size = round(self.fs * self.window * 0.001)
        self._data_buffer = np.zeros((self.channel_num, self._data_buffer_size))

    def set_data_processor_params(self, fs, window):
        self.fs = fs
        self.window = window

    def process_sample(self, data):
        self._data_buffer[:, 1:] = self._data_buffer[:, : -1]
        self._data_buffer[:, 0] = data
        data = np.sqrt(1 / self._data_buffer_size * np.sum(np.square(self._data_buffer), axis=1))
        # vrms = np.mean(self.data_buffer, axis=1)
        # print(vrms)
        return data

    def reset_data_processor(self):
        self._data_buffer.fill(0)


class ClutterRemoval(DataProcessor):

    def __init__(self, signal_clutter_ratio=0):  # interval in ms
        super().__init__(data_processor_type=DataProcessorType.ClutterRemoval)
        self.signal_clutter_ratio = signal_clutter_ratio
        self._clutter = None

    def evoke_function(self):
        pass

    def set_data_processor_params(self, signal_clutter_ratio):
        self.signal_clutter_ratio = signal_clutter_ratio

    def process_sample(self, data):
        if self._clutter is None:
            self._clutter = data
        else:
            self._clutter = self.signal_clutter_ratio * self._clutter + (1 - self.signal_clutter_ratio) * data

        data = data - self._clutter
        return data

    def reset_data_processor(self):
        self._clutter = None


data_processor_lookup_table = {
    DataProcessorType.NotchFilter: NotchFilter,
    DataProcessorType.ButterworthLowpassFilter: ButterworthLowpassFilter,
    DataProcessorType.ButterworthHighpassFilter: ButterworthHighpassFilter,
    DataProcessorType.ButterworthBandpassFilter: ButterworthBandpassFilter,
    DataProcessorType.RootMeanSquare: RootMeanSquare,
    DataProcessorType.ClutterRemoval: ClutterRemoval
}


def run_data_processors(data, data_processor_pipeline: list[DataProcessor]):
    for data_processor in data_processor_pipeline:
        # if data_processor.data_processor_valid and data_processor.data_processor_activated:
        data = data_processor.process_buffer(data)

    return data

# if __name__ == '__main__':
#     a = NotchFilter(w0=60, Q=20, fs=300)
#     c = ButterworthBandpassFilter()
#     dict_record = c.__dict__
#     b = NotchFilter()
#     for key, value in dict_record.items():
#         setattr(b, key, value)
#     print('John')
