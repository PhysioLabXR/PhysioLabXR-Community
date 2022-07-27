import time
import numpy as np

from pylsl import StreamInlet, LostError, resolve_byprop

from exceptions.exceptions import LSLStreamNotFoundError, LSLChannelMismatchError
from rena import config
from stream_shared import lsl_continuous_resolver


class LSLInletInterface:
    """
    LSLInletInterface will  not try to connect to the outlet at init


    """
    def __init__(self, lsl_stream_name, num_chan):
        """

        @param lsl_stream_name:
        @param num_chan: the number of channels as in the preset. It will throw an error if when starting the sensor,
        it finds that the number of channels in the opened streams is different from this number, which is from the
        preset
        """
        # self.streams = resolve_byprop('name', lsl_stream_name, timeout=0.1)
        # if len(self.streams) < 1:
        #     raise AttributeError('Unable to find LSL Stream with given type {0}'.format(lsl_stream_name))
        # self.inlet = StreamInlet(self.streams[0])

        self.lsl_stream_name = lsl_stream_name
        self.lsl_num_chan = num_chan
        self.streams = None
        self.inlet = None
        pass

    def start_sensor(self):
        # connect to the sensor
        self.streams = resolve_byprop('name', self.lsl_stream_name, timeout=0.1)
        if len(self.streams) < 1:
            raise LSLStreamNotFoundError('Unable to find LSL Stream with given name {0}'.format(self.lsl_stream_name))
        self.inlet = StreamInlet(self.streams[0])
        self.inlet.open_stream()
        actual_num_channels = self.inlet.channel_count

        try:
            assert actual_num_channels == self.lsl_num_chan
        except AssertionError:
            self.inlet.close_stream()
            raise LSLChannelMismatchError(
                'The preset has {0} channel names, but the \n stream in LAN has {1} channels'.format(self.lsl_num_chan, actual_num_channels))

        print('LSLInletInterface: resolved, created and opened inlet for lsl stream with type ' + self.lsl_stream_name)

        # read the channel names is there's any
        # tell the sensor to start sending frames
    def is_stream_available(self):
        available_streams = [x.name() for x in lsl_continuous_resolver.results()]
        return self.lsl_stream_name in available_streams

    def process_frames(self):
        # return one or more frames of the sensor
        try:
            frames, timestamps = self.inlet.pull_chunk()
        except LostError:
            frames, timestamps = [], []
            pass  # TODO handle stream lost
        # frames = np.transpose(frames).astype()
        return np.transpose(frames), timestamps

    def stop_sensor(self):
        if self.inlet:
            self.inlet.close_stream()
        print('LSLInletInterface: inlet stream closed.')

    def info(self):
        return self.inlet.info()

    def get_num_chan(self):
        return self.lsl_num_channels

    def get_nominal_srate(self):
        return self.streams[0].nominal_srate()


def run_test():
    data = np.empty(shape=(config.UNITY_LSL_CHANNEL_SIZE, 0))
    print('Started streaming')
    start_time = time.time()
    while 1:
        try:
            new_data, _ = unityLSL_inferface.process_frames()
            print(new_data)
            if len(new_data) > 0:
                data = np.concatenate((data, new_data), axis=-1)  # get all data and remove it from internal buffer
        except KeyboardInterrupt:
            f_sample = data.shape[-1] / (time.time() - start_time)
            print('Stopped streaming, sampling rate = ' + str(f_sample))
            break
    return data


if __name__ == "__main__":
    unityLSL_inferface = LSLInletInterface()
    unityLSL_inferface.start_sensor()
    data = run_test()
    unityLSL_inferface.stop_sensor()
