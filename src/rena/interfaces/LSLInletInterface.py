import time
import numpy as np

from pylsl import StreamInlet, LostError, resolve_byprop

from src.rena import config


class LSLInletInterface:

    def __init__(self, lsl_data_type):
        self.streams = resolve_byprop('name', lsl_data_type, timeout=0.1)
        if len(self.streams) < 1:
            raise AttributeError('Unable to find LSL Stream with given type {0}'.format(lsl_data_type))
        self.inlet = StreamInlet(self.streams[0])

        self.lsl_data_type = lsl_data_type
        self.lsl_num_channels = self.inlet.channel_count
        pass

    def start_sensor(self):
        # connect to the sensor
        self.streams = resolve_byprop('name', self.lsl_data_type, timeout=0.1)
        if len(self.streams) < 1:
            raise AttributeError('Unable to find LSL Stream with given type {0}'.format(self.lsl_data_type))
        self.inlet = StreamInlet(self.streams[0])
        self.inlet.open_stream()
        print('LSLInletInterface: resolved, created and opened inlet for lsl stream with type ' + self.lsl_data_type)

        # read the channel names is there's any
        # tell the sensor to start sending frames

    def process_frames(self):
        # return one or more frames of the sensor
        try:
            frames, timestamps = self.inlet.pull_chunk()
        except LostError:
            frames, timestamps = [], []
            pass  # TODO handle stream lost
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
