
from pylsl import StreamInlet, resolve_stream

import config

import numpy as np
import matplotlib.pyplot as plt

def main():
    print("looking for stream with type " + config.INFERENCE_LSL_NAME)
    streams = resolve_stream('type', config.INFERENCE_LSL_NAME)

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    eye_data_accumulated = np.empty((0, 2))
    timestamp_accumulated = []

    print('Entering inference look')
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        try:
            data, timestamps = inlet.pull_chunk(max_samples=config.EYE_TOTAL_POINTS_PER_INFERENCE)
            if len(data) > 0:
                timestamp_accumulated += timestamps  # accumulate timestamps
                eye_data_accumulated = np.concatenate([eye_data_accumulated, data])

            print(data.shape)
            if len(eye_data_accumulated) >= config.EYE_TOTAL_POINTS_PER_INFERENCE:
                # samples = np.reshape(eye_data_accumulated, newshape=(config.EYE_SAMPLES_PER_INFERENCE, config.EYE_INFERENCE_WINDOW_TIMESTEPS, -1))
                print(eye_data_accumulated.shape)
            # samples.append((timestamp, sample))
        except KeyboardInterrupt:
            pass
            # if len(samples) > 1:
            #     f_sample = 1. / ((samples[-1][0] - samples[0][0]) / len(samples))
            #     print('Interrupted, f_sample is ' + str(f_sample))
            # break

if __name__ == '__main__':
    main()