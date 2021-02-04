
from pylsl import StreamInlet, resolve_stream

import config

import numpy as np

def main():
    # first resolve an EEG stream on the lab network
    print("looking for an samples stream...")
    streams = resolve_stream('type', config.INFERENCE_LSL_NAME)

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    samples = []
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        try:
            data, timestamp = inlet.pull_chunk(max_samples=config.EYE_SAMPLES_PER_INFERENCE * config.EYE_INFERENCE_WINDOW_TIMESTEPS)
            np.reshape(data, newshape=(config.EYE_SAMPLES_PER_INFERENCE, config.EYE_INFERENCE_WINDOW_TIMESTEPS, -1))
            # print(timestamp, sample)
            # samples.append((timestamp, sample))
        except KeyboardInterrupt:
            if len(samples) > 1:
                f_sample = 1. / ((samples[-1][0] - samples[0][0]) / len(samples))
                print('Interrupted, f_sample is ' + str(f_sample))
            break

if __name__ == '__main__':
    main()