"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream
from pylsl import *

def main():
    # first resolve an EEG stream on the lab network
    print("looking for an LSL Tobii stream...")
    streams = resolve_streams()
    print('streams resolved, creating inlets')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print('inlet created, entering pull data loop')

    samples = []
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        try:
            sample, timestamp = inlet.pull_sample()
            print('no data')
            print(timestamp, sample)
            samples.append((timestamp, sample))
        except KeyboardInterrupt:
            if len(samples) > 1:
                f_sample = 1. / ((samples[-1][0] - samples[0][0]) / len(samples))
                print('Interrupted, f_sample is ' + str(f_sample))
            break

if __name__ == '__main__':
    main()