"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream


def main():
    # first resolve an EEG stream on the lab network
    print("looking for LSL stream...")
    streams = resolve_stream('name', 'obci_eeg1')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    samples = []
    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        try:
            sample, timestamp = inlet.pull_sample()
            print(timestamp, sample)
            print(len(sample))
            samples.append((timestamp, sample))
        except KeyboardInterrupt:
            if len(samples) > 1:
                f_sample = 1. / ((samples[-1][0] - samples[0][0]) / len(samples))
                print('Interrupted, f_sample is ' + str(f_sample))
            break

if __name__ == '__main__':
    main()