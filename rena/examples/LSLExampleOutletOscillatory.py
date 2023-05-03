"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import sys
import getopt
import numpy as np
import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock


def main(argv):
    srate = 2048  # updated sampling rate
    name = 'Dummy-8Chan'  # updated stream name
    print('Stream name is ' + name)
    type = 'EEG'
    n_channels = 2  # updated number of channels
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'

    try:
        opts, args = getopt.getopt(argv, "hs:c:n:t:", longopts=["srate=", "channels=", "name=", "type"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--srate"):
            srate = float(arg)
        elif opt in ("-c", "--channels"):
            n_channels = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--type"):
            type = arg

    # create a new stream info with updated parameters
    info = StreamInfo(name, type, n_channels, srate, 'float32', 'myuid1234')

    # create a stream outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # create a new sample with two sine wave oscillatory components at 50 Hz and 150 Hz
            t = time.time()  # current time in seconds
            sample = [np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 50 * t)]
            outlet.push_sample(sample)
        sent_samples += required_samples
        # sleep for a bit before sending the next batch of samples
        time.sleep(0.01)


if __name__ == '__main__':
    main(sys.argv[1:])
