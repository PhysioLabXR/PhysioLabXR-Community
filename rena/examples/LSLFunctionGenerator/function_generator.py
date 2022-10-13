"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import random
import sys
import getopt
import string
import numpy as np
import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock

from rena.utils.rena_dsp_utils import RealtimeButterBandpass
#
import time

import pyaudio
import matplotlib.pyplot as plt
from rena.utils.data_utils import signal_generator


def main(argv):
    letters = string.digits



    # signal0 = signal_generator(f=10, fs=5000, duration=1, amp=1)
    signal1 = signal_generator(f=50, fs=1000, duration=100, amp=1)
    signal2 = signal_generator(f=100, fs=1000, duration=100, amp=1)
    signal3 = signal1 + signal2
    signal3 = np.transpose([signal3] * 8).T
    input_signal = signal3
    rena_filter = RealtimeButterBandpass(lowcut=40, highcut=60, fs=1000, order=4, channel_num=8)

    srate = 1000
    name = 'Dummy-8Chan'
    print('Stream name is ' + name)
    type = 'EEG'
    n_channels = 8
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

    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo(name, type, n_channels, srate, 'float32', 'someuuid1234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0
    sample_index = 0
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # make a new random n_channels sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            # mysample = [rand()*10 for _ in range(n_channels)]
            mysample = input_signal[:, sample_index]
            mysample = rena_filter.process_sample(mysample)
            # now send it
            outlet.push_sample(mysample)
            sample_index += 1
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.001)


if __name__ == '__main__':
    main(sys.argv[1:])
