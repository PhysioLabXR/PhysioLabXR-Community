"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import random
import sys
import getopt
import string

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock


def LSLTestStream(stream_name):
    letters = string.digits
    id = (''.join(random.choice(letters) for i in range(3)))

    srate = 2048
    print('Test stream name is ' + stream_name)
    type = 'EEG'
    n_channels = 81
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'
    info = StreamInfo(stream_name, type, n_channels, srate, 'float32', 'someuuid1234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # make a new random n_channels sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            mysample = [rand() for _ in range(n_channels)]
            # now send it
            outlet.push_sample(mysample)
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(1e-3)