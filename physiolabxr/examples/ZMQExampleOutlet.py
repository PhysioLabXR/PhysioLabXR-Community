"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import random
import sys
import getopt
import string
from collections import deque

import numpy as np
import time
from random import random as rand

import pylsl
import zmq
from pylsl import local_clock


def main():
    topic = "CamCapture"
    srate = 120
    port = "5557"

    c_channels = 3
    width = 64
    height = 64
    n_channels = c_channels * width * height

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % port)

    # next make an outlet
    print("now sending data...")
    send_times = deque(maxlen=srate * 10)
    start_time = time.time()
    sent_samples = 0
    while True:
        elapsed_time = time.time() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        if required_samples > 0:
            samples = np.random.rand(required_samples * n_channels).reshape((required_samples, -1))
            samples = (samples * 255).astype(np.uint8)
            for sample_ix in range(required_samples):
                mysample = samples[sample_ix]
                socket.send_multipart([bytes(topic, "utf-8"), np.array(local_clock()), mysample])
                send_times.append(time.time())
            sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.01)
        if len(send_times) > 0:
            fps = len(send_times) / (np.max(send_times) - np.min(send_times))
            print("Send FPS is {0}".format(fps), end='\r')
        print(f'current timestamp is {pylsl.local_clock()}', end='\r', flush=True)

if __name__ == '__main__':
    main()
