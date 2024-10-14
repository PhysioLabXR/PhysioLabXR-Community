"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import random
import sys
import getopt
import string

import time
from collections import deque
from random import random as rand

import numpy as np
import zmq
from pylsl import StreamInfo, StreamOutlet, local_clock

from physiolabxr.presets.PresetEnums import DataType


def LSLTestStream(stream_name, n_channels=81, srate=2048):
    print('Test stream name is ' + stream_name)
    type = 'EEG'
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

def SampleDefinedLSLStream(stream_name, sample, n_channels=81, srate=2048, dtype='double64'):
    print('Test stream name is ' + stream_name)
    type = 'EEG'
    info = StreamInfo(stream_name, type, n_channels, srate, dtype, 'someuuid1234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    # print("Example")
    # print(sample.dtype)
    sent_samples = 0
    sent_id = 0
    sent_data = []
    print(f'dim:{sample.shape}')
    # time.sleep(2)

    while True:
        if sent_samples > sample.shape[-1]:
            break
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            outlet.push_sample(sample[:, sent_id])
            sent_id += 1
            if sent_id == sample.shape[1]:
                break
        if sent_id == sample.shape[1]:
            break
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(1e-3)
    del outlet

def ZMQTestStream(stream_name, port, num_channels=3*800*800, srate=30, data_type=DataType.uint8, **kwargs):
    topic = stream_name

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
            samples = np.random.rand(required_samples * num_channels).reshape((required_samples, -1))
            samples = (samples * 255).astype(data_type.get_data_type())
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


def SampleDefinedZMQStream(stream_name, sample, port, srate=2048):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % port)

    print('Test stream name is ' + stream_name)

    print("now sending data...")
    start_time = time.time()
    sent_samples = 0
    sent_id = 0

    print(f'dim:{sample.shape}')
    while True:
        elapsed_time = time.time() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples

        if required_samples > 0:
            for _ in range(required_samples):
                socket.send_multipart([bytes(stream_name, "utf-8"),
                                       np.array(local_clock()),
                                       sample[:, sent_id].copy()])
                sent_id += 1
                if sent_id == sample.shape[1]:
                    sent_id = 0

        sent_samples += required_samples

        if sent_samples >= sample.shape[1]:
            break

        # now send it and wait for a bit before trying again.
        time.sleep(0.01)

    socket.close()
    context.term()