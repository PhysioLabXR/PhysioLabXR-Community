import time
from collections import deque

import numpy as np
import zmq

topic = "python_zmq_my_stream_name"  # name of the publisher's topic / stream name
srate = 15  # we will send 15 frames per second
port = "5557"  # ZMQ port number

# we will send a random image of size 400x400 with 3 color channels
c_channels = 3
width = 400
height = 400
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
            socket.send_multipart([bytes(topic, "utf-8"), np.array(time.time()), mysample])  # send frame in the order: topic, timestamp, data
            send_times.append(time.time())
        sent_samples += required_samples
    # now send it and wait for a bit before trying again.
    time.sleep(0.01)
    if len(send_times) > 0:
        fps = len(send_times) / (np.max(send_times) - np.min(send_times))
        print("Send FPS is {0}".format(fps), end='\r')
    print(f'current timestamp is {time.time()}', end='\r', flush=True)