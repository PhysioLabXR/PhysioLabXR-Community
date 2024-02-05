import time
from pylsl import StreamInfo, StreamOutlet, local_clock
from random import random as rand

# create a new stream info and outlet

stream_name = 'python_lsl_my_stream_name'

n_channels = 8
# using the local_clock() to track elapsed time
start_time = local_clock()
# track how many samples we have sent
sent_samples = 0
# set the sampling rate to 100 Hz
nominal_sampling_rate = 100

info = StreamInfo(stream_name, 'my_stream_type', n_channels, nominal_sampling_rate, 'float32',
                  'my_stream_id')
outlet = StreamOutlet(info)

# send data
while True:
    # calculate how many samples we need to send since the last call
    elapsed_time = local_clock() - start_time
    required_samples = int(nominal_sampling_rate * elapsed_time) - sent_samples
    for sample_ix in range(required_samples):
        mysample = [rand() * 10 for _ in range(n_channels)]
        # now send it
        outlet.push_sample(mysample)
    sent_samples += required_samples
    # now send it and wait for a bit before trying again.
    time.sleep(0.01)
