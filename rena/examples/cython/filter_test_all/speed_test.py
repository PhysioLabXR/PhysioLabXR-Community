#
import time

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import scipy

from rena.examples.cython.filter_test_all.filters import RealtimeButterBandpass
from rena.utils.data_utils import signal_generator
from rena.utils.realtime_DSP import RealtimeButterBandpass

# signal0 = signal_generator(f=10, fs=5000, duration=1, amp=1)
signal1 = signal_generator(f=50, fs=5000, duration=10, amp=1)
signal2 = signal_generator(f=100, fs=5000, duration=10, amp=1)

signal3 = signal1+signal2


signal3 = np.transpose([signal3] * 1000).T

input_signal = signal3

rena_filter = RealtimeButterBandpass(lowcut=40, highcut=60, fs=5000, order=4, channel_num=1000)

#############
# scipy_out = scipy.signal.filtfilt(rena_filter.b, rena_filter.a, input_signal)
# plt.plot(scipy_out.T)
# plt.show()
##############

# plt.plot(signal3[1,:])
# plt.plot("cython filter before")
# plt.show()
#
#
# cython_start_time = time.time()
# output_cython = rena_filter.process_buffer_cython(signal3)
# print("cython cost: ", time.time()-cython_start_time)
# plt.plot(output_cython[1,:])
# plt.title("cython filter")
# plt.show()

plt.plot(signal3[0,:])
plt.plot("python filter before")
plt.show()

rena_filter.reset_tap()

python_start_time = time.time()
output = rena_filter.process_buffer(signal3)
print("python cost: ", time.time()-python_start_time)
plt.plot(output[0,:])
plt.title("python filter")
plt.show()






