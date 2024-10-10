#
import time

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
#
from physiolabxr.examples.cython.filter_test_all.filters import RealtimeButterBandpass
from physiolabxr.utils.data_utils import signal_generator

signal0 = signal_generator(f=10, fs=500, duration=20, amp=1)
signal1 = signal_generator(f=50, fs=500, duration=20, amp=1)
signal2 = signal_generator(f=100, fs=500, duration=20, amp=1)

signal3 = signal0+signal1+signal2


signal3 = np.transpose([signal3] * 10).T

input_signal = signal3

rena_filter = RealtimeButterBandpass(lowcut=10, highcut=60, fs=500, order=5, channel_num=10)


plt.plot(signal3[1,:])
plt.plot("cython filter before")
plt.show()


cython_start_time = time.time()
output_cython = rena_filter.process_buffer_cython(signal3)
print("cython cost: ", time.time()-cython_start_time)
plt.plot(output_cython[1,:])
plt.title("cython filter")
plt.show()

plt.plot(signal3[1,:])
plt.plot("python filter before")
plt.show()

rena_filter.reset_tap()

python_start_time = time.time()
output = rena_filter.process_buffer(signal3)
print("python cost: ", time.time()-python_start_time)
plt.plot(output[1,:])
plt.title("python filter")
plt.show()






