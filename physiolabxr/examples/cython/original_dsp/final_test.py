#
import time

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from physiolabxr.utils.data_utils import signal_generator
from realtime_DSP import *
# signal0 = signal_generator(f=10, fs=5000, duration=1, amp=1)
signal1 = signal_generator(f=50, fs=5000, duration=10, amp=1)
signal2 = signal_generator(f=100, fs=5000, duration=10, amp=1)

signal3 = signal1+signal2


signal3 = np.transpose([signal3] * 1000).T

input_signal = signal3

rena_filter = RealtimeButterBandpass(lowcut=40, highcut=60, fs=5000, order=4, channel_num=1000)

plt.plot(signal3[0,0:3000])
plt.plot("python filter before")
plt.show()

rena_filter.reset_tap()

python_start_time = time.time()
output = rena_filter.process_buffer(signal3)
print("python cost: ", time.time()-python_start_time)
plt.plot(output[0,0:3000])
plt.title("python filter")
plt.show()






