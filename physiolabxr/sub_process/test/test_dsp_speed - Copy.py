from physiolabxr.utils.realtime_DSP import *
import numpy as np

i = 1
filter = RealtimeButterBandpass(lowcut=5, highcut=50, fs=5000, order=5, channel_num=50)
while True:
    data = np.random.normal(0, 1, size=(1, 50))
    data = filter.process_sample(data)

    i += 1
    if i % 10000 == 0:
        print(i)
        filter.reset_tap()
