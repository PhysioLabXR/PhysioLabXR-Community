import pyxdf
import matplotlib.pyplot as plt
import numpy as np

# data, header = pyxdf.load_xdf('C:/Recordings/CurrentStudy/exp1/untitled.xdf')
data, header = pyxdf.load_xdf('C:/Recordings/CurrentStudy/exp1/untitled.xdf')

for stream in data:
    y = stream['time_series']

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        plt.plot(stream['time_stamps'], y)
    else:
        raise RuntimeError('Unknown stream format')

plt.show()