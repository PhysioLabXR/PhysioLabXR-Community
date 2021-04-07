from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt
from scipy.sparse.linalg import spsolve


class RealtimeNotch:
    def __init__(self, w0=60,  Q=20, sampling_frequency=250):
        b, a = iirnotch(w0=w0, Q=20, fs=1000)

        x_tap = np.zeros(len(b))
        y_tap = np.zeros(len(a))








