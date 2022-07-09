from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt
from scipy.sparse.linalg import spsolve

b, a = iirnotch(60, 20, fs=1000)
print(b, a)

tap = np.zeros(len(a))
print(tap)
