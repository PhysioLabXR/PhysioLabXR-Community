#
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
p = pyaudio.PyAudio()

volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 0.01   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
samples = volume*(np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

plt.plot(samples)
plt.show()
#