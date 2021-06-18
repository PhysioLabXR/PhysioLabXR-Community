import numpy as np
from scipy.signal import iirnotch, filtfilt
from scipy import signal
import matplotlib.pyplot as plt

def notch_filter(data, w0, bw, fs, channel_format='first'):
    assert len(data.shape) == 2

    quality_factor = w0 / bw
    b, a = iirnotch(w0, quality_factor, fs)

    if channel_format == 'last':
        output_signal = np.array([filtfilt(b, a, data[:, i]) for i in range(data.shape[-1])])
    elif channel_format == 'first':
        output_signal = np.array([filtfilt(b, a, data[i, :]) for i in range(data.shape[0])])
    else:
        raise Exception('Unrecognized channgel format, must be either "first" or "last"')
    return output_signal





# Create/view notch filter
samp_freq = 1000  # Sample frequency (Hz)
notch_freq = 60.0  # Frequency to be removed from signal (Hz)
quality_factor = 30.0  # Quality factor
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
freq, h = signal.freqz(b_notch, a_notch, fs = samp_freq)
plt.figure('filter')
plt.plot( freq, 20*np.log10(abs(h)))

# Create/view signal that is a mixture of two frequencies
f1 = 17
f2 = 60
t = np.linspace(0.0, 1, 1000)
y_pure = np.sin(f1 * 2.0*np.pi*t) + np.sin(f2 * 2.0*np.pi*t)
plt.figure('result')
plt.subplot(211)
plt.plot(t, y_pure, color = 'r')
plt.show()
# apply notch filter to signal
y_notched = signal.filtfilt(b_notch, a_notch, y_pure)

# plot notch-filtered version of signal
plt.subplot(212)
plt.plot(t, y_notched, color = 'r')
plt.show()

# bandpass filter