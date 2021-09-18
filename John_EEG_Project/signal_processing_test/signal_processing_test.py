import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pickle
import json
from numpy.fft import fft, fftfreq
from scipy import signal

from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter
from scipy import signal
import mne
from mne.preprocessing import create_eog_epochs, ICA, create_ecg_epochs
from scipy.signal import butter

from utils.data_utils import RNStream, integer_one_hot, corrupt_frame_padding, time_series_static_clutter_removal
from sklearn.preprocessing import OneHotEncoder

file_path = "C:/Recordings/eeg_project/John_test_SLRBS_TEST/09_16_2021_18_04_55-Exp_SLRBS_BOTH_HAND-Sbj_John-Ssn_1.dats"
stream = RNStream(file_path)
data = stream.stream_in(ignore_stream=('monitor1', '0'), jitter_removal=False)

n_channels = 8
sampling_freq = 250
ch_names = ["F3", "F4", "T3", "C3", "C4", "T4", "O1", "O2"]
ch_types = ['eeg']*8
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')
info['description'] = 'My custom dataset'
print(info)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


b_notch, a_notch = signal.iirnotch(w0=60, Q=20, fs=250.0)
b_butter, a_butter = butter_bandpass(lowcut=7, highcut=13, fs=250, order=6)

eeg = signal.filtfilt(b_notch, a_notch, data['OpenBCI_Cyton_8'][0]*0.000001, padlen=0)
eeg = signal.filtfilt(b_butter, a_butter, eeg, padlen=0)

eeg_raw = mne.io.RawArray(eeg, info)
eeg_raw.plot(show_scrollbars=False, start=70, duration=10)


# eog_evoked = create_eog_epochs(eeg_raw).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()

ica = ICA(n_components=8, max_iter=300, random_state=97)
ica.fit(eeg_raw)

eeg_raw.load_data()
ica.plot_sources(eeg_raw, show_scrollbars=False,start=70, stop=80)

ica.plot_components()

ica.plot_overlay(eeg_raw, exclude=[0, 1, 2, 3], picks='eeg', start=2000, stop=3000)

# ica.plot_properties(eeg_raw, picks=[0, 1, 2, 3, 4, 5, 6, 7])
# eeg_raw.plot_psd(fmax=50)
# eeg_raw.plot(duration=5, n_channels=8)
ica.exclude = [0, 1, 5, 6, 7]


reconst_raw = eeg_raw.copy()
ica.apply(reconst_raw)

# eeg_raw.plot(n_channels=8,
#          show_scrollbars=False, title='Raw signal', start=50, duration=10)

reconst_raw.plot(n_channels=8,
                 show_scrollbars=False, title='recon signal', start=70, duration=20)

# reconst_raw.plot(n_channels=8,
#                  show_scrollbars=False, title='recon signal', start=50, duration=10)
#
# reconst_raw.plot(n_channels=8,
#                  show_scrollbars=False, title='recon signal', start=60, duration=10)
#
# reconst_raw.plot(n_channels=8,
#                  show_scrollbars=False, title='recon signal', start=70, duration=10)
del reconst_raw
