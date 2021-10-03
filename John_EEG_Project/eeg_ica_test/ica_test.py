import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pickle
import json

from mne.preprocessing import create_eog_epochs, ICA, create_ecg_epochs

from utils.data_utils import RNStream, integer_one_hot, corrupt_frame_padding, time_series_static_clutter_removal
from sklearn.preprocessing import OneHotEncoder

file_path = "09_15_2021_02_10_20-Exp_eegmu-Sbj_ag-Ssn_1.dats"
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


eeg_raw = mne.io.RawArray(data['OpenBCI_Cyton_8'][0]*0.000001, info)
eeg_raw.plot(show_scrollbars=False)


# eog_evoked = create_eog_epochs(eeg_raw).average()
# eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()

ica = ICA(n_components=8, max_iter='auto', random_state=97)
ica.fit(eeg_raw)

eeg_raw.load_data()
ica.plot_sources(eeg_raw, show_scrollbars=False,start=20, stop=170)

ica.plot_components()

ica.plot_overlay(eeg_raw, exclude=[0,1,2,3], picks='eeg', start=2000, stop=20000)

# ica.plot_properties(eeg_raw, picks=[0, 1, 2, 3, 4, 5, 6, 7])
# eeg_raw.plot_psd(fmax=50)
# eeg_raw.plot(duration=5, n_channels=8)
ica.exclude = [0, 2, 3]


reconst_raw = eeg_raw.copy()
ica.apply(reconst_raw)

# eeg_raw.plot(n_channels=8,
#          show_scrollbars=False, title='Raw signal', start=50, duration=10)

reconst_raw.plot(n_channels=8,
                 show_scrollbars=False, title='recon signal', start=40, duration=10)

reconst_raw.plot(n_channels=8,
                 show_scrollbars=False, title='recon signal', start=50, duration=10)

reconst_raw.plot(n_channels=8,
                 show_scrollbars=False, title='recon signal', start=60, duration=10)

reconst_raw.plot(n_channels=8,
                 show_scrollbars=False, title='recon signal', start=70, duration=10)

del reconst_raw
# plt.plot(data['OpenBCI_Cyton_8'][0][0][1000:2000])
# plt.show()