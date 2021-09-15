import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pickle
import json
from utils.data_utils import RNStream, integer_one_hot, corrupt_frame_padding, time_series_static_clutter_removal
from sklearn.preprocessing import OneHotEncoder

file_path = "09_15_2021_02_10_20-Exp_eegmu-Sbj_ag-Ssn_1.dats"
stream = RNStream(file_path)
data = stream.stream_in(ignore_stream=('monitor1', '0'), jitter_removal=False)

n_channels = 8
sampling_freq = 250
ch_names = ["F3", "F4", "T3", "C3", "C4", "T4", "O1", "O2"]
ch_types = ['eeg'] * 8
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')
info['description'] = 'My custom dataset'
print(info)


eeg_raw = mne.io.RawArray(data['OpenBCI_Cyton_8'][0]*0.000001, info)
eeg_raw.plot()

# eeg_raw.plot_psd(fmax=50)
# eeg_raw.plot(duration=5, n_channels=8)

# plt.plot(data['OpenBCI_Cyton_8'][0][0][1000:2000])
# plt.show()