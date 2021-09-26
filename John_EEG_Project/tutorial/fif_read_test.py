import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy
from scipy.io import savemat

file_name = '../data/sm04/meeg/raw/sm04_007_raw.fif'
raw = mne.io.read_raw_fif(fname=file_name)
raw.pick_types(eeg=True)
dig = raw.info['dig']
# save dig to data dir
print(dig[0]['kind'])
dig_eeg = dig[31:]
dig_eeg_dict = {}
for channel_dig in dig_eeg:
    channel_id = channel_dig['ident']
    dig_eeg_dict['EEG_'+str(channel_id)] = channel_dig['r']

# file_name = '../data/sm04/mri/T1-neuromag/sets/COR-raij-100510-191654.fif'
# transform = mne.read_trans(file_name)
# fiff = mne.io.fiff_open(file_name)
# # find coordinate
#
# data file
data_file = '../data/sm04/meeg/avg_maxfeog8/sm04_AVG_Single_Task_offl.fif'
evoked_data = mne.read_evokeds(data_file)

train = evoked_data[0]
finger = evoked_data[1]
train.pick_types(eeg=True)
finger.pick_types(eeg=True)

evoked_data_dict = {}
train_data_dict = {}
finger_data_dict = {}

missing_channel = [0, 39]

i = 0
for index in range(0, len(finger.data)+len(missing_channel)):
    if index in missing_channel:
        train_data_dict['EEG_'+str(index)] = np.array([])
        finger_data_dict['EEG_'+str(index)] = np.array([])
    else:
        train_data_dict['EEG_'+str(index)] = np.array(train.data[i])
        finger_data_dict['EEG_'+str(index)] = np.array(finger.data[i])
        i += 1

train_ts = train.times
finger_ts = finger.times

train_dict = {'data':train_data_dict, 'time_stamp':train_ts}
finger_dict = {'data':finger_data_dict, 'time_stamp':finger_ts}

evoked_data_dict = {'train': train_dict, 'finger': finger_dict}
dig_eeg_dict = dig_eeg_dict


# with open('evoked_data_dict.m', 'wb') as handle:
#     pickle.dump(evoked_data_dict, handle, protocol=4)

savemat("evoked_data_dict.mat", dict(evoked_data_dict))
savemat("dig_eeg_dict.mat", dig_eeg_dict)
# electrode extraction


# scipy.io.savemat('./test.mat',bad)


# elctrode EEG 65 is 61 we subtract 4 after 60
# sample_data_folder = mne.datasets.sample.data_path()
