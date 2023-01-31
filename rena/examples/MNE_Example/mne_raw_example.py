import numpy as np
import mne
import inspect
from typing import Union
from mne.io import RawArray



# def add_events_to_data(data_array: Union[np.ndarray, RawArray], data_timestamp, events, event_names, event_filters, deviate=25e-2):
#     event_array = np.zeros(data_timestamp.shape)
#     event_ids = {}
#     deviant = 0
#     for i, e_filter in enumerate(event_filters):
#         filtered_events = np.array([e for e in events if e_filter(e)])
#         event_ts = [e.timestamp for e in filtered_events]
#
#
#         event_data_indices = [np.argmin(np.abs(data_timestamp - t)) for t in event_ts if np.min(np.abs(data_timestamp - t)) < deviate]
#
#         if len(event_data_indices) > 0:
#             deviate_event_count = len(event_ts) - len(event_data_indices)
#             if deviate_event_count > 0: print("Removing {} deviate events".format(deviate_event_count))
#             deviant += deviate_event_count
#
#             event_array[event_data_indices] = i + 1
#             event_ids[event_names[i]] = i + 1
#         else:
#             print(f'Unable to find event with name {event_names[i]}, skipping')
#     if type(data_array) is np.ndarray:
#         rtn = np.concatenate([data_array, np.expand_dims(event_array, axis=1)], axis=1)
#     elif type(data_array) is RawArray:
#         print()
#         stim_index = data_array.ch_names.index('stim')
#         rtn = data_array.copy()
#         rtn._data[stim_index, :] = event_array
#     else:
#         raise Exception(f'Unsupported data type {type(data_array)}')
#     return rtn, event_ids, deviant

# n_channels = 32
# sampling_freq = 200  # in Hertz
# info = mne.create_info(n_channels, sfreq=sampling_freq)
# print(info)
#
# ch_names = [f'MEG{n:03}' for n in range(1, 10)] + ['EOG001']
# ch_types = ['mag', 'grad', 'grad'] * 3 + ['eog']
# info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
# print(info)
#
#
# ch_names = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2']
# ch_types = ['eeg'] * 7
# info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
# info.set_montage('standard_1020')
#
# times = np.linspace(0, 1, sampling_freq, endpoint=False)
# sine = np.sin(20 * np.pi * times)
# cosine = np.cos(10 * np.pi * times)
# data = np.array([sine, cosine])
#
# info = mne.create_info(ch_names=['10 Hz sine', '5 Hz cosine'],
#                        ch_types=['misc'] * 2,
#                        sfreq=sampling_freq)
#
# simulated_raw = mne.io.RawArray(data, info)
# simulated_raw.plot(show_scrollbars=False, show_scalebars=False)

sampling_rate = 250
data_duration = 2
channel_num = 8
data_array = np.random.rand(8, data_duration * sampling_rate)

channel_types = ['eeg'] * 8
channel_names = [
    "Fp1",
    "Fp2",
    "C3",
    "C4",
    "P7",
    "P8",
    "O1",
    "O2"
]
montage = 'standard_1005'
info = mne.create_info(channel_names, sampling_rate, channel_types)
info['description'] = 'P300Speller'
print(info)

raw_data = mne.io.RawArray(data_array, info)
print(raw_data)
event_id = dict(target=1, non_target=2)

# info = mne.create_info(['STI'], raw_data.info['sfreq'], ['stim'])
# stim_raw = mne.io.RawArray(stim_data, info)
# raw.add_channels([stim_raw], force_update_info=True)

events = [1, 2, 1, 1, 2]
zeros = [0,0,0,0,0]
# events_timestamp_index = [100, 200, 300, 350, 400]
event_timestamps = [0.1, 0.3, 0.7, 1.0, 1.5, 1.8]
events_timestamp_index = np.array(event_timestamps)*250
# event_marker = np.vstack((events_timestamp, zeros,events))
# event_marker = np.array(event_marker).T

# raw_data.add_events(event_marker)



