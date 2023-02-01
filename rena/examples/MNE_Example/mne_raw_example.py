import math

import numpy as np
import mne
import inspect
from typing import Union
from mne.io import RawArray


def add_events_to_data(data_array: Union[np.ndarray, RawArray], data_timestamp, events, event_names, event_filters,
                       deviate=25e-2):
    event_array = np.zeros(data_timestamp.shape)
    event_ids = {}
    deviant = 0
    for i, e_filter in enumerate(event_filters):
        filtered_events = np.array([e for e in events if e_filter(e)])
        event_ts = [e.timestamp for e in filtered_events]

        event_data_indices = [np.argmin(np.abs(data_timestamp - t)) for t in event_ts if
                              np.min(np.abs(data_timestamp - t)) < deviate]

        if len(event_data_indices) > 0:
            deviate_event_count = len(event_ts) - len(event_data_indices)
            if deviate_event_count > 0: print("Removing {} deviate events".format(deviate_event_count))
            deviant += deviate_event_count

            event_array[event_data_indices] = i + 1
            event_ids[event_names[i]] = i + 1
        else:
            print(f'Unable to find event with name {event_names[i]}, skipping')
    if type(data_array) is np.ndarray:
        rtn = np.concatenate([data_array, np.expand_dims(event_array, axis=1)], axis=1)
    elif type(data_array) is RawArray:
        print()
        stim_index = data_array.ch_names.index('stim')
        rtn = data_array.copy()
        rtn._data[stim_index, :] = event_array
    else:
        raise Exception(f'Unsupported data type {type(data_array)}')
    return rtn, event_ids, deviant


def generate_mne_stim_channel(data_ts, event_ts, events, deviate=25e-2):
    stim_array = np.zeros((1, data_ts.shape[0]))

    # event_data_indices = []
    # for t_e in event_ts:
    #     min_ts = math.inf
    #     min_ts_index = None
    #     for i, t_d in enumerate(data_ts):
    #         t_diff = abs(t_e - t_d)
    #         if t_diff < min_ts:
    #             min_ts = t_diff
    #             min_ts_index = i
    #     event_data_indices.append(min_ts_index)

    event_data_indices = [np.argmin(np.abs(data_ts - t)) for t in event_ts if
                              np.min(np.abs(data_ts - t)) < deviate]

    for index, event_data_index in enumerate(event_data_indices):
        stim_array[0, event_data_index] = events[index]


    return stim_array

def add_stim_channel_to_raw_array(raw_array, stim_data, stim_channel_name = 'STI'):
    # if len(stim_data.shape)==1:
    #     stim_data = stim_data.reshape(1,stim_data.shape[0])
    info = mne.create_info([stim_channel_name], raw_array.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw_array.add_channels([stim_raw], force_update_info=True)


sampling_rate = 250
data_duration = 2
channel_num = 8
data_array = np.random.rand(8, data_duration * sampling_rate)
data_ts = np.linspace(0, 2.0, num=500, endpoint=False)
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
event_id = {'target':1, 'non_target':2}
info = mne.create_info(channel_names, sampling_rate, channel_types)
info['description'] = 'P300Speller'
print(info)

raw_data = mne.io.RawArray(data_array, info)
print(raw_data)


# info = mne.create_info(['STI'], raw_data.info['sfreq'], ['stim'])
# stim_raw = mne.io.RawArray(stim_data, info)
# raw.add_channels([stim_raw], force_update_info=True)



events = [1, 2, 1, 1, 2, 1]
zeros = [0, 0, 0, 0, 0]
# events_timestamp_index = [100, 200, 300, 350, 400]
event_ts = [0.1001, 0.3001, 0.7005, 1.0, 1.5, 1.8]
events_timestamp_index = np.array(event_ts) * 250
# event_marker = np.vstack((events_timestamp, zeros,events))
# event_marker = np.array(event_marker).T

stim_data = generate_mne_stim_channel(data_ts=data_ts, event_ts=event_ts, events=events)
add_stim_channel_to_raw_array(raw_array=raw_data, stim_data=stim_data)
flashing_events = mne.find_events(raw_data, stim_channel='STI')
epochs = mne.Epochs(raw_data, flashing_events, tmin=-0.1, tmax=0.1, baseline=(-0.1, 0), event_id=event_id, preload=True)
# target_epochs = epochs['target']
# non_target_epochs = epochs['non_target']
# epochs.get_data()
# epochs_eeg = epochs.pick_types(eeg=True)
evoked = epochs.average(by_event_types = True)




