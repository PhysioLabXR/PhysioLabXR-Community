import pickle

import mne

from MGH.TMS_Project.exg_utils import my_range, collect_evoked_responses
from utils.data_utils import RNStream
import numpy as np
file_path = "07_06_2022_16_55_42-Exp_TMS-Sbj_Aapo-Ssn_0.dats"
stream = RNStream(file_path)
data = stream.stream_in(ignore_stream=('monitor1', '0'), jitter_removal=False)
fs = 5000
# interval = 1/5000
#
import matplotlib.pyplot as plt


raw_data = data['BrainAmp_ExG'][0]
# raw_data_timestamp = data['BrainAmp_ExG'][1]
raw_data_timestamp = my_range(0, raw_data.shape[1], 1/fs)

event_markers =data['BrainAmp_EventMarker'][0]
# event_markers_timestamp = data['BrainAmp_EventMarker'][1]
event_markers_timestamp = my_range(0, event_markers.shape[1], 1/fs)

evoked_responses = collect_evoked_responses(raw_data, raw_data_timestamp, event_markers, event_markers_timestamp)

with open('aapo_evoked_responses.pickle', 'wb') as handle:
    pickle.dump(evoked_responses, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('John')
for evoked_response in evoked_responses:
    plt.plot(evoked_response['evoked_ts'], evoked_response['evoked_response'].T-np.mean(evoked_response['evoked_response']))

plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Time in ms')
plt.ylabel('mV')
plt.title('MEPs')
plt.show()

# def onset_detection():
# zero_index = 100
calibration_start= 120
calibration_end = 200
detection_start_index = 200
detection_end_index = 300



active_evoked_responses = []
for evoked_response in evoked_responses:
    pick_range = evoked_response['evoked_response'][: ,detection_start_index:detection_end_index]
    if np.max(pick_range)-np.min(pick_range)>=60:
        active_evoked_responses.append(evoked_response)


for evoked_response in active_evoked_responses:
    plt.plot(evoked_response['evoked_ts'], evoked_response['evoked_response'].T-np.mean(evoked_response['evoked_response']))

plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Time in ms')
plt.ylabel('mV')
plt.title('active MEPs')
plt.show()

# for selected


# n_channels = 1
# sampling_freq = 5000
# ch_names = ["EMG_RH"]
# ch_types = ['emg']
# info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
#
# # info.set_montage('standard_1020')
# info['description'] = 'My custom dataset'
# # print(info)
# emg_data = data['BrainAmp_ExG'][0]
# emg_raw = mne.io.RawArray(emg_data, info)
#
# print('John')
