import numpy as np

from utils.data_utils import RNStream
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

file_path = 'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_16_52_54-Exp_realitynavigation-Sbj_0-Ssn_1.dats'
em_stream_name = 'Unity.RotationWheel.EventMarkers'

rns = RNStream(file_path)
data = rns.stream_in(ignore_stream=('monitor1', '0'))

data_stream = data[em_stream_name][0]
timestamps_stream = data[em_stream_name][1]
event_label_stream = data_stream[-1, :]
item_count = 5
offset = 2
trial_count = 24
trial_started_index = 0
for target_label in range(offset, offset + 5):
    target_onset_em = np.logical_and(event_label_stream == target_label, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
    started_em = event_label_stream >= 1
    started = False
    target_count = 0
    target_indices = []
    clean_count = 0
    trial_count = 0
    for i in range(1, len(target_onset_em)):
        if started_em[i - 1] == 0 and started_em[i] == 1:
            started = True
            trial_count += 1
            trial_started_index = i
        if started:
            if target_onset_em[i] == 1:
                target_count += 1
                target_indices.append(i)
        if started_em[i - 1] == 1 and started_em[i] == 0:
            if not (target_count >= 1 and target_count <= 2):
                print('bad trial, target count for label {0} is {1}, removing trail'.format(target_count, target_label))
                event_label_stream[trial_started_index:i] = - np.ones(i - trial_started_index)
                trial_count -= 1
            elif target_count == 2:
                print('cleaning event label at {0} from {1} to {2}'.format(target_indices[1], event_label_stream[target_indices[1]], 1))
                event_label_stream[target_indices[1]] = 1
                clean_count += 1
            started = False
            target_count = 0
            target_indices = []
    print('cleaned {0} event markers in {1} trials for label {2}'.format(clean_count, trial_count, target_label))

    # extract the time when the target label - 4 is present
    figure(figsize=(40, 6), dpi=80)

    target_onset_em = np.logical_and(event_label_stream == 1, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
    plt.scatter(timestamps_stream, target_onset_em, c='r')

    target_onset_em = np.logical_and(event_label_stream == target_label, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
    plt.scatter(timestamps_stream, target_onset_em, c='b')
    plt.title('Event label: {0}'.format(target_label))
    plt.show()
    assert np.count_nonzero(target_onset_em) == trial_count

data[em_stream_name][0][-1, :] = event_label_stream
rns = RNStream('{0} CLEANED.dats'.format(file_path.split('.')[0]))
rns.stream_out(data)