import numpy as np

from utils.data_utils import RNStream
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

file_path = 'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_17_03_52-Exp_realitynavigation-Sbj_0-Ssn_2.dats'
em_stream_name = 'Unity.RotationWheel.EventMarkers'
target_label = 1

def plot_stream(stream, timestamps):
    timestamps = timestamps - timestamps[0]  # baseline the timestamps
    plt.plot(timestamps, stream)
    plt.xlabel('Time (sec)')
    plt.show()


rns = RNStream(file_path)
data = rns.stream_in(ignore_stream=('monitor1', '0'))
# plot_stream(data['Unity.VisualSearch.EventMarkers'][0][-1, :], data['Unity.VisualSearch.EventMarkers'][1])
# plot_stream(data['Unity.RotationWheel.EventMarkers'][0][-1, :], data['Unity.RotationWheel.EventMarkers'][1])

data_stream = data[em_stream_name][0]
timestamps_stream = data[em_stream_name][1]
event_label_stream = data_stream[-1, :]

# extract the time when the target label - 4 is present
figure(figsize=(30, 6), dpi=80)

target_label = 1
target_onset_em = np.logical_and(event_label_stream == target_label, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
plt.scatter(timestamps_stream, target_onset_em, c='r')

target_label = 3
target_onset_em = np.logical_and(event_label_stream == target_label, np.concatenate([np.array([0]), np.diff(event_label_stream)]) != 0)
plt.scatter(timestamps_stream, target_onset_em, c='b')
plt.show()


# a = np.count_nonzero(target_onset_em)
# target_onset_ts = timestamps_stream[target_onset_em]

