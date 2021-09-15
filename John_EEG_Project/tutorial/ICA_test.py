import numpy as np
import mne

n_channels = 32
sampling_freq = 200
ch_names = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'O1', 'O2']
ch_types = ['eeg'] * 7
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
info.set_montage('standard_1020')

info['description'] = 'My custom dataset'
info['bads'] = ['O1']  # Names of bad channels
print(info)

times = np.linspace(0, 1, sampling_freq, endpoint=False)
sine = np.sin(20 * np.pi * times)
cosine = np.cos(10 * np.pi * times)
data = np.array([sine, cosine])

info = mne.create_info(ch_names=['10 Hz sine', '5 Hz cosine'],
                       ch_types=['misc'] * 2,
                       sfreq=sampling_freq)

simulated_raw = mne.io.RawArray(data, info)
simulated_raw.plot(show_scrollbars=False, show_scalebars=False)


data = np.array([[0.2 * sine, 1.0 * cosine],
                 [0.4 * sine, 0.8 * cosine],
                 [0.6 * sine, 0.6 * cosine],
                 [0.8 * sine, 0.4 * cosine]])

simulated_epochs = mne.EpochsArray(data, info)
simulated_epochs.plot(picks='misc', show_scrollbars=False)
print(simulated_epochs.events[:, -1])


events = np.column_stack((np.arange(0, 800, sampling_freq),
                          np.zeros(4, dtype=int),
                          np.array([1, 2, 1, 2])))
event_dict = dict(condition_A=1, condition_B=2)
simulated_epochs = mne.EpochsArray(data, info, tmin=-0.3, events=events,
                                   event_id=event_dict)
simulated_epochs.plot(picks='misc', show_scrollbars=False, events=events,
                      event_id=event_dict)

# Create the Evoked object
evoked_array = mne.EvokedArray(data.mean(axis=0), info, tmin=-0.5,
                               nave=data.shape[0], comment='simulated')
print(evoked_array)
evoked_array.plot()

