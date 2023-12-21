import copy
import os
from collections import defaultdict
from typing import Union, Dict, List

import numpy as np
import scipy
from scipy.signal import spectrogram


def get_event_locked_data(event_marker, data, events_of_interest, tmin, tmax, srate, return_last_event_time=False, event_channel=0, verbose=None, **kwargs):
    """
    this function is used to get event locked data from a single modality or multiple modalities

    @param event_marker: tuple of event marker and its timestamps
    @param data: tuple of data and its timestamps, can be a dictionary of multiple modalities
    @param events_of_interest: iterable of event markers with which to get event aligned data
    @param tmin: time before event marker to include in the epoch
    @param tmax: time after event marker to include in the epoch
    @param srate: sampling rate of the datar
    @param reject: int, reject the epoch with peak-to-peak amplitude greater than this value
    @param return_last_event_time: whether to return the time of the last found in the data
    @param verbose: whether to print out the number of events found for each event marker

    @return: dictionary of event marker and its corresponding event locked data. The keys are the event markers
            if return_last_event_time is True, return the time of the last found event is also returned
    """
    # check if data is multi-modal
    if isinstance(data, dict):
        assert len(data) > 0, 'data must be a non-empty dictionary'
        assert set(data.keys()) == set(tmin.keys()) == set(tmax.keys()) == set(srate.keys()), 'data, tmin, and tmax must have the same keys'
        locked_data_modalities = {}
        for modality, v in data.items():
            _tmin = tmin[modality]
            _tmax = tmax[modality]
            _data = v
            _srate = srate[modality]
            args = {'event_marker': event_marker, 'events_of_interest': events_of_interest, 'event_channel': event_channel, 'data': _data,
                    'tmin': _tmin, 'tmax': _tmax, 'srate': _srate, 'return_last_event_time': True}
            _locked_data, latest_event_start_time = _get_event_locked_data(**{**args, **kwargs})
            locked_data_modalities[modality] = _locked_data

        # make the modalities be the secondary keys and the event markers be the primary keys
        rtn = {e: {m: locked_data_modalities[m][e] for m in locked_data_modalities.keys() if e in locked_data_modalities[m]} for e in events_of_interest}
        if return_last_event_time:
            return rtn, latest_event_start_time
        else:
            return rtn
    else:
        args = {'event_marker': event_marker, 'events_of_interest': events_of_interest, 'event_channel': event_channel, 'data': data,
                'tmin': tmin, 'tmax': tmax, 'srate': srate, 'return_last_event_time': True}
        return _get_event_locked_data(**{**args, **kwargs})


def _get_event_locked_data(event_marker, event_channel, data, events_of_interest, tmin, tmax, srate, return_last_event_time=False, verbose=None, reject=None):
    """
    @param event_marker: tuple of event marker and its timestamps
    @param data: tuple of data and its timestamps
    @param events_of_interest: iterable of event markers with which to get event aligned data
    @param return_last_event_time: whether to return the time of the last found in the data

    @return: dictionary of event marker and its corresponding event locked data. The keys are the event markers
    """
    assert tmin < tmax, 'tmin must be less than tmax'
    event_marker, event_marker_time = event_marker
    event_marker = event_marker[event_channel]
    data, data_time = data
    events_of_interest = [e for e in events_of_interest if e in event_marker]
    rtn = {e: [] for e in events_of_interest}
    latest_event_start_time = -1
    epoch_length = int((tmax - tmin) * srate)
    reject_count = defaultdict(int)
    for e in events_of_interest:
        this_event_marker_time = event_marker_time[event_marker == e]
        data_event_starts = [np.argmin(abs(data_time - (s+tmin))) for s in this_event_marker_time]
        data_event_ends = [epoch_length + s for s in data_event_starts]
        for i, j, e_time in zip(data_event_starts, data_event_ends, this_event_marker_time):
            if j < len(data_time):  # if the epoch is not cut off by the end of the data
                if reject is not None:
                    if np.max(np.max(data[:, i:j], axis=0) - np.min(data[:, i:j], axis=0)) > reject:
                        reject_count[e] += 1
                        continue
                rtn[e].append(data[:, i:j])
                latest_event_start_time = max(latest_event_start_time, e_time)
    # convert to numpy arrays
    rtn = {k: np.array(v) for k, v in rtn.items() if len(v) > 0}
    if verbose:
        [print(f"Found {len(v)} events for event marker {k}{f', rejected {reject_count[k]}' if reject is not None else ''}") for k, v in rtn.items()]
    if return_last_event_time:
        return rtn, latest_event_start_time
    else:
        return rtn


def buffer_event_locked_data(event_locked_data: dict, buffer: dict):
    """
    @param event_locked_data: can be either single-modal or multi-modal:
        single-modal: dictionary of event marker and its corresponding event locked data. The keys are the event markers
        multi-modal
    @param buffer: dictionary of event marker and its corresponding buffer. The keys are the event markers
    @return: dictionary of event marker and its corresponding event locked data. The keys are the event markers
    """
    # check if is multi-modal
    rtn = copy.deepcopy(buffer)
    if isinstance(event_locked_data[list(event_locked_data.keys())[0]], dict):
        assert len(event_locked_data) > 0, 'data must be a non-empty dictionary'
        for event_name, modality_data in event_locked_data.items():
            if event_name in rtn and type(rtn[event_name]) is not dict:
                raise ValueError(f"modality_data must be a dictionary, got {type(modality_data)}. "
                                 f"Did you call buffer_event_locked_data with a single-modal event_locked_data?")
            for modality, v in modality_data.items():
                if event_name in rtn:
                    if modality in rtn[event_name]:
                        rtn[event_name][modality] = np.concatenate([rtn[event_name][modality], v], axis=0)
                    else:
                        rtn[event_name][modality] = v
                else:
                    rtn[event_name] = {modality: v}
        return rtn
    else:
        return _buffer_event_locked_data(event_locked_data, rtn)


def _buffer_event_locked_data(event_locked_data: dict, buffer: dict):
    rtn = copy.deepcopy(buffer)
    for k, v in event_locked_data.items():
        if k in buffer:
            v = np.concatenate([buffer[k], v], axis=0)
        else:
            v = np.array(v)
        rtn[k] = v
    return rtn


def get_baselined_event_locked_data(event_locked_data, baseline_t, srate, pick: int = None):
    """
    @param event_locked_data: dictionary of event marker and its corresponding event locked data. The keys are the event markers
    """
    rtn = {}
    pick = [pick] if isinstance(pick, int) else list(range(event_locked_data[list(event_locked_data.keys())[0]].shape[1])) if pick is None else pick
    for k, v in event_locked_data.items():
        d = v[:, pick]
        d = d - np.mean(d[:, :, :int(baseline_t * srate)], axis=2, keepdims=True)
        rtn[k] = d
    return rtn



def visualize_epochs(epochs, colors=None, picks: Union[List, Dict]=None, title='', out_dir=None, fig_size=(12.8, 7.2),
                     tmin=-0.1, tmax=0.8):
    """
    Plot the event locked epochs.

    @param epochs: dictionary of event marker and its corresponding event locked data. The keys are the event markers
        The values are (n_epochs, n_channels, n_times)
    @param colors:
    @param picks:
    @param title:
    @param verbose:
    @param fig_size:
    @param is_plot_timeseries:
    @param tmin:
    @param tmax:
    @return:
    """
    import matplotlib
    import matplotlib.pyplot as plt
    # Set the figure size for the plot
    plt.rcParams["figure.figsize"] = fig_size
    events = list(epochs.keys())

    if colors is None:
        # get distinct color for each unique y
        colors = {e: c for e, c in zip(events, matplotlib.cm.tab10(range(len(events))))}

    if picks is None:
        picks = [f'channel {i}' for i in range(epochs[events[0]].shape[1])]


    # Plot each EEG channel for each event type
    for ch_index, ch_name in enumerate(picks):
        for e in events:
            x = epochs[e]
            x_mean = np.mean(x[:, ch_index], axis=0)
            x1 = x_mean + scipy.stats.sem(x[:, ch_index], axis=0)  # this is the upper envelope
            x2 = x_mean - scipy.stats.sem(x[:, ch_index], axis=0)
            time_vector = np.linspace(tmin, tmax, x.shape[-1])
            # Plot the EEG data as a shaded area
            plt.fill_between(time_vector, x1, x2, where=x2 <= x1, facecolor=colors[e], interpolate=True, alpha=0.5)
            plt.plot(time_vector, x_mean, c=colors[e], label=f'{e}, N={x.shape[0]}')

            # Set the labels and title for the plot
        plt.xlabel('Time (sec)')
        plt.ylabel(f'{ch_name}, shades are SEM')
        plt.legend()
        title_text = f"{ch_name}{'-' + title if len(title) > 0 else ''}"
        plt.title(title_text)

        # Save or show the plot
        if out_dir:
            plt.savefig(os.path.join(out_dir, f'{title_text}.png'))
            plt.clf()
        else:
            plt.show()


def visualize_erd(epochs, tmin, tmax, srate, colors=None, picks: Union[List, Dict]=None, title='', out_dir=None, fig_size=(12.8, 7.2),
                     nperseg=128, noverlap_denom=2, nfft=512):
    """

    nfft: Number of FFT points
    """
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = fig_size
    events = list(epochs.keys())

    noverlap = nperseg // noverlap_denom  # Overlap between segments
    freq_min = 1
    freq_max = 30

    if picks is None:
        picks = [f'channel {i}' for i in range(epochs[events[0]].shape[1])]

    # Plot each EEG channel for each event type
    all_spectrograms = {}

    for ch_index, ch_name in enumerate(picks):
        for e in events:
            x = epochs[e]
            # x_mean = np.mean(x[:, ch_index], axis=0)
            # x1 = x_mean + scipy.stats.sem(x[:, ch_index], axis=0)  # this is the upper envelope
            # x2 = x_mean - scipy.stats.sem(x[:, ch_index], axis=0)
            # time_vector = np.linspace(tmin, tmax, x.shape[-1])
            # Plot the EEG data as a shaded area
            # plt.fill_between(time_vector, x1, x2, where=x2 <= x1, facecolor=colors[e], interpolate=True, alpha=0.5)
            # plt.plot(time_vector, x_mean, c=colors[e], label=f'{e}, N={x.shape[0]}')

            spectrograms = []
            frequencies = []
            for i in range(len(x)):  # iterate over the epochs
                f, t, Sxx = spectrogram(x[i, ch_index], fs=srate, nperseg=nperseg, noverlap=noverlap,nfft=nfft)
                spectrograms.append(Sxx)
                frequencies.append(f)
            mean_spectrograms = np.mean(np.array(spectrograms), axis=0)
            freqs = np.array(frequencies)[0]
            all_spectrograms[ch_name, e] = np.log10(mean_spectrograms)

    min_freq_index = np.argmin(np.abs(freqs - freq_min))
    max_freq_index = np.argmin(np.abs(freqs - freq_max))

    for ch_index, ch_name in enumerate(picks):
        all_mean_spectrograms = np.stack([v[1] for k, v in all_spectrograms.items() if ch_name in k])
        vmax = np.max(all_mean_spectrograms)
        vmin = np.min(all_mean_spectrograms)

        for e in events:
            mean_spectrograms = all_spectrograms[ch_name, e]
            im = plt.imshow(mean_spectrograms[min_freq_index:max_freq_index], cmap='jet', aspect='auto',
                            extent=[t.min(), t.max(), freqs[min_freq_index], freqs[max_freq_index]], vmin=vmin, vmax=vmax)
            plt.colorbar(im)
            title_text = f'{title}: channel {ch_name}, class {e}'
            plt.title(title_text)
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.show()

            # Save or show the plot
            if out_dir:
                plt.savefig(os.path.join(out_dir, f'{title_text}.png'))
                plt.clf()
            else:
                plt.show()




