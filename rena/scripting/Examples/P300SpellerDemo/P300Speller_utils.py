import numpy as np
import mne
from rena.scripting.Examples.P300SpellerDemo.P300Speller_params import *
import matplotlib.pyplot as plt
import scipy
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
import seaborn as sns
from sklearn import metrics


def train_logistic_regression(X, y, model):
    X = X.reshape(X.shape[0],-1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    confusion_matrix(y_test, y_pred)


def confusion_matrix(y_test, y_pred):
    cm = metrics.confusion_matrix(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()




def add_stim_channel_to_raw_array(raw_array, stim_data, stim_channel_name='STI'):
    # if len(stim_data.shape)==1:
    #     stim_data = stim_data.reshape(1,stim_data.shape[0])
    info = mne.create_info([stim_channel_name], raw_array.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info)
    raw_array.add_channels([stim_raw], force_update_info=True)


def separate_p300_speller_event_and_info_markers(markers, ts):
    markers = markers[0]
    events_info = []
    event_indices = [index for index, i in enumerate(markers) if i in [TARGET_MARKER, NONTARGET_MARKER]]

    event_ts = ts[event_indices]
    events = markers[event_indices]

    for event_index in event_indices:
        events_info.append((markers[event_index + 1], markers[event_index + 2]))

    events = np.reshape(events, (1, -1))
    return events, event_ts, events_info



def visualize_eeg_epochs(epochs, event_groups, colors, title='', out_dir=None, verbose='INFO', fig_size=(12.8, 7.2), is_plot_timeseries=True, is_plot_topo_map=True, gaze_behavior=None):
    mne.set_log_level(verbose=verbose)
    plt.rcParams["figure.figsize"] = fig_size

    if is_plot_timeseries:
        for ch in eeg_picks:
            for event_name, events in event_groups.items():
                try:
                    y = epochs.crop(tmin_eeg_viz, tmax_eeg_viz)[event_name].pick_channels([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(tmin_eeg_viz, tmax_eeg_viz, y.shape[-1])
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=colors[event_name], interpolate=True, alpha=0.5)
                plt.plot(time_vector, y_mean, c=colors[event_name], label='{0}, N={1}'.format(event_name, y.shape[0]))
            plt.xlabel('Time (sec)')
            plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
            plt.legend()

            # plot gaze behavior if any
            # if gaze_behavior:
            #     if type(gaze_behavior[0]) is Saccade:
            #         durations = [x.duration for x in gaze_behavior if x.epoched]
            #         plt.twinx()
            #         n, bins, patches = plt.hist(durations, bins=10)
            #         plt.ylim(top=max(n) / 0.2)
            #         plt.ylabel('Saccade duration histogram')

            plt.legend()
            plt.title('{0} - Channel {1}'.format(title, ch))
            if out_dir:
                plt.savefig(os.path.join(out_dir, '{0} - Channel {1}.png'.format(title, ch)))
                plt.clf()
            else:
                plt.show()

    # get the min and max for plotting the topomap

    # if is_plot_topo_map:
    #     evoked = epochs.average()
    #     vmax_EEG = np.max(evoked.get_data())
    #     vmin_EEG = np.min(evoked.get_data())
    #
    #     for event_name, events in event_groups.items():
    #         try:
    #             epochs[events].average().plot_topomap(times=np.linspace(tmin_eeg_viz, tmax_eeg_viz, 6), size=3., title='{0} {1}'.format(event_name, title), time_unit='s', scalings=dict(eeg=1.), vlim=(vmin_EEG, vmax_EEG))
    #         except KeyError:  # meaning this event does not exist in these epochs
    #             continue