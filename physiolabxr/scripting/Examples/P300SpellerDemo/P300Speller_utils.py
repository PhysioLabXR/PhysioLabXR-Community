import pickle
from datetime import datetime

import numpy as np
import mne
from imblearn.over_sampling import SMOTE
from physiolabxr.scripting.Examples.P300SpellerDemo.P300Speller_params import *
import matplotlib.pyplot as plt
import scipy
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
import seaborn as sns
from sklearn import metrics




def p300_speller_process_raw_data(raw, l_freq, h_freq, notch_f, picks):
    """
    Filters the raw EEG data for P300 Speller.

    Args:
        raw (mne.io.Raw): Raw EEG data.
        l_freq (float): Low cut frequency for the bandpass filter.
        h_freq (float): High cut frequency for the bandpass filter.
        notch_f (float): Frequency from the power source to remove using a notch filter. Typically 60Hz.
        picks (list of str or str): List of channel names to filter.

    Returns:
        mne.io.Raw: Filtered raw EEG data.

    """
    # Apply bandpass filter to remove unwanted frequency ranges
    raw_processed = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, picks=picks)

    # Apply notch filter to remove power line noise at 60 Hz
    raw_processed = raw_processed.copy().notch_filter(freqs=notch_f, picks=picks)

    # Resample data if needed
    # raw_processed = raw_processed.copy().resample(sfreq=resampling_rate)

    return raw_processed


def train_logistic_regression(X, y, model, test_size=0.2):
    """
    Trains a logistic regression model on the input data and prints the confusion matrix.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target variable.
        model (LogisticRegression): An instance of LogisticRegression from scikit-learn.
        test_size (float): Proportion of the data to reserve for testing. Default is 0.2.

    Returns:
        None.

    Raises:
        TypeError: If model is not an instance of LogisticRegression.
        ValueError: If test_size is not between 0 and 1.

    """
    # Check if model is an instance of LogisticRegression
    if not isinstance(model, LogisticRegression):
        raise TypeError("model must be an instance of LogisticRegression.")

    # Check if test_size is between 0 and 1
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)

    # Fit the model to the training data and make predictions on the test data
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Print the confusion matrix
    confusion_matrix(y_test, y_pred)


def confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots a confusion matrix for the predicted vs. actual labels and prints the accuracy score.

    Args:
        y_test (np.ndarray): Actual labels of the test set.
        y_pred (np.ndarray): Predicted labels of the test set.

    Returns:
        None.

    Raises:
        TypeError: If either y_test or y_pred are not numpy arrays.

    """
    # Check if y_test and y_pred are numpy arrays
    if not isinstance(y_test, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_test and y_pred must be numpy arrays.")

    # Calculate the confusion matrix and f1 score
    cm = metrics.confusion_matrix(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='macro')

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()


def generate_mne_stim_channel(data_ts, event_ts, events, deviate=25e-2):
    """
    Generates an MNE stimulus channel from event markers and timestamps.

    Args:
        data_ts (np.ndarray): Timestamps for the data stream.
        event_ts (np.ndarray): Timestamps for event markers.
        events (np.ndarray): Event markers.
        deviate (float): Maximum acceptable jitter interval.

    Returns:
        array: MNE stimulus channel data.

    """
    stim_array = np.zeros((1, data_ts.shape[0]))
    events = np.reshape(events, (1, -1))
    event_data_indices = [np.argmin(np.abs(data_ts - t)) for t in event_ts if
                          np.min(np.abs(data_ts - t)) < deviate]

    for index, event_data_index in enumerate(event_data_indices):
        stim_array[0, event_data_index] = events[0, index]

    return stim_array


# def add_stim_channel_to_raw_array(raw_array, stim_data, stim_channel_name='STI'):
#     """
#
#     @param raw_array: MNE raw data structure
#     @param stim_data: stim stream
#     @param stim_channel_name: stim channel name
#     """
#     info = mne.create_info([stim_channel_name], raw_array.info['sfreq'], ['stim'])
#     stim_raw = mne.io.RawArray(stim_data, info)
#     raw_array.add_channels([stim_raw], force_update_info=True)


def add_stim_channel(raw_array, data_ts, event_ts, events, stim_channel_name='STI', deviate=25e-2):
    """
    Add a stimulation channel to the MNE raw data object.

    Args:
        raw_array (mne.io.RawArray): MNE raw data object.
        data_ts (numpy.ndarray): Timestamps for the data stream.
        event_ts (numpy.ndarray): Timestamps for event markers.
        events (numpy.ndarray): Event markers.
        stim_channel_name (str): Name of the stimulation channel. Default is 'STI'.
        deviate (float): Maximum acceptable jitter interval. Default is 0.25.

    Returns:
        None
    """
    stim_array = generate_mne_stim_channel(data_ts, event_ts, events, deviate=deviate)
    info = mne.create_info([stim_channel_name], raw_array.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_array, info)
    raw_array.add_channels([stim_raw], force_update_info=True)


def rebalance_classes(x, y, by_channel=False):
    """
    Resamples the data to balance the classes using SMOTE algorithm.

    Parameters:
        x (np.ndarray): Input data array of shape (epochs, channels, samples).
        y (np.ndarray): Target labels array of shape (epochs,).
        by_channel (bool): If True, balance the classes separately for each channel. Otherwise,
            balance the classes for the whole input data.

    Returns:
        tuple: A tuple containing the resampled input data and target labels as numpy arrays.
    """
    epoch_shape = x.shape[1:]

    if by_channel:
        y_resample = None
        channel_data = []
        channel_num = epoch_shape[0]

        # Loop through each channel and balance the classes separately
        for channel_index in range(0, channel_num):
            sm = SMOTE(random_state=42)
            x_channel = x[:, channel_index, :]
            x_channel, y_resample = sm.fit_resample(x_channel, y)
            channel_data.append(x_channel)

        # Expand dimensions for each channel array and concatenate along the channel axis
        channel_data = [np.expand_dims(x, axis=1) for x in channel_data]
        x = np.concatenate([x for x in channel_data], axis=1)
        y = y_resample

    else:
        # Reshape the input data to 2D array and balance the classes
        x = np.reshape(x, newshape=(len(x), -1))
        sm = SMOTE(random_state=42)
        x, y = sm.fit_resample(x, y)

        # Reshape the input data back to its original shape
        x = np.reshape(x, newshape=(len(x),) + epoch_shape)

    return x, y


# def separate_p300_speller_event_and_info_markers(markers, ts):
#     markers = markers[0]
#     events_info = []
#     event_indices = [index for index, i in enumerate(markers) if i in [TARGET_MARKER, NONTARGET_MARKER]]
#
#     event_ts = ts[event_indices]
#     events = markers[event_indices]
#
#     for event_index in event_indices:
#         events_info.append((markers[event_index + 1], markers[event_index + 2]))
#
#     events = np.reshape(events, (1, -1))
#     return events, event_ts, events_info


def visualize_eeg_epochs(epochs, event_groups, colors, eeg_picks, title='', out_dir=None, verbose='INFO', fig_size=(12.8, 7.2),
                         is_plot_timeseries=True):
    """
    Visualize EEG epochs for different event types and channels.

    Args:
        epochs (mne.Epochs): The EEG epochs to visualize.
        event_groups (dict): A dictionary mapping event names to lists of event IDs. Only events in these groups will be plotted.
        colors (dict): A dictionary mapping event names to colors to use for plotting.
        eeg_picks (list): A list of EEG channels to plot.
        title (str, optional): The title to use for the plot. Default is an empty string.
        out_dir (str, optional): The directory to save the plot to. If None, the plot will be displayed on screen. Default is None.
        verbose (str, optional): The verbosity level for MNE. Default is 'INFO'.
        fig_size (tuple, optional): The size of the figure in inches. Default is (12.8, 7.2).
        is_plot_timeseries (bool, optional): Whether to plot the EEG data as a timeseries. Default is True.

    Returns:
        None

    Raises:
        None

    """

    # Set the verbosity level for MNE
    mne.set_log_level(verbose=verbose)

    # Set the figure size for the plot
    plt.rcParams["figure.figsize"] = fig_size

    # Plot each EEG channel for each event type
    if is_plot_timeseries:
        for ch in eeg_picks:
            for event_name, events in event_groups.items():
                try:
                    # Get the EEG data for the specified event type and channel
                    y = epochs.crop(tmin_eeg_viz, tmax_eeg_viz)[event_name].pick_channels([ch]).get_data().squeeze(1)
                except KeyError:  # meaning this event does not exist in these epochs
                    continue
                y_mean = np.mean(y, axis=0)
                y1 = y_mean + scipy.stats.sem(y, axis=0)  # this is the upper envelope
                y2 = y_mean - scipy.stats.sem(y, axis=0)

                time_vector = np.linspace(tmin_eeg_viz, tmax_eeg_viz, y.shape[-1])

                # Plot the EEG data as a shaded area
                plt.fill_between(time_vector, y1, y2, where=y2 <= y1, facecolor=colors[event_name], interpolate=True,
                                 alpha=0.5)
                plt.plot(time_vector, y_mean, c=colors[event_name], label='{0}, N={1}'.format(event_name, y.shape[0]))

            # Set the labels and title for the plot
            plt.xlabel('Time (sec)')
            plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
            plt.legend()
            plt.title('{0} - Channel {1}'.format(title, ch))

            # Save or show the plot
            if out_dir:
                plt.savefig(os.path.join(out_dir, '{0} - Channel {1}.png'.format(title, ch)))
                plt.clf()
            else:
                plt.show()

def save_data(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
