import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from scipy.interpolate import interp1d
from typing import Dict, Any, Union, Tuple


def bandpass_and_notch_filter(data: np.ndarray,
                              sampling_rate: float,
                              lowcut: float = 0.1,
                              highcut: float = 30,
                              notch_freq: float = 60.0) -> np.ndarray:
    """
    Apply a bandpass filter and then a notch filter to the EEG data.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).
    sampling_rate : float
        Sampling rate of the EEG data in Hz.
    lowcut : float
        Lower frequency cutoff for the bandpass filter (Hz).
    highcut : float
        Upper frequency cutoff for the bandpass filter (Hz).
    notch_freq : float
        Frequency for the notch filter (Hz).

    Returns
    -------
    np.ndarray
        Filtered EEG data with the same shape as input.
    """
    nyquist = sampling_rate / 2.0

    # Bandpass filter design
    b_band, a_band = signal.butter(N=4,
                                   Wn=[lowcut / nyquist, highcut / nyquist],
                                   btype='band')
    filtered_data = signal.filtfilt(b_band, a_band, data, axis=1)

    # Notch filter design
    b_notch, a_notch = signal.iirnotch(w0=notch_freq, Q=30, fs=sampling_rate)
    filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=1)

    return filtered_data


def detect_bad_channels(data: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """
    Detect bad channels based on their statistical outliers.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).
    z_threshold : float
        Z-score threshold for detecting bad channels.

    Returns
    -------
    np.ndarray
        Indices of bad channels.
    """
    channel_std = np.std(data, axis=1)
    channel_range = np.ptp(data, axis=1)

    std_z = zscore(channel_std)
    range_z = zscore(channel_range)

    bad_channels = np.where((np.abs(std_z) > z_threshold) | (np.abs(range_z) > z_threshold))[0]
    return bad_channels


def remove_artifacts(data: np.ndarray,
                     timestamps: np.ndarray,
                     artifact_threshold: float = 5.0) -> np.ndarray:
    """
    Remove artifacts by interpolating over samples that deviate strongly
    from the channel mean (in terms of standard deviations).

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).
    timestamps : np.ndarray
        Timestamps corresponding to each sample.
    artifact_threshold : float
        Threshold in standard deviations for detecting artifacts.

    Returns
    -------
    np.ndarray
        Artifact-corrected EEG data.
    """
    data_clean = data.copy()
    channel_mean = np.mean(data_clean, axis=1, keepdims=True)
    channel_std = np.std(data_clean, axis=1, keepdims=True)

    artifact_mask = np.abs(data_clean - channel_mean) > (artifact_threshold * channel_std)

    for chan in range(data_clean.shape[0]):
        bad_samples = artifact_mask[chan, :]
        good_samples = ~bad_samples

        if np.any(bad_samples) and np.any(good_samples):
            interp_func = interp1d(timestamps[good_samples],
                                   data_clean[chan, good_samples],
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value='extrapolate')
            data_clean[chan, bad_samples] = interp_func(timestamps[bad_samples])

    return data_clean


def calculate_metrics(data: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
    """
    Calculate metrics for the preprocessed EEG data.

    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (channels, samples).
    timestamps : np.ndarray
        Timestamps corresponding to each sample.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing SNR, RMS, signal power, noise power, invalid channel indices,
        and total time range.
    """
    signal_power = np.mean(np.square(data), axis=1)
    noise_power = np.var(np.diff(data, axis=1), axis=1) + 1e-10

    # Compute SNR
    valid_idx = (signal_power > 0) & (noise_power > 0)
    snr = np.zeros_like(signal_power)
    snr[valid_idx] = 10 * np.log10(signal_power[valid_idx] / noise_power[valid_idx])

    rms = np.sqrt(np.mean(np.square(data), axis=1))
    total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    return {
        'SNR': snr,
        'RMS': rms,
        'Signal_Power': signal_power,
        'Noise_Power': noise_power,
        'Invalid_Channels': np.where(~valid_idx)[0],
        'Time_Range': total_time
    }


def preprocess_eeg(eeg_data: Union[list, np.ndarray],
                   timestamps: Union[list, np.ndarray],
                   lowcut: float = 0.1,
                   highcut: float = 30.0,
                   notch_freq: float = 60.0,
                   z_threshold: float = 3.0,
                   artifact_threshold: float = 5.0) -> Dict[str, Any]:
    """
    Complete EEG preprocessing pipeline including filtering, bad channel detection,
    and artifact removal.

    Parameters
    ----------
    eeg_data : list or np.ndarray
        Raw EEG data with shape (channels, samples).
    timestamps : list or np.ndarray
        Timestamps for each sample.
    lowcut : float
        Lower frequency cutoff for bandpass filter (Hz).
    highcut : float
        Upper frequency cutoff for bandpass filter (Hz).
    notch_freq : float
        Frequency for notch filter (Hz).
    z_threshold : float
        Z-score threshold for bad channel detection.
    artifact_threshold : float
        Standard deviation threshold for artifact detection.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'processed_data': np.ndarray, the preprocessed EEG data.
        - 'timestamps': np.ndarray, the timestamps of the data.
        - 'sampling_rate': float, the computed sampling rate.
        - 'bad_channels': np.ndarray, indices of detected bad channels.
        - 'metrics': Dict[str, Any], various computed metrics.
    """
    # Convert inputs to numpy arrays
    eeg_array = np.asarray(eeg_data, dtype=float)
    timestamps_array = np.asarray(timestamps, dtype=float)

    if eeg_array.ndim != 2:
        raise ValueError("eeg_data must be a 2D array (channels × samples).")

    if len(timestamps_array) != eeg_array.shape[1]:
        raise ValueError("Number of timestamps must match the number of samples in eeg_data.")

    # Compute sampling rate
    duration = timestamps_array[-1] - timestamps_array[0]
    if duration <= 0:
        raise ValueError("Invalid timestamps: end time not greater than start time.")
    sampling_rate = eeg_array.shape[1] / duration

    # Filtering
    filtered_data = bandpass_and_notch_filter(eeg_array, sampling_rate, lowcut, highcut, notch_freq)

    # Bad channel detection (optional handling: you might want to remove or mark them)
    bad_channels = detect_bad_channels(filtered_data, z_threshold=z_threshold)

    # Artifact removal by interpolation
    clean_data = remove_artifacts(filtered_data, timestamps_array, artifact_threshold=artifact_threshold)

    # Compute metrics
    metrics = calculate_metrics(clean_data, timestamps_array)

    return {
        'processed_data': clean_data,
        'timestamps': timestamps_array,
        'sampling_rate': sampling_rate,
        'bad_channels': bad_channels,
        'metrics': metrics
    }


def save_processed_eeg(results: Dict[str, Any], filename: str) -> pd.DataFrame:
    """
    Save processed EEG data and timestamps to a CSV file.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary output from preprocess_eeg, containing processed data, timestamps, etc.
    filename : str
        Base filename to save the CSV (without extension).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the saved data.
    """
    data_dict = {'Timestamp': results['timestamps']}
    processed_data = results['processed_data']
    bad_channels = results['bad_channels']

    # Create columns for each channel
    for i in range(processed_data.shape[0]):
        channel_name = f'Channel_{i + 1}'
        if i in bad_channels:
            channel_name += '_interpolated'
        data_dict[channel_name] = processed_data[i, :]

    df = pd.DataFrame(data_dict)
    csv_filename = f"{filename}.csv"
    df.to_csv(csv_filename, index=False)
    return df
