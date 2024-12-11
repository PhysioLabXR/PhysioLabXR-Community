import numpy as np
from scipy import signal
from scipy.stats import zscore
from scipy.interpolate import interp1d
import pandas as pd


def preprocess_eeg(eeg_data, timestamps, lowcut=0.1, highcut=30, notch_freq=60, z_threshold=3, artifact_threshold=5):
    """
    Complete EEG preprocessing pipeline including filtering, bad channel detection,
    and artifact removal.

    Parameters:
    -----------
    eeg_data : list or numpy.ndarray
        Raw EEG data (channels × samples)
    timestamps : list or numpy.ndarray
        Timestamps for each sample
    lowcut : float
        Lower frequency cutoff for bandpass filter (Hz)
    highcut : float
        Upper frequency cutoff for bandpass filter (Hz)
    notch_freq : float
        Frequency for notch filter (Hz)
    z_threshold : float
        Z-score threshold for bad channel detection
    artifact_threshold : float
        Standard deviation threshold for artifact detection

    Returns:
    --------
    dict
        Processed EEG data and associated metrics
    """
    # Convert inputs to numpy arrays
    eeg_array = np.array(eeg_data, dtype=float)
    timestamps_array = np.array(timestamps, dtype=float)

    # Calculate sampling rate
    sampling_rate = len(timestamps_array) / (timestamps_array[-1] - timestamps_array[0])

    def apply_filters(data):
        nyquist = sampling_rate / 2
        # Bandpass filter
        b, a = signal.butter(4, [lowcut / nyquist, highcut / nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=1)
        # Notch filter
        notch_b, notch_a = signal.iirnotch(notch_freq, Q=30, fs=sampling_rate)
        return signal.filtfilt(notch_b, notch_a, filtered, axis=1)

    def detect_bad_channels(data):
        channel_std = np.std(data, axis=1)
        channel_range = np.ptp(data, axis=1)
        std_z = zscore(channel_std)
        range_z = zscore(channel_range)
        return np.where((np.abs(std_z) > z_threshold) |
                        (np.abs(range_z) > z_threshold))[0]

    def remove_artifacts(data):
        data_clean = data.copy()
        chan_std = np.std(data_clean, axis=1, keepdims=True)
        chan_mean = np.mean(data_clean, axis=1, keepdims=True)
        artifact_mask = np.abs(data_clean - chan_mean) > (artifact_threshold * chan_std)

        for chan in range(data_clean.shape[0]):
            bad_samples = artifact_mask[chan, :]
            good_samples = ~bad_samples
            if np.any(bad_samples):
                interp_func = interp1d(
                    timestamps_array[good_samples],
                    data_clean[chan, good_samples],
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                data_clean[chan, bad_samples] = interp_func(timestamps_array[bad_samples])
        return data_clean

    def calculate_metrics(data):
        signal_power = np.mean(np.square(data), axis=1)
        noise_power = np.var(np.diff(data, axis=1), axis=1) + 1e-10
        snr = np.zeros_like(signal_power)
        valid_idx = (signal_power > 0) & (noise_power > 0)
        snr[valid_idx] = 10 * np.log10(signal_power[valid_idx] / noise_power[valid_idx])
        rms = np.sqrt(np.mean(np.square(data), axis=1))

        return {
            'SNR': snr,
            'RMS': rms,
            'Signal_Power': signal_power,
            'Noise_Power': noise_power,
            'Invalid_Channels': np.where(~valid_idx)[0],
            'Time_Range': timestamps_array[-1] - timestamps_array[0]
        }

    # Execute pipeline
    filtered_data = apply_filters(eeg_array)
    bad_channels = detect_bad_channels(filtered_data)
    clean_data = remove_artifacts(filtered_data)
    metrics = calculate_metrics(clean_data)

    return {
        'processed_data': clean_data,
        'timestamps': timestamps_array,
        'sampling_rate': sampling_rate,
        'bad_channels': bad_channels,
        'metrics': metrics
    }


def save_processed_eeg(results, filename):
    """
    Save processed EEG data to CSV with metadata
    """
    data_dict = {'Timestamp': results['timestamps']}
    for i in range(results['processed_data'].shape[0]):
        channel_name = f'Channel_{i + 1}'
        if i in results['bad_channels']:
            channel_name += '_interpolated'
        data_dict[channel_name] = results['processed_data'][i, :]

    df = pd.DataFrame(data_dict)
    csv_filename = f"{filename}.csv"
    df.to_csv(csv_filename, index=False)
    return df