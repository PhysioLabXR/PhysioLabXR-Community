import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet, local_clock
from random import random as rand


def generate_eeg_data(num_channels=8, noise_level=0.5, primary_eeg_frequency=12, t=0.0):
    """
    Simulates EEG (Electroencephalography) data by generating sinusoidal signals with specified properties.

    The function creates a set of sinusoidal signals, each representing the EEG data for a single channel.
    It then adds Gaussian noise to these signals to simulate real-world EEG data more accurately.

    Parameters:
    - num_channels (int): The number of EEG channels to simulate. Each channel will have its own sinusoidal signal. Default is 8.
    - noise_level (float): The standard deviation of the Gaussian noise added to the signals. This controls how noisy the simulated EEG data will be. Default is 0.5.
    - primary_frequency (float): The frequency of the sinusoidal wave in Hz, representing the primary frequency component of the EEG data. Default is 12Hz.
    - t (float or array-like): The time points at which the EEG signals are sampled. Can be a single number or an array of numbers. Default is 0.

    Returns:
    - eeg_signals_with_noise (ndarray): An array of shape (num_channels,) containing the simulated EEG data for each channel, with noise added.

    """

    # Generate a sinusoidal signal to simulate brainwave activity

    eeg_signals = np.array([np.sin(2 * np.pi * primary_eeg_frequency * t) for i in range(num_channels)])

    # Add random noise to the signal
    noise = noise_level * np.random.randn(num_channels)
    eeg_signals_with_noise = eeg_signals + noise

    return eeg_signals_with_noise


# Usage
num_channels = 100
duration = 1000  # seconds
num_samples = 256
sampling_rate = 256  # Hz
noise_level = 1
primary_frequency = 12  # Hz

# using the local_clock() to track elapsed time
start_time = local_clock()
# track how many samples we have sent
sent_samples = 0

# recursively send this data to LSL
timestamp = 0.0

stream_name = 'Dummy-EEG'
info = StreamInfo(stream_name, 'my_stream_type', num_channels, sampling_rate, 'float32',
                  'my_stream_id')
outlet = StreamOutlet(info)

while True:
    elapsed_time = local_clock() - start_time
    required_samples = int(sampling_rate * elapsed_time) - sent_samples

    for sample_ix in range(required_samples):
        eeg_frame = generate_eeg_data(num_channels, noise_level, primary_frequency, timestamp)
        outlet.push_sample(eeg_frame.flatten(), timestamp=timestamp)
        timestamp += 1 / sampling_rate
    sent_samples += required_samples
    # now send it and wait for a bit before trying again.
    time.sleep(0.01)
