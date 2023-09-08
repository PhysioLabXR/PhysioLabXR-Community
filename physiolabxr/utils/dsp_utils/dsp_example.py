# This is an example of how to use the dsp modules

import numpy as np
from physiolabxr.utils.dsp_utils.dsp_modules import ButterworthLowpassFilter
import matplotlib.pyplot as plt


def signal_generator(f, fs, duration, amp):
    '''
    Generate a sine wave signal
    :param f: frequency
    :param fs: sampling frequency
    :param duration: duration of the signal
    :param amp: amplitude of the signal
    '''
    wave = amp * (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
    return wave


channel_num = 1000

signal1 = signal_generator(f=50, fs=1000, duration=1, amp=1)
signal2 = signal_generator(f=100, fs=1000, duration=1, amp=1)
signal3 = signal1 + signal2
signal3 = np.transpose([signal3] * channel_num).T

input_signal = signal3

digital_filter = ButterworthLowpassFilter()

# set channel number
digital_filter.set_channel_num(channel_num)
# set data processor params
digital_filter.set_data_processor_params(fs=1000, cutoff=70, order=4)
# set data processor
digital_filter.evoke_data_processor()
# activate data processor
digital_filter.activate_data_processor()
# process data
output = digital_filter.process_buffer(input_signal)

input_channel_0 = input_signal[0, :]

output_channel_0 = output[0, :]



##################################################

plt.title('Input Signal')
# plot the first 3000 samples in the first channel
plt.plot(input_channel_0)
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.show()

##################################################

fft_result_input = np.fft.fft(input_channel_0)
fft_freqs_input = np.fft.fftfreq(len(input_channel_0))

# Shift the zero frequency component to the center of the spectrum
fft_result_input = np.fft.fftshift(fft_result_input)
fft_freqs_input = np.fft.fftshift(fft_freqs_input)

# Calculate the magnitude of the Fourier Transform
fft_magnitude_input = np.abs(fft_result_input)

# Plot the Fourier Transform with frequency on the x-axis
plt.plot(fft_freqs_input, fft_magnitude_input)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform of Input Time Series Data')
plt.grid(True)
plt.show()

##################################################

plt.title('Output Signal')
# plot the first 3000 samples in the first channel
plt.plot(output_channel_0)
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.show()

##################################################

fft_result_output = np.fft.fft(output_channel_0)
fft_freqs_output = np.fft.fftfreq(len(output_channel_0))

# Shift the zero frequency component to the center of the spectrum
fft_result_output = np.fft.fftshift(fft_result_output)
fft_freqs_output = np.fft.fftshift(fft_freqs_output)

# Calculate the magnitude of the Fourier Transform
fft_magnitude_output = np.abs(fft_result_output)

# Plot the Fourier Transform with frequency on the x-axis
plt.plot(fft_freqs_output, fft_magnitude_output)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform of Output Time Series Data')
plt.grid(True)
plt.show()

##################################################
