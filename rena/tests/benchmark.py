import pickle

import numpy as np
from matplotlib import pyplot as plt

from rena.tests.test_utils import update_test_cwd

metrics = 'update buffer time', 'plot data time'
sampling_rates_to_test = np.linspace(1, 2048, 10)
num_channels_to_test = np.linspace(1, 500, 10)

update_test_cwd()
a = pickle.load(open("single_stream_benchmark_results.p", 'rb'))

results = a['results']
test_space = a['test_space']

for measure in metrics:
    num_channels_in_test_space = np.array([num_channels for num_channels, _ in test_space])
    means = np.zeros(len(num_channels_to_test))
    for i, num_chan in enumerate(num_channels_to_test):
        means[i] = np.mean(np.array(results[measure])[num_channels_in_test_space == num_chan][:, 0])
    plt.scatter(num_channels_to_test, means)
    plt.plot(num_channels_to_test, means)
    plt.title(f"Rena Benchmark: single stream {measure} across number of channels")
    plt.xlabel("Number of channels")
    plt.ylabel(f'{measure} (seconds)')
    plt.show()

    srates_in_test_space = np.array([srate for _, srate in test_space])
    means = np.zeros(len(sampling_rates_to_test))
    for i, srate in enumerate(sampling_rates_to_test):
        means[i] = np.mean(np.array(results[measure])[srates_in_test_space == srate][:, 0])
    plt.scatter(sampling_rates_to_test, means)
    plt.plot(sampling_rates_to_test, means)
    plt.title(f"Rena Benchmark: single stream {measure} across sampling rates")
    plt.xlabel("Sampling Rate (Hz)")
    plt.ylabel(f'{measure} (seconds)')
    plt.show()
