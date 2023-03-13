import math
import pickle

import numpy as np
from matplotlib import pyplot as plt

from rena.tests.test_utils import update_test_cwd

metrics = 'update buffer time', 'plot data time'
sampling_rates_to_test = np.linspace(1, 2048, 10)
num_channels_to_test = np.linspace(1, 500, 10)
num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

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

for measure in metrics:
    result_matrix = np.zeros((len(sampling_rates_to_test), len(num_channels_to_test), 2))  # last dimension is mean and std
    for i, num_channels in enumerate(num_channels_to_test):
        for j, sampling_rate in enumerate(sampling_rates_to_test):
            result_matrix[i, j] = results[measure][test_space.index((num_channels, sampling_rate))]
    plt.imshow(result_matrix[:, :, 0])
    plt.xticks(ticks=list(range(len(sampling_rates_to_test))), labels=sampling_rates_to_test)
    plt.yticks(ticks=list(range(len(num_channels_to_test))), labels=num_channels_to_test)
    plt.xlabel("Sampling Rate (Hz)")
    plt.ylabel("Number of channels")
    plt.title(f'Rena Benchmark: single stream: {measure} (seconds)')
    plt.colorbar()
    plt.show()