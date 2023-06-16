import math
import pickle

import numpy as np

from rena.tests.test_utils import update_test_cwd, plot_replay_benchmark_results

metrics = 'replay push data loop time', 'timestamp reenactment accuracy'
# test_time_second_per_stream = 60
# num_streams_to_test = [1, 3, 5, 7, 9]
# sampling_rates_to_test = np.linspace(1, 2048, 10)
# num_channels_to_test = np.linspace(1, 128, 10)

test_time_second_per_stream = 10
num_streams_to_test = [1, 3]
sampling_rates_to_test = np.linspace(1, 2048, 2)
num_channels_to_test = np.linspace(1, 128, 2)

num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

update_test_cwd()
results = pickle.load(open("replay_benchmark.p", 'rb'))
test_axes = results['test_axes']

plot_replay_benchmark_results(results, test_axes=test_axes, metrics=metrics, notes="")