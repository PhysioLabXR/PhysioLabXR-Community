import math
import pickle

import numpy as np

from tests.test_utils import update_test_cwd, plot_viz_benchmark_results

metrics = ['viz fps']
num_streams_to_test = [1, 3, 7]
sampling_rates_to_test = np.linspace(1, 2048, 10)
num_channels_to_test = np.linspace(1, 128, 10)

num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

update_test_cwd()
results = pickle.load(open("single_stream_benchmark.p", 'rb'))
# results_with_recording = results['results_with_recording']
results_without_recording = results['results_without_recording']
test_axes = results['test_axes']

# visualize_benchmark_results(results_with_recording, test_axes=test_axes, metrics=metrics, notes="With recording")
plot_viz_benchmark_results(results_without_recording, test_axes=test_axes, metrics=metrics, notes="")

