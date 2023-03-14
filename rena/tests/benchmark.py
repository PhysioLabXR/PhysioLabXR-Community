import math
import pickle

import numpy as np
from matplotlib import pyplot as plt

from rena.tests.test_utils import update_test_cwd, visualize_benchmark_results

metrics = 'update buffer time', 'plot data time'
sampling_rates_to_test = np.linspace(1, 2048, 10)
num_channels_to_test = np.linspace(1, 500, 10)
num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

update_test_cwd()
results = pickle.load(open("single_stream_benchmark.p", 'rb'))
results_with_recording = results['results_with_recording']
results_without_recording = results['results_without_recording']
test_axes = results['test_axes']

visualize_benchmark_results(results_with_recording, test_axes=test_axes, metrics=metrics, notes="With recording")
visualize_benchmark_results(results_with_recording, test_axes=test_axes, metrics=metrics, notes="Without recording")

