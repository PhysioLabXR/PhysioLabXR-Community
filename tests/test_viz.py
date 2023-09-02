import numpy as np
from matplotlib import pyplot as plt

from physiolabxr.utils.data_utils import reject_outliers


def plot_replay_benchmark_results(results, test_axes, metrics, notes=''):
    """
    the key for results[measure] are the test axes, these keys must be in the same order as test axes
    @param results:
    @param test_axes:
    @param metrics:
    @return:
    """
    sampling_rates_to_test = test_axes["sampling rate (Hz)"]
    num_channels_to_test = test_axes["number of channels"]
    num_streams_to_test = test_axes["number of streams"]
    num_streams_to_test.remove(9)
    # if 'replay push data loop time' in metrics:
    #     for n_streams in num_streams_to_test:
    #         visualize_metrics_across_num_chan_sampling_rate(results, ['replay push data loop time'], n_streams, sampling_rates_to_test, num_channels_to_test, notes=notes)

    if 'timestamp reenactment accuracy' in metrics:
        timestamps_renactments_discrepencies = []
        for n_streams in num_streams_to_test:
            results_this_n_streams = np.empty(0)
            for i, num_channels in enumerate(num_channels_to_test):
                for j, sampling_rate in enumerate(sampling_rates_to_test):
                    results_this_n_streams = np.concatenate([results_this_n_streams, results['timestamp reenactment accuracy'][n_streams, num_channels, sampling_rate]['timestamp reenactment accuracy']])
            timestamps_renactments_discrepencies.append(results_this_n_streams)
        for i in range(len(timestamps_renactments_discrepencies)):
            timestamps_renactments_discrepencies[i] = [x for x in timestamps_renactments_discrepencies[i] if x < 3e-5]
        plt.boxplot(timestamps_renactments_discrepencies)
        plt.xlabel('number of streams')
        plt.ylabel('discrepancy between original and replayed stream timestamps (second)')
        plt.show()


def visualize_metrics_across_num_chan_sampling_rate(results, metrics, number_of_streams, sampling_rates_to_test, num_channels_to_test, notes=''):
    for measure in metrics:
        result_matrix = np.zeros((len(sampling_rates_to_test), len(num_channels_to_test), 2))  # last dimension is mean and std
        for i, num_channels in enumerate(num_channels_to_test):
            for j, sampling_rate in enumerate(sampling_rates_to_test):
                result_matrix[i, j] = results[measure][number_of_streams, num_channels, sampling_rate][measure]
        plt.imshow(result_matrix[:, :, 0], cmap='plasma')
        plt.xticks(ticks=list(range(len(sampling_rates_to_test))), labels=sampling_rates_to_test)
        plt.yticks(ticks=list(range(len(num_channels_to_test))), labels=num_channels_to_test)
        plt.xlabel("Sampling Rate (Hz)")
        plt.ylabel("Number of channels")
        plt.title(f'{number_of_streams} stream{"s" if number_of_streams > 1 else ""}: {measure}. {notes}')
        plt.colorbar()
        plt.show()
