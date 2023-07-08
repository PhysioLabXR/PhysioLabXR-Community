"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/rena
Otherwise, you will get either import error or file not found error
"""
import math
import pickle

import numpy as np
import pytest

from rena.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from rena.tests.test_utils import ContextBot, get_random_test_stream_names, run_visualization_benchmark, app_fixture, \
    run_replay_benchmark, plot_viz_benchmark_results
from rena.tests.test_viz import plot_replay_benchmark_results


@pytest.fixture
def app_main_window(qtbot):
    app, test_renalabapp_main_window = app_fixture(qtbot)
    yield test_renalabapp_main_window
    app.quit()

@pytest.fixture
def context_bot(app_main_window, qtbot):
    test_context = ContextBot(app=app_main_window, qtbot=qtbot)
    yield test_context
    test_context.clean_up()


def test_stream_visualization_dummy_streams_performance(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_time_second_per_stream = 60
    num_streams_to_test = [1, 5, 3, 7, 9]
    sampling_rates_to_test = np.linspace(1, 2048, 10)
    num_channels_to_test = np.linspace(1, 128, 10)
    metrics = 'update buffer time', 'plot data time', 'viz fps'

    num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
    sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

    test_axes = {"number of streams": num_streams_to_test, "number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}

    num_tests = len(num_streams_to_test) * len(sampling_rates_to_test) * len(num_channels_to_test)
    test_stream_names = get_random_test_stream_names(np.sum([n_stream * len(sampling_rates_to_test) * len(num_channels_to_test) for n_stream in num_streams_to_test]))

    print(f"Testing performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {2 * (num_tests * (test_time_second_per_stream + 3))} seconds.")

    test_context = ContextBot(app_main_window, qtbot)

    results_without_recording = run_visualization_benchmark(app_main_window, test_context, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False)
    pickle.dump({'results_without_recording': results_without_recording, 'test_axes': test_axes}, open("single_stream_benchmark.p", 'wb'))

    plot_viz_benchmark_results(results_without_recording, test_axes=test_axes, metrics=metrics, notes="")


def test_stream_visualization_real_streams_performance(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_time_second_per_stream = 60
    test_combos = [['EEG', 'EventMarker'],
                   ['EEG', 'EventMarker', 'Eyetracking'],
                   ['EEG', 'EventMarker', 'fMRI'],
                   ['EEG', 'EventMarker', 'Eyetracking', 'fMRI'],
                   ['EEG', 'EventMarker', 'Eyetracking', 'fMRI', 'CamCapture']]
    metrics = 'update buffer time', 'plot data time', 'viz fps'

    # test_axes = {"number of streams": num_streams_to_test, "number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}
    # num_tests = len(num_streams_to_test) * len(sampling_rates_to_test) * len(num_channels_to_test)
    # test_stream_names = get_random_test_stream_names(np.sum([n_stream * len(sampling_rates_to_test) * len(num_channels_to_test) for n_stream in num_streams_to_test]))
    #
    # print(f"Testing performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    # print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {2 * (num_tests * (test_time_second_per_stream + 3))} seconds.")
    #
    # test_context = ContextBot(app_main_window, qtbot)
    #
    # results_without_recording = run_visualization_benchmark(app_main_window, test_context, test_stream_names, num_streams_to_test, num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, is_reocrding=False)
    # pickle.dump({'results_without_recording': results_without_recording, 'test_axes': test_axes}, open("single_stream_benchmark.p", 'wb'))
    #
    # plot_viz_benchmark_results(results_without_recording, test_axes=test_axes, metrics=metrics, notes="")


def test_replay_data_throughput(app_main_window, qtbot) -> None:
    results_path = "replay_benchmark_pairwise.p"
    test_time_second_per_stream = 30
    num_streams_to_test = [9]
    sampling_rates_to_test = np.linspace(1024, 2048, 2)
    num_channels_to_test = np.linspace(100, 128, 2)

    # test_time_second_per_stream = 10
    # num_streams_to_test = [1]
    # sampling_rates_to_test = np.linspace(1, 2048, 1)
    # num_channels_to_test = np.linspace(1, 128, 1)
    metrics = 'replay push data loop time', 'timestamp reenactment accuracy'

    num_channels_to_test = [math.ceil(x) for x in num_channels_to_test]
    sampling_rates_to_test = [math.ceil(x) for x in sampling_rates_to_test]

    test_axes = {"number of streams": num_streams_to_test, "number of channels": num_channels_to_test, "sampling rate (Hz)": sampling_rates_to_test}

    num_tests = len(num_streams_to_test) * len(sampling_rates_to_test) * len(num_channels_to_test)
    test_stream_names = get_random_test_stream_names(np.sum([n_stream * len(sampling_rates_to_test) * len(num_channels_to_test) for n_stream in num_streams_to_test]))

    print(f"Testing replaying performance for a single stream, with sampling rates: {sampling_rates_to_test}\n, #channels {num_channels_to_test}. ")
    print(f"Test time per stream is {test_time_second_per_stream}, with {num_tests} tests. ETA {num_tests * (test_time_second_per_stream + 3)} seconds.")

    test_context = ContextBot(app_main_window, qtbot)

    results = run_replay_benchmark(app_main_window, test_context, test_stream_names, num_streams_to_test,num_channels_to_test, sampling_rates_to_test, test_time_second_per_stream, metrics, results_path)
    pickle.dump({'results': results, 'test_axes': test_axes}, open(results_path, 'wb'))

    plot_replay_benchmark_results(results, test_axes=test_axes, metrics=metrics, notes="")