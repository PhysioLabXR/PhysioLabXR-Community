"""
If you have get file not found error, make sure you set the working directory to .../RealityNavigation/physiolabxr
Otherwise, you will get either import error or file not found error
"""
import math
import pickle

import numpy as np
import pytest

from physiolabxr.configs.configs import AppConfigs
AppConfigs(_reset=True)  # create the singleton app configs object

from tests.test_utils import ContextBot, get_random_test_stream_names, run_visualization_benchmark, app_fixture, \
    run_replay_benchmark, plot_viz_benchmark_results, run_visualization_simulation_benchmark, \
    plot_viz_simulation_benchmark_results
from tests.test_viz import plot_replay_benchmark_results
from physiolabxr.presets.PlotConfig import ImageConfig, ImageFormat
from physiolabxr.presets.PresetEnums import PresetType, DataType
from physiolabxr.presets.GroupEntry import PlotFormat

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
    num_streams_to_test = [1, 3, 5, 7, 9]
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


def test_stream_visualization_simulation_streams_performance(app_main_window, qtbot) -> None:
    '''
    Adding active stream
    :param app:
    :param qtbot:
    :return:
    '''
    test_time_second_per_combo = 60  # 60
    test_combos = [['EEG', 'Trigger'],
                   ['EEG', 'Trigger', 'Eyetracking'],
                   ['EEG', 'Trigger', 'Eyetracking', 'CamCapture'],
                   ['EEG', 'Trigger', 'fMRI'],
                   ['EEG', 'Trigger', 'Eyetracking', 'fMRI'],
                   ['EEG', 'Trigger', 'Eyetracking', 'fMRI', 'CamCapture']
                   ]
    metrics = 'update buffer time', 'viz fps'

    test_stream_params = {'EEG':          {'srate': 2048, 'num_channels': 128, 'preset_type': PresetType.ZMQ, "data_type": DataType.float32},
                          'Trigger':      {'srate': 2048, 'num_channels': 1, 'preset_type': PresetType.ZMQ, "data_type": DataType.float32},
                          'Eyetracking':  {'srate': 1200, 'num_channels': 51, 'preset_type': PresetType.ZMQ, "data_type": DataType.float32},
                          'CamCapture':   {'srate': 30, 'num_channels': 1080 * 1920 * 3, 'plot_format': PlotFormat.IMAGE, 'data_type': DataType.uint8, 'preset_type': PresetType.ZMQ,
                                               'plot_configs': {'image_config':
                                                                  {"width": 1080,
                                                                   "height": 1920,
                                                                   "image_format": ImageFormat.rgb
                                                                   }
                                                              }
                                              },  # assuming a 1080 * 720 color video
                          'fMRI': {'srate': 1, 'num_channels': 64 * 64 * 42, 'preset_type': PresetType.ZMQ, "data_type": DataType.float32}}

    test_context = ContextBot(app_main_window, qtbot)

    results_without_recording = run_visualization_simulation_benchmark(app_main_window, test_context, test_combos, test_stream_params, test_time_second_per_combo, metrics, is_reocrding=False)
    pickle.dump(results_without_recording, open("benchmark_simulation.p", 'wb'))
    plot_viz_simulation_benchmark_results(results_without_recording, notes="")  # TODO add back viz benchmark results


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