import math
import pickle

from tests.test_utils import ContextBot, get_random_test_stream_names, run_visualization_benchmark, app_fixture, \
    run_replay_benchmark, plot_viz_benchmark_results

results = pickle.load(open(r'D:\PycharmProjects\RenaLabApp\benchmark_srate_nchan.p', 'rb'))
plot_viz_benchmark_results(results['results_without_recording'], test_axes=results['test_axes'], metrics=['update buffer time', 'plot data time', 'viz fps'], notes="")