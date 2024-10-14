import pickle
import time

import numpy as np
from matplotlib import pyplot as plt

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import load_trace_file, run_dbscan_on_gaze

trace_path = '/Users/apocalyvec/Downloads/good_FixationTrace.csv'

trace_list = load_trace_file(trace_path)

# gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
# g2w = Gaze2Word(gaze_data_path)

# with open('g2w.pkl', 'wb') as f:
#     pickle.dump(g2w, f)

# load it back
with open('g2w.pkl', 'rb') as f:
    g2w = pickle.load(f)

# plot the gaze traces in a 2d scatter plot with lines connecting the points
for i, trace in enumerate(trace_list):
    # create dummy timestamps for the trace, 0-1 second
    dummy_timestamps = np.linspace(0, 1, len(trace))

    plt.figure(figsize=(10, 10))
    # plot the user's gaze trace
    plt.scatter(trace[:, 0], trace[:, 1])
    plt.plot(trace[:, 0], trace[:, 1], label='User Trace')

    # plot the ideal trace for the word 'good'
    ideal_trace = g2w.vocab_traces['good']
    plt.scatter(ideal_trace[:, 0], ideal_trace[:, 1], color='orange')
    plt.plot(ideal_trace[:, 0], ideal_trace[:, 1], color='orange', label='Ideal Trace')

    # run dbscan on the user trace before predicting
    dbscan_trace = run_dbscan_on_gaze(trace, dummy_timestamps, 0.1, 3, True)
    plt.scatter(dbscan_trace[:, 0], dbscan_trace[:, 1], color='red')
    plt.plot(dbscan_trace[:, 0], dbscan_trace[:, 1], color='red', label='DBSCAN Trace')

    # get the prediction
    print(f"Predicting for trace {i}")
    start_time = time.perf_counter()
    top_k = g2w.predict(4, trace, timestamps=dummy_timestamps, run_dbscan=True, prefix='', verbose=True, filter_by_starting_letter=0.45, njobs=16)
    print(f'Time taken for prediction: {(pred_time := time.perf_counter() - start_time):.8f}s')

    plt.legend()
    plt.title(f'Trace {i}, predicts: {top_k}, \n npoints: {len(trace)}, pred time: {pred_time:.2f}s')
    plt.show()
