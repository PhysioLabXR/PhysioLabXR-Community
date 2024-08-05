import pickle
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import load_trace_file
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.gaze2word import Gaze2Word

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
    plt.figure(figsize=(10, 10))
    plt.scatter(trace[:, 0], trace[:, 1])
    plt.plot(trace[:, 0], trace[:, 1], label='User Trace')

    # plot the ideal trace for the word 'good'
    ideal_trace = g2w.vocab_traces['good']

    plt.scatter(ideal_trace[:, 0], ideal_trace[:, 1], color='orange')
    plt.plot(ideal_trace[:, 0], ideal_trace[:, 1], color='orange', label='Ideal Trace')

    # get the prediction
    # print(f"Predicting for trace {i}")
    # start_time = time.perf_counter()
    # top_k = g2w.predict(4, trace)
    # print(f'Time taken for prediction: {(pred_time := time.perf_counter() - start_time):.2f}s')

    # run dbscan on the user trace before predicting
    plt.scatter(ideal_trace[:, 0], ideal_trace[:, 1], color='red')
    plt.plot(ideal_trace[:, 0], ideal_trace[:, 1], color='red', label='DBSCAN Trace')

    start_time = time.perf_counter()
    top_k = g2w.predict(4, trace, run_dbscan=True, prefix='', verbose=True, filter_by_starting_letter=0.2, njobs=16)
    print(f'Time taken for prediction : {(pred_time := time.perf_counter() - start_time):.2f}s')

    plt.legend()
    plt.title(f'Trace {i}, predicts: {top_k}, \n npoints: {len(trace)}, pred time: {pred_time:.2f}s')
    plt.show()
