import warnings

from joblib import Parallel, delayed
from collections import OrderedDict
import pickle
from enum import Enum
from typing import Union, List, Optional

import pandas as pd
import numpy as np
import re

from dtw import dtw
from sklearn.cluster import DBSCAN

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import run_dbscan_on_gaze

class Gaze2Hotkey:
    def __init__(self, gaze_data_path):
        self.hotkey_traces = []
        self.hotkeys = ["copy", "paste"]

    def predict(self, gaze_trace: np.ndarray, timestamps=None,
                run_dbscan: bool = False, dbscan_eps: float = 0.5, dbscan_min_samples: int = 5,
                njobs: int = 1,
                return_prob: float = False,
                verbose=False) -> list:

        if run_dbscan:
            gaze_trace = run_dbscan_on_gaze(gaze_trace, timestamps, dbscan_eps, dbscan_min_samples, verbose)

        if len(gaze_trace) < 3:
            return []

        template_traces = self.hotkey_traces
        hotkeys = self.hotkeys
        if njobs == 1:
            distances = [dtw(gaze_trace, template_trace, keep_internals=True, dist_method='euclidean').distance for
                         template_trace in template_traces.items()]
        else:
            distances = Parallel(n_jobs=njobs)(delayed(dtw)(gaze_trace, template_trace, keep_internals=True, dist_method='euclidean') for
                         template_trace in template_traces.items())
            distances = [result.distance for result in distances]

        if return_prob:  # turn distance into probs
            distances = np.array(distances)
            distance_probs = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            return [(h, d) for h, d in zip(hotkeys, distance_probs)]
        else:
            hotkey = [(w, d) for w, d in sorted(zip(hotkeys, distances), key=lambda x: x[1])[0]]
            return hotkey


