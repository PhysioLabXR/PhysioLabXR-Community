# ----------------------------------------------------------------------
# BENCHMARK: DTW vs. GlanceWriter on synthetic 200 Hz gaze traces
# ----------------------------------------------------------------------
import os
import pickle

import numpy as np, time, random

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.gaze2word import *

# 1) Choose any test word that exists in the vocab
TEST_WORD   = "hello"                 # change freely
N_REPS      = 20                      # repeat to smooth timing jitter
SIGMA_NOISE = 0.05                    # ≈5 % of key-width Gaussian noise
FPS         = 200                     # eye-tracker sample rate (Hz)

# 2) Utility to build a noisy 200 Hz trace for any word --------------
def make_trace(word: str, letter_locs: dict[str, np.ndarray],
               fps: int = 200, sigma: float = 0.05) -> np.ndarray:
    """
    • Uses the *ideal* straight-line segment between consecutive letters.
    • Puts fps/10 samples on each segment (empirically smooth enough).
    • Adds i.i.d. Gaussian noise N(0, sigma² × keyWidth²) to each point.
    """
    pts = []
    for a, b in zip(word, word[1:]):
        seg = np.linspace(letter_locs[a], letter_locs[b], num=fps//10,
                          endpoint=False)
        pts.append(seg)
    pts.append(letter_locs[word[-1]][None, :])          # final hold
    path = np.vstack(pts)

    # assume key-width ≈ avg x-distance between QWERTY neighbours
    key_w = np.mean(np.diff(sorted({p[0] for p in letter_locs.values()})))
    noise = np.random.normal(0, sigma*key_w, size=path.shape)
    return path + noise

if __name__ == "__main__":
    gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
    # gaze_data_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\GazeData.csv'

    if os.path.exists('g2w.pkl'):
        with open('g2w.pkl', 'rb') as f:
            g2w = pickle.load(f)
    else:
        g2w = Gaze2Word(gaze_data_path)
        with open('g2w.pkl', 'wb') as f:
            pickle.dump(g2w, f)

    # 3) Pre-build one trace per repetition so both decoders see identical input
    traces = [make_trace(TEST_WORD, g2w.letter_locations,
                         fps=FPS, sigma=SIGMA_NOISE)
              for _ in range(N_REPS)]

    # 4) TIME THE DTW DECODER -------------------------------------------
    t0 = time.perf_counter()
    for i, trace in enumerate(traces):
        print(f"Running dtw predict for trace {i+1}/{N_REPS}...")
        g2w.predict(k=5, gaze_trace=trace, run_dbscan=False,
                    njobs=1, return_prob=False)
    dtw_time = (time.perf_counter() - t0) / N_REPS

    # 5) TIME THE GLANCEWRITER DECODER -----------------------------------
    t0 = time.perf_counter()
    for i, trace in enumerate(traces):
        print(f"Running predict_glancewriter for trace {i+1}/{N_REPS}...")
        g2w.predict_glancewriter(5, trace)
    gw_time = (time.perf_counter() - t0) / N_REPS

    # 6) REPORT ----------------------------------------------------------
    print(f"RESULTS over {N_REPS} repetitions, word='{TEST_WORD}', "
          f"trace ≈{FPS} Hz, σ_noise={SIGMA_NOISE:.2f}")
    print(f"• predict (DTW)              : {dtw_time*1e3:8.2f} ms avg")
    print(f"• predict_glancewriter      : {gw_time*1e3:8.2f} ms avg "
          f"({dtw_time/gw_time:4.1f}× faster)")