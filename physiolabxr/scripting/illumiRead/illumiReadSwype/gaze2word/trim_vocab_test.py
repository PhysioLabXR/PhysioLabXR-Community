import pickle
import time

import pandas as pd
from matplotlib import pyplot as plt
from nltk import RegexpTokenizer

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import load_trace_file

trace_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\TraceSamples\good_FixationTrace.csv'

trace_list = load_trace_file(trace_path)

# gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
# g2w = Gaze2Word(gaze_data_path)

# with open('g2w.pkl', 'wb') as f:
#     pickle.dump(g2w, f)

# load it back
with open('g2w.pkl', 'rb') as f:
    g2w = pickle.load(f)


# load the sentences from the xlsx file
file_path = 'Short_Long_Sentences.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Extracting sentences from the columns
sentences = df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()
sentences = [s for s in sentences if isinstance(s, str)]
tokenizer = RegexpTokenizer(r'\w+')
words = [tokenizer.tokenize(s) for s in sentences]
words = [word for sublist in words for word in sublist]  # flatten the list

g2w.trim_vocab(words)

for i, trace in enumerate(trace_list):

    start_time = time.perf_counter()
    top_k = g2w.predict(4, trace, run_dbscan=True, prefix='', verbose=True, filter_by_starting_letter=0.45, use_trimmed_vocab=True, njobs=16)
    print(f'Time spent predicting with vocab trimmed: {(pred_time := time.perf_counter() - start_time):.8f}s: top_k: {top_k}')

    start_time = time.perf_counter()
    top_k = g2w.predict(4, trace, run_dbscan=True, prefix='', verbose=True, filter_by_starting_letter=0.45, njobs=16)
    print(f'Time spent predicting w/o vocab trimmed: {(pred_time := time.perf_counter() - start_time):.8f}s: top_k: {top_k}')