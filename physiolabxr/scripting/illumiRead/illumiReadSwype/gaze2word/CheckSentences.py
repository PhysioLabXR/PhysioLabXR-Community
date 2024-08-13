import pickle
import time
import warnings

import pandas as pd
from matplotlib import pyplot as plt
from nltk import RegexpTokenizer

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import load_trace_file


# file_to_check = '/Users/apocalyvec/Downloads/session1-short-longe-sentences.xlsx'
file_to_check = '/Users/apocalyvec/Downloads/session5-short-longe-sentences.xlsx'

# gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
# g2w = Gaze2Word(gaze_data_path)

# with open('g2w.pkl', 'wb') as f:
#     pickle.dump(g2w, f)

# load it back
with open('g2w.pkl', 'rb') as f:
    g2w = pickle.load(f)

df = pd.read_excel(file_to_check, sheet_name='Sheet1', header=None)

# go through every item in the df and fix if it starts or ends with a space
for i, row in df.iterrows():
    sent = row.iloc[0]
    if isinstance(sent, str) and sent.startswith(' '):
        df.at[i, 0] = sent.lstrip()
        print(f"Fixed starting space in sentence: {sent}")
    if isinstance(sent, str) and sent.endswith(' '):
        df.at[i, 0] = sent.rstrip()
        print(f"Fixed ending space in sentence: {sent}")

    sent = row.iloc[1]
    if isinstance(sent, str) and sent.startswith(' '):
        df.at[i, 1] = sent.lstrip()
        print(f"Fixed starting space in sentence: {sent}")
    if isinstance(sent, str) and sent.endswith(' '):
        df.at[i, 1] = sent.rstrip()
        print(f"Fixed ending space in sentence: {sent}")

# save the fixed df
df.to_excel(file_to_check, index=False, header=False)

sentences = df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()
sentences = [s for s in sentences if isinstance(s, str)]
tokenizer = RegexpTokenizer(r'\w+')

all_good = True
for s in sentences:
    words = tokenizer.tokenize(s)
    for w in words:
        if len(g2w.ngram_model.models[2][(w.lower(), )]) == 0:
            print(f"word {w} not in vocab. Sentence: {s}")
            all_good = False
            break

    # check if there's punctuation in the sentence
    if any(char in '.,?!' for char in s):
        print(f"punctuation in sentence: {s}")
        all_good = False

    # check if there's any more-than-one-space in the sentence
    if '  ' in s:
        print(f"more than one space in sentence: {s}")
        all_good = False

    # check if there's a space at the start or end of the sentence
    if s.startswith(' ') or s.endswith(' '):
        print(f"space at start or end of sentence: {s}")
        all_good = False

print(f"all good: {all_good}")