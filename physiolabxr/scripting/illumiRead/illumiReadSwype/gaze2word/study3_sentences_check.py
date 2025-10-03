"""
Checks for OOV words in study 3 sentences
"""
import glob
import os
import pickle

import pandas as pd
import re

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.gaze2word import Gaze2Word

reload_g2w = False

# Specify the folder containing your .xlsx files
folder_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/StudySentences/user_study3_session_sentences'
gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'

if reload_g2w:
    g2w = Gaze2Word(gaze_data_path)
    with open('g2w.pkl', 'wb') as f:
        pickle.dump(g2w, f)
else:
    # load it back
    with open('g2w.pkl', 'rb') as f:
        g2w = pickle.load(f)

# Use glob to find all .xlsx files in the folder
xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
xlsx_files = [f for f in xlsx_files if '~$' not in os.path.basename(f)]

all_sentences = []
all_words = []

for file_path in xlsx_files:
    # Read the .xlsx file into a pandas DataFrame
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise Exception(f"Failed to load {file_path}: {e}")
    # Assuming there's only one column in your spreadsheet containing sentences
    # Extract that column (the first column) and append sentences to all_sentences
    for sentence in df.iloc[:, 0]:
        # Make sure the cell is not empty/NaN
        if pd.notna(sentence):
            all_sentences.append(str(sentence))
            words_in_sentence = set(re.findall(r'\b\w+\b', sentence.lower()))
            words_in_sentence = [w for w in words_in_sentence if len(w) > 1]
            all_words.extend(words_in_sentence)
            for word in words_in_sentence:
                if word not in g2w.vocab_traces:
                    print(f"Out of vocab word: {word} In file {os.path.basename(file_path)} from sentence {sentence}")

# check trim vocab works
g2w.trim_vocab(all_words)