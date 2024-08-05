""" This file implements Gaze2Word class that predicts the next word based on the gaze trace from Sweyepe and previous text.

Get started:

    # change the gaze data path: gaze_data_path

    g2w = Gaze2Word(gaze_data_path)  # create an instance of the Gaze2Word class
    perfect_gaze_trace = g2w.vocab_traces['hello']  # create a perfect gaze trace for the word 'hello'

    # predict the top k most likely next word based on the gaze trace from Sweyepe and previous text
    pred_wo_prefix = g2w.predict(k, gaze_trace, prefix=None, ngram_alpha=0.2, ngram_epsilon=1e-8)

    # predict the top k most likely next word based on the gaze trace from Sweyepe and previous text
    pred_w_prefix = g2w.predict(k, gaze_trace, prefix='have a nice', ngram_alpha=0.2, ngram_epsilon=1e-8)

You can find more examples of the usage of the Gaze2Word class in the __main__ block at the end of the script.

For more information regarding the parameters, read the docs of the Gaze2Word class.

Known issues:
* the inference can be slow when the gaze sequence is long. Mostly from the quadratic time complexity of the DTW algorithm.

"""

from collections import OrderedDict
import pickle

import pandas as pd
import numpy as np
import re

from dtw import dtw

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.ngram import NGramModel
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.vocab import Vocab


def parse_tuple(val):
    # Remove the parentheses and split the string into individual components
    return tuple(map(float, val.strip('()').split()))

class Gaze2Word:
    """Predict word from gaze trace

    The probability combines the dtw distance between the gaze trace and the ideal trace of the word, and the ngram model

    Tips for using this module:
    1. First time initialization can take a while as the text copora are downloaded, and processed. It is recommended to
       save the instance using serializer like pickle for future use.
    2. Adjust the parameter ngram_alpha in the predict method to find an optimal value. Read more in the predict method
    """

    def __init__(self, gaze_data_path, ngram_n=3):
        # TODO change the gaze data to be downloaded automatically

        letters = []
        hit_point_local = []
        key_ground_truth_local = []

        # Read and process the lines
        with open(gaze_data_path, 'r') as file:
            header = file.readline()  # Read the header
            # skip the first line and read the rest
            for i, line in enumerate(file):
                if re.match(r'^[a-zA-Z]', line):
                    parts = line.strip().split(',')
                    letters.append(parts[0])  # Extract the letter
                    hit_point_local.append(parse_tuple(parts[2]))
                    key_ground_truth_local.append(parse_tuple(parts[3]))

        # Create a DataFrame from the extracted data
        df = pd.DataFrame({
            'Letter': letters,
            'HitPointLocal': hit_point_local,
            'KeyGroundTruthLocal': key_ground_truth_local
        })

        # for each unique letter in the letter column, get their KeyGroundTruthLocal put them in a 2d array
        grouped = df.groupby('Letter')['KeyGroundTruthLocal']
        self.letter_locations = dict()
        for letter, group in grouped:
            # Convert the 'KeyGroundTruthLocal' tuples to a 2D array
            ground_truth_array = np.array(list(group))
            assert np.all(np.std(ground_truth_array, axis=0) < np.array([5e-6,
                                                                         5e-6])), f"std of the groundtruth is not close to zero, letter is {letter}, std is {np.std(ground_truth_array, axis=0)}"
            self.letter_locations[letter] = np.mean(ground_truth_array, axis=0)

        # create ideal traces for each word in the vocab
        self.vocab = Vocab()

        self.vocab_traces = OrderedDict()
        self.ngram_model = NGramModel(n=ngram_n)

        for word in self.vocab.vocabulary:
            # Split the word into individual letters
            letters = list(word)
            # Get the corresponding KeyGroundTruthLocal for each letter
            try:
                self.vocab_traces[word] = np.array([self.letter_locations[letter] for letter in letters])
            except KeyError:
                print(f"Ignoring word '{word}' as it contains not a letter")

    def predict(self, k, gaze_trace, prefix=None, ngram_alpha=0.2, ngram_epsilon=1e-8, return_prob=True):
        """Predict the top k most likely next word based on the gaze trace from Sweyepe and previous text

        Tips:
        1. Always use context even if it's empty, because
            see example long_gaze_trace.
        2. play around with the parameter ngram_alpha to find an optimal value.
        3. if you have other probabilities that you want to combine with this one, you can set return_prob to True.

        Args:
            k: int: top k candidate words
            gaze_trace: list of tuples: gaze trace from Sweyepe
            prefix: str: previous text, if is not None, the prediction will be based on the previous text.
                If is None, the prediction will be based on the gaze trace only
            ngram_alpha: float: between 0 and 1, the weight of the ngram model in the prediction
            return_prob: bool: whether to return the probabilities of the top choise.

        Returns:
            list of str: top k candidate words
        """
        assert 0 <= ngram_alpha <= 1, "ngram_alpha should be between 0 and 1"
        distances = [dtw(gaze_trace, template_trace, keep_internals=True, dist_method='euclidean').distance for
                     word, template_trace in self.vocab_traces.items()]

        if prefix is not None:
            # tops = [(word, distances) for word, distances in sorted(zip(self.vocab.vocab_list, distances), key=lambda x: x[1])[:k * 50]]
            tops = [(word, distances) for word, distances in sorted(zip(self.vocab.vocab_list, distances), key=lambda x: x[1])][:k * 10]
            distance_top_words = [word for word, _ in tops]
            distances = [distance for _, distance in tops]

            ngram_preds = self.ngram_model.predict_next(prefix, k='all', ignore_punctuation=True, return_prob=True)
            # combine the distances with the ngram predictions
            # first turn the distances into probabilities
            distances = np.array(distances)
            distance_probs = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

            # find the corresponding ngram probabilities
            ngram_probs = np.array([ngram_preds[word] for word in distance_top_words]) + ngram_epsilon

            combined_prob = (ngram_probs ** ngram_alpha) * (distance_probs ** (1 - ngram_alpha))

            return [(word if not return_prob else word, -prob) for word, prob in sorted(zip(distance_top_words, -combined_prob), key=lambda x: x[1])[:k]]
        else:
            tops = [(w, d) for w, d in sorted(zip(self.vocab.vocab_list, distances), key=lambda x: x[1])[:k]]

            if return_prob: # turn distance into probs
                words = [w for w, distance in tops]
                distances = [d for _, d in tops]
                distances = np.array(distances)
                distance_probs = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
                return [(w, d) for w, d in zip(words, distance_probs)]
            else:
                return tops


if __name__ == '__main__':
    # gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
    gaze_data_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\GazeData.csv'

    g2w = Gaze2Word(gaze_data_path)

    with open('g2w.pkl', 'wb') as f:
        pickle.dump(g2w, f)

    # load it back
    with open('g2w.pkl', 'rb') as f:
        g2w = pickle.load(f)

    # simple test ######################################################################################################
    perfect_gaze_trace = g2w.vocab_traces['hello']

    print(f"Predicted perfect w/o context: {g2w.predict(5, perfect_gaze_trace)}")
    print(f"Predicted perfect w/ context: {g2w.predict(5, perfect_gaze_trace, prefix='')}")

    # test with a long gaze trace (won't work without prefix) ##########################################################
    long_gaze_trace = np.concatenate([np.linspace(g2w.letter_locations['h'], g2w.letter_locations['e'], num=2),
                                            np.linspace(g2w.letter_locations['e'], g2w.letter_locations['l'], num=2),
                                            np.linspace(g2w.letter_locations['l'], g2w.letter_locations['l'], num=2),
                                            np.linspace(g2w.letter_locations['l'], g2w.letter_locations['o'], num=2),
                                            np.linspace(g2w.letter_locations['o'], g2w.letter_locations['o'], num=2)])
    print(f"Predicted long w/o context: {g2w.predict(5, long_gaze_trace)}")
    print(f"Predicted long w/ context: {g2w.predict(5, long_gaze_trace, prefix='')}")

    # test with context  ###############################################################################################
    noisy_gaze_trace = np.concatenate([np.linspace(g2w.letter_locations['d'], g2w.letter_locations['d'], num=2),
                                       np.linspace(g2w.letter_locations['d'], g2w.letter_locations['s'], num=2),
                                       np.linspace(g2w.letter_locations['s'], g2w.letter_locations['a'], num=2),
                                       np.linspace(g2w.letter_locations['a'], g2w.letter_locations['a'], num=2),
                                       np.linspace(g2w.letter_locations['a'], g2w.letter_locations['r'], num=2),
                                       np.linspace(g2w.letter_locations['r'], g2w.letter_locations['y'], num=2),
                                       np.linspace(g2w.letter_locations['y'], g2w.letter_locations['y'], num=2),
                                       ])
    
    print(noisy_gaze_trace)
    print(f"Predicted noisy w/o context: {g2w.predict(5, noisy_gaze_trace, prefix=None)}")
    print(f"Predicted noisy w/ context: {g2w.predict(5, noisy_gaze_trace, prefix='have a nice')}")

