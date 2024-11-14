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

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import parse_letter_locations, \
    run_dbscan_on_gaze
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.ngram import NGramModel
from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.vocab import Vocab


class PREFIX_OPTIONS(Enum):
    FREQUENCY = 1


class Gaze2Word:
    """Predict word from gaze trace

    The probability combines the dtw distance between the gaze trace and the ideal trace of the word, and the ngram model

    Attributes:
        letter_locations: OrderedDict: the average location of the letters in the gaze data
        letters: list: the list of letters in the gaze data
        vocab: Vocab: the vocabulary of the words
        vocab_traces: OrderedDict[str, ndarray]: the ideal traces of the words in the vocabulary
        vocab_traces_starting_letter: OrderedDict[str, OrderedDict[str, ndarray]]: the ideal traces of the words in the vocabulary, grouped by the starting letter
        ngram_model: NGramModel: the ngram model used for prediction


    Tips for using this module:
    1. First time initialization can take a while as the text copora are downloaded, and processed. It is recommended to
       save the instance using serializer like pickle for future use.
    2. Adjust the parameter ngram_alpha in the predict method to find an optimal value. Read more in the predict method

    IMPORTANT NOTE:
    1. this class doesn't predict single-character words.
    """

    def __init__(self, gaze_data_path, ngram_n=3):
        # TODO change the gaze data to be downloaded automatically
        self.letter_locations = parse_letter_locations(gaze_data_path)
        self.letters = list(self.letter_locations.keys())
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

        # keep a dict of words with starting letter
        self.vocab_traces_starting_letter = OrderedDict()
        for l in self.letters:
            _vocab_traces_starting_letter = {word: trace for word, trace in self.vocab_traces.items() if word[0] == l}
            self.vocab_traces_starting_letter[l] = _vocab_traces_starting_letter

        self.trimmed_vocab_traces_starting_letter = None
        self.trimmed_vocab_traces = None
        self.trimmed_vocab_list = None

    def trim_vocab(self, vocab_list: List[str], verbose=True) -> None:
        """Use a smaller set of vocabs.

        Args:
            vocab_list: list: the list of words to keep in the vocab
        """
        assert len(vocab_list) > 0, "vocab_list should not be empty"
        # turn the vocabs to lowercase
        vocab_list = [word.lower() for word in vocab_list]
        # remove the single-character words as g2w doesn't predict single-character words
        vocab_list = [word for word in vocab_list if len(word) > 1]

        # check that all words in the vocab_list are in the vocab
        for word in vocab_list:
            if word not in self.vocab_traces:
                raise ValueError(f"In vocab_list, word '{word}' is not in the vocab. Make sure all words in the vocab_list are in the vocab")

        # create a trimmed version of the vocab traces
        self.trimmed_vocab_traces = {word: trace for word, trace in self.vocab_traces.items() if word in vocab_list}

        assert len(self.trimmed_vocab_traces) > 0, "The trimmed vocab is empty. Make sure the vocab_list is a subset of the original vocab"

        self.trimmed_vocab_list = vocab_list

        # create a trimmed version of the vocab traces
        _vocab_traces_starting_letter = OrderedDict()
        for starting_letter, words in self.vocab_traces_starting_letter.items():
            _vocab_traces_starting_letter[starting_letter] = {word: trace for word, trace in words.items() if word in vocab_list}
        self.trimmed_vocab_traces_starting_letter = _vocab_traces_starting_letter

        if verbose:
            print(f"Trimmed the vocab from {len(self.vocab_traces)} to {len(self.trimmed_vocab_traces)} words")


    def predict(self, k: int, gaze_trace: np.ndarray, timestamps=None,
                prefix: Optional[Union[str, PREFIX_OPTIONS]]=None, ngram_alpha: float=0.05, ngram_epsilon: float=1e-8,
                run_dbscan: bool=False, dbscan_eps: float= 0.1, dbscan_min_samples: int=3,
                filter_by_starting_letter: Optional[float]=None,
                njobs: int=1,
                return_prob: float=True,
                use_trimmed_vocab: bool=False,
                verbose=False) -> list:
        """Predict the top k most likely next word based on the gaze trace from Sweyepe and previous text

        Tips:
        1. Always use context even if it's empty, because
            see example long_gaze_trace.
        2. play around with the parameter ngram_alpha to find an optimal value.
        3. if you have other probabilities that you want to combine with this one, you can set return_prob to True.
        4. set run_dbscan to True to reduce the number of points in the gaze trace.

        Args:
            k: int: top k candidate words
            gaze_trace: 2d array of shape (t, 2): gaze trace from Sweyepe, the first dimension is time, and the second
                dimension is x and y.
                for example, a gaze trace of shape (100, 2) has 100 time points.
            timestamps: 1d array of shape (t,), or None: timestamps for the gaze_trace.
                If given, it should be of the same length as the gaze_trace. It will used in computing DBSCAN,
                such as that DBSCAN takes the order of the points into account.It is highly recommended you
                supply this argument if you are using DBSCAN to make the prediction more accurate.

                If None, the gaze_trace will be treated as a bunch of points without any order information
                when computing DBSCAN.

            Ngram related parameters:
            ------------------------
            prefix: str: previous text, None, or PREFIX_OPTIONS

                If is not None, the prediction will be based on the previous text.
                    Note that an empty string is also a valid prefix. The ngram will prepend the start token to the prefix,
                    meaning the prediction will likely to be words that commonly follow the start token/aka the beginning of the sentence.

                If is None, the prediction will be based on the gaze trace only.

                (Not yet implemented) If is PREFIX_OPTIONS.FREQUENCY, the prediction will be based on the frequency of the words in the text.
                    this is similar to setting the prefix to the empty string, but instead of frequency of the first word,
                    it's the frequency of the whole text.

            ngram_alpha: float: between 0 and 1, the weight of the ngram model in the prediction
            ngram_epsilon: float: a small value to add to the ngram probabilities to avoid zero probabilities

            Cluster related parameters:
            ---------------------------
            run_dbscan: bool: whether to run DBSCAN on the gaze trace before prediction
            dbscan_eps: float: the maximum distance between two samples for one to be considered as in the neighborhood of the other
            dbscan_min_samples: int: the number of samples in a neighborhood for a point to be considered as a core point

            Using a smaller vocab:
            ---------------------
            use_trimmed_vocab: bool: whether to use a smaller vocab.
                You need to call the trim_vocab method to use this feature. If not, it will throw a warning and use the full vocab.
                This is particularly useful when you know your vocabs are drawn from a pool beforehand.

            return_prob: bool: whether to return the probabilities of the top choise.

        Returns:
            list of str: top k candidate words
        """
        assert 0 <= ngram_alpha <= 1, "ngram_alpha should be between 0 and 1"

        if run_dbscan:
            gaze_trace = run_dbscan_on_gaze(gaze_trace, timestamps, dbscan_eps, dbscan_min_samples, verbose)

        if len(gaze_trace) == 0:
            return []

        if len(gaze_trace) == 1:
            gaze_point = gaze_trace[0]
            # just return the word that is closest to the gaze point
            distances = [np.linalg.norm(pos - gaze_point) for letter, pos in self.letter_locations.items()]
            return self.letters[np.argmin(distances)]

        # whether to use the trimmed vocab
        if use_trimmed_vocab and self.trimmed_vocab_traces is not None:
            vocab_traces_starting_letter = self.trimmed_vocab_traces_starting_letter
            template_traces = self.trimmed_vocab_traces
            vocab_list = self.trimmed_vocab_list
        else:
            if use_trimmed_vocab and self.trimmed_vocab_traces is None:
                warnings.warn("Warning: use_trimmed_vocab is set to True, but the trimmed vocab is not set. Using the full vocab instead.")
            vocab_traces_starting_letter = self.vocab_traces_starting_letter
            template_traces = self.vocab_traces
            vocab_list = self.vocab.vocab_list

        if filter_by_starting_letter is not None:
            # find the letters in self.letters that are within the filter_by_starting_letter radius
            first_gaze_point = gaze_trace[0]
            possible_letters = [letter for letter, pos in self.letter_locations.items() if np.linalg.norm(pos - first_gaze_point) < filter_by_starting_letter]
            template_traces = {word: trace for letter in possible_letters for word, trace in vocab_traces_starting_letter[letter].items()}
            vocab_list = list(template_traces.keys())
            if verbose: print(f"Filtering by starting letter reduced the number of words from {len(self.vocab_traces)} to {len(template_traces)}")
        else:
            template_traces = template_traces
            vocab_list = vocab_list

        if njobs == 1:
            distances = [dtw(gaze_trace, template_trace, keep_internals=True, dist_method='euclidean').distance for
                         word, template_trace in template_traces.items()]
        else:
            distances = Parallel(n_jobs=njobs)(delayed(dtw)(gaze_trace, template_trace, keep_internals=True, dist_method='euclidean') for
                         word, template_trace in template_traces.items())
            distances = [result.distance for result in distances]

        if prefix is not None and isinstance(prefix, str):
            # tops = [(word, distances) for word, distances in sorted(zip(vocab_list, distances), key=lambda x: x[1])[:k * 50]]
            tops = [(word, distances) for word, distances in sorted(zip(vocab_list, distances), key=lambda x: x[1])][:k * 10]
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
        elif prefix is None:
            tops = [(w, d) for w, d in sorted(zip(vocab_list, distances), key=lambda x: x[1])[:k]]

            if return_prob: # turn distance into probs
                words = [w for w, distance in tops]
                distances = [d for _, d in tops]
                distances = np.array(distances)
                distance_probs = 1 - (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
                return [(w, d) for w, d in zip(words, distance_probs)]
            else:
                return tops
        elif prefix is PREFIX_OPTIONS.FREQUENCY:
            raise NotImplementedError("prefix option not implemented")
        else:
            raise ValueError("prefix should be None, str, or PREFIX_OPTIONS")


# if __name__ == '__main__':
    # gaze_data_path = '/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
    
    # gaze_data_path = r'C:\Users\Season\Documents\PhysioLab\physiolabxr\scripting\illumiRead\illumiReadSwype\gaze2word\GazeData.csv'

    # g2w = Gaze2Word(gaze_data_path)

    # with open('g2w.pkl', 'wb') as f:
    #     pickle.dump(g2w, f)

    # # load it back
    # with open('g2w.pkl', 'rb') as f:
    #     g2w = pickle.load(f)

    # if os.path.exists('g2w.pkl'):
    #     with open('g2w.pkl', 'rb') as f:
    #         g2w = pickle.load(f)
    # else:
    #     g2w = Gaze2Word(gaze_data_path)
    #     with open('g2w.pkl', 'wb') as f:
    #         pickle.dump(g2w, f)


    # # simple test ######################################################################################################
    # perfect_gaze_trace = g2w.vocab_traces['hello']

    # print(f"Predicted perfect w/o context: {g2w.predict(5, perfect_gaze_trace)}")
    # print(f"Predicted perfect w/ context: {g2w.predict(5, perfect_gaze_trace, prefix='')}")

    # # test with a long gaze trace (won't work without prefix) ##########################################################
    # long_gaze_trace = np.concatenate([np.linspace(g2w.letter_locations['h'], g2w.letter_locations['e'], num=2),
    #                                         np.linspace(g2w.letter_locations['e'], g2w.letter_locations['l'], num=2),
    #                                         np.linspace(g2w.letter_locations['l'], g2w.letter_locations['l'], num=2),
    #                                         np.linspace(g2w.letter_locations['l'], g2w.letter_locations['o'], num=2),
    #                                         np.linspace(g2w.letter_locations['o'], g2w.letter_locations['o'], num=2)])
    # print(f"Predicted long w/o context: {g2w.predict(5, long_gaze_trace)}")
    # print(f"Predicted long w/ context: {g2w.predict(5, long_gaze_trace, prefix='')}")

    # # test with context  ###############################################################################################
    # noisy_gaze_trace = np.concatenate([np.linspace(g2w.letter_locations['d'], g2w.letter_locations['d'], num=2),
    #                                    np.linspace(g2w.letter_locations['d'], g2w.letter_locations['s'], num=2),
    #                                    np.linspace(g2w.letter_locations['s'], g2w.letter_locations['a'], num=2),
    #                                    np.linspace(g2w.letter_locations['a'], g2w.letter_locations['a'], num=2),
    #                                    np.linspace(g2w.letter_locations['a'], g2w.letter_locations['r'], num=2),
    #                                    np.linspace(g2w.letter_locations['r'], g2w.letter_locations['y'], num=2),
    #                                    np.linspace(g2w.letter_locations['y'], g2w.letter_locations['y'], num=2),
    #                                    ])
    
    # print(noisy_gaze_trace)
    # print(f"Predicted noisy w/o context: {g2w.predict(5, noisy_gaze_trace, prefix=None)}")
    # print(f"Predicted noisy w/ context: {g2w.predict(5, noisy_gaze_trace, prefix='have a nice')}")

    