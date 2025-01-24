import time
from collections import defaultdict, Counter

import nltk
import numpy as np
from nltk import TreebankWordDetokenizer
from numpy import ndarray
from nltk.corpus import reuters, brown, gutenberg, inaugural
from scipy.stats import norm, multivariate_normal

from physiolabxr.scripting.illumiRead.illumiReadSwype.gaze2word.g2w_utils import parse_letter_locations

import ssl

class Tap2Char:
    """Predict the character based on the tap location

    This class is used to avoid the "fat finger" problem,
    where the user's finger may cover multiple characters on the screen.


    Attributes:
        letter_locations (dict: str->2-element ndarray): A dictionary mapping each letter to its location on the screen.
        letters (list): A list of all the letters on the screen
    """

    def __init__(self, gaze_data_path, verbose=True):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        self.letter_locations = parse_letter_locations(gaze_data_path)
        self.letters = list(self.letter_locations.keys())

        # build the character prediction
        nltk.download('punkt')
        nltk.download('reuters')
        nltk.download('brown')
        nltk.download('gutenberg')
        nltk.download('inaugural')
        combined_corpus = list(reuters.sents()) + list(brown.sents()) + list(gutenberg.sents()) + list(inaugural.sents())

        self.char_model = defaultdict(Counter)
        self.build_char_model(combined_corpus, verbose)

    def build_char_model(self, corpus, verbose=True):
        """Builds a character-level bigram model from the given corpus.

        Args:
            corpus (list of str):
        """
        if verbose:
            print("Building character-level bigram model...")
        detokenizer = TreebankWordDetokenizer()

        for i, line in enumerate(corpus):
            if verbose: print(f"Processing {i}/{len(corpus)} lines...", end='\r')
            # Iterate through each line in the corpus
            detokenized_line = detokenizer.detokenize(line)
            detokenized_line = detokenized_line.lower()
            # Create bigrams and update the model
            for i in range(len(detokenized_line) - 1):
                prefix = detokenized_line[i]
                next_char = detokenized_line[i + 1]
                self.char_model[prefix][next_char] += 1

    def gaussian_probability_2d(self, distance_vector, std_dev_x, std_dev_y):
        """Compute the probability using a 2D Gaussian function centered at the tap position.

        Args:
            distance_vector (ndarray): The distance vector from the tap position to the character location.
            std_dev_x (float): The standard deviation for the Gaussian distribution in the x direction.
            std_dev_y (float): The standard deviation for the Gaussian distribution in the y direction.

        Returns:
            float: The probability of selecting the character based on its distance.
        """
        cov_matrix = np.diag([std_dev_x ** 2, std_dev_y ** 2])
        return multivariate_normal.pdf(distance_vector, mean=[0, 0], cov=cov_matrix)

    def predict(self, tap_position: ndarray, prefix, std_dev_x=0.2835, std_dev_y=0.378, alpha=0.2):
        """predicts the character based on the tap position.

        Args:
            tap_position (ndarray): The tap position on the screen
            prefix (str): The prefix of the word
                for example, if the user has already typed 'hel', the prefix is 'hel'
            std_dev_x (float): The standard deviation for the Gaussian distribution in the x direction.
                This number should equal the key button size in the x direction.
            std_dev_y (float): The standard deviation for the Gaussian distribution in the y direction.
                This number should equal the key button size in the y direction.
            alpha (float): the weight of the prefix in the prediction
        """
        # sample the probabilities of nearby characters with a gaussian distribution centered at the tap position
        distances = {
            letter: np.linalg.norm(tap_position - np.array(self.letter_locations[letter]))
            for letter in self.letters
        }
        probabilities = {
            letter: self.gaussian_probability_2d(distances[letter], std_dev_x, std_dev_y)
            for letter in self.letters
        }

        total_prob = sum(probabilities.values())
        normalized_probs_tap = {letter: prob / total_prob for letter, prob in probabilities.items()}

        if prefix == '':
            # return the sorted probabilities
            return sorted(normalized_probs_tap.items(), key=lambda x: x[1], reverse=True)

        # find the probably of each character given the prefix
        prefix_probs = Counter()
        for letter in normalized_probs_tap.keys():
            prefix_probs[letter] = self.char_model[prefix[-1]].get(letter, 0)

        # normalize the probabilities
        total_prob = sum(prefix_probs.values())
        normalized_probs_prefix = {letter: prob / total_prob for letter, prob in prefix_probs.items()}
        # combine the probabilities
        combined_probs = {letter: alpha * normalized_probs_prefix[letter] + (1 - alpha) * normalized_probs_tap[letter] for letter in self.letters}

        return sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)




if __name__ == '__main__':
    gaze_data_path = 'C:/Users/6173-group/Documents/PhysioLabXR/physiolabxr/scripting/illumiRead/illumiReadSwype/gaze2word/GazeData.csv'
    t2c = Tap2Char(gaze_data_path)

    tap_position = (t2c.letter_locations['l'] + t2c.letter_locations['k']) / 2  # between 'l' and 'k'

    start_time = time.perf_counter()
    print(f"{t2c.predict(tap_position, 'hel') = }")
    print(f"Time spent predicting: {time.perf_counter() - start_time:.8f}s")
