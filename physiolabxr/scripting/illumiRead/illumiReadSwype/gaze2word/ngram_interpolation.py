import pickle
import string
import time

import nltk
from nltk.util import ngrams
from collections import Counter, defaultdict
import ssl
from nltk.corpus import reuters, brown, gutenberg, inaugural

import nltk
import ssl
import string
from collections import defaultdict, Counter
from nltk.util import ngrams
from nltk.corpus import reuters, brown, gutenberg, inaugural

import ssl
import nltk
from nltk.corpus import reuters, brown, gutenberg, inaugural
from nltk.util import ngrams
from collections import defaultdict, Counter
import string

import ssl
import nltk
from nltk.corpus import reuters, brown, gutenberg, inaugural
from nltk.util import ngrams
from collections import defaultdict, Counter
import string


class InterpolatedNGramModel:
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, top_unigrams=300):
        """
        A trigram model with interpolation over tri-, bi-, and uni-gram counts.

        alpha, beta, gamma are the interpolation weights, which should sum to 1.
        top_unigrams is how many of the most frequent unigrams to consider at prediction time.
        """
        # Ensure SSL for NLTK downloads if needed
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download necessary NLTK data (quiet=True to suppress output)
        nltk.download('punkt', quiet=True)
        nltk.download('reuters', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('gutenberg', quiet=True)
        nltk.download('inaugural', quiet=True)

        # Check interpolation weights
        if not abs((alpha + beta + gamma) - 1.0) < 1e-8:
            raise ValueError("Interpolation weights alpha, beta, gamma must sum to 1.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Data structures for unigrams, bigrams, trigrams
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)

        # We'll store total counts for each bigram/trigram prefix to speed up MLE computations
        self.bigram_prefix_sums = {}  # prefix -> (sum_of_counts, distinct_word_count_for_prefix)
        self.trigram_prefix_sums = {}  # (prefix1, prefix2) -> (sum_of_counts, distinct_word_count)

        # Combined corpus from multiple sources
        combined_corpus = (
                list(reuters.sents())
                + list(brown.sents())
                + list(gutenberg.sents())
                + list(inaugural.sents())
        )

        # Build the counts
        self.build_model(combined_corpus)

        # Total tokens for unigrams
        self.total_unigrams = sum(self.unigram_counts.values())
        self.vocab_size = len(self.unigram_counts)

        # Precompute top unigrams to consider for fallback
        self.top_unigrams = [
            w for w, _ in self.unigram_counts.most_common(top_unigrams)
            if w not in ('<s>', '</s>')
        ]

    def build_model(self, corpus):
        """
        Builds unigram, bigram, and trigram counts from the given corpus.
        Also precomputes prefix sums for bigrams/trigrams to speed up lookups.
        """
        for sentence in corpus:
            # Lowercase
            sentence = [word.lower() for word in sentence]
            # Pad with two start tokens for trigrams, plus end token
            sentence = ['<s>', '<s>'] + sentence + ['</s>']

            # 1) Unigrams
            for w in sentence:
                self.unigram_counts[w] += 1

            # 2) Bigrams
            for b in ngrams(sentence, 2):
                prefix = b[0]
                word = b[1]
                self.bigram_counts[prefix][word] += 1

            # 3) Trigrams
            for t in ngrams(sentence, 3):
                prefix = (t[0], t[1])
                word = t[2]
                self.trigram_counts[prefix][word] += 1

        # Precompute sums for each bigram prefix
        for prefix, cnts in self.bigram_counts.items():
            total_count = sum(cnts.values())
            distinct_count = len(cnts)
            self.bigram_prefix_sums[prefix] = (total_count, distinct_count)

        # Precompute sums for each trigram prefix
        for prefix, cnts in self.trigram_counts.items():
            total_count = sum(cnts.values())
            distinct_count = len(cnts)
            self.trigram_prefix_sums[prefix] = (total_count, distinct_count)

    def predict_next(self, prefix, k=5, ignore_punctuation=False, epsilon=1e-8):
        """
        Predict the next word(s) given a prefix, using trigram/bigram/unigram interpolation.

        prefix (str): The text prefix we want to base our prediction on.
        k (int): Number of top candidates to return ('all' or an integer).
        ignore_punctuation (bool): Filter out punctuation from candidate words.
        epsilon (float): Smoothing constant for counts.

        Returns a list of (word, probability) sorted by descending probability.
        """
        # Tokenize and lowercase
        tokens = nltk.word_tokenize(prefix.lower())

        # If fewer than 2 tokens, pad with <s>
        if len(tokens) < 2:
            tokens = ['<s>'] * (2 - len(tokens)) + tokens

        # We'll look up these contexts
        trigram_prefix = (tokens[-2], tokens[-1])
        bigram_prefix = tokens[-1]

        # Gather candidate words from:
        #  1) trigram next words
        #  2) bigram next words
        #  3) top unigrams (to allow fallback)
        candidate_words = set()

        if trigram_prefix in self.trigram_counts:
            candidate_words.update(self.trigram_counts[trigram_prefix].keys())

        if bigram_prefix in self.bigram_counts:
            candidate_words.update(self.bigram_counts[bigram_prefix].keys())

        # Add top unigrams. This prevents iterating over ALL unigrams, which is slow.
        candidate_words.update(self.top_unigrams)

        if ignore_punctuation:
            candidate_words = {
                w for w in candidate_words
                if w not in string.punctuation and w not in ('<s>', '</s>')
            }

        # Get MLE prefix sums (cached) to avoid calling sum(...) each time
        tri_prefix_total, tri_prefix_vocab = self.trigram_prefix_sums.get(trigram_prefix, (0, 0))
        bi_prefix_total, bi_prefix_vocab = self.bigram_prefix_sums.get(bigram_prefix, (0, 0))

        # Calculate probabilities
        word_probs = {}
        for w in candidate_words:
            # Trigram MLE
            tri_count = self.trigram_counts[trigram_prefix][w] if trigram_prefix in self.trigram_counts else 0
            # p_tri = (tri_count + epsilon) / (tri_prefix_total + epsilon * tri_prefix_vocab)
            # Use len(self.trigram_counts[trigram_prefix]) for distinct next words
            if tri_prefix_vocab == 0:
                p_tri = (tri_count + epsilon) / (1.0 + epsilon)  # fallback if no data
            else:
                p_tri = (tri_count + epsilon) / (tri_prefix_total + epsilon * tri_prefix_vocab)

            # Bigram MLE
            bi_count = self.bigram_counts[bigram_prefix][w] if bigram_prefix in self.bigram_counts else 0
            if bi_prefix_vocab == 0:
                p_bi = (bi_count + epsilon) / (1.0 + epsilon)
            else:
                p_bi = (bi_count + epsilon) / (bi_prefix_total + epsilon * bi_prefix_vocab)

            # Unigram MLE
            uni_count = self.unigram_counts[w]
            p_uni = (uni_count + epsilon) / (self.total_unigrams + epsilon * self.vocab_size)

            # Interpolation
            p_interpolated = self.alpha * p_tri + self.beta * p_bi + self.gamma * p_uni
            word_probs[w] = p_interpolated

        # Sort by probability (descending)
        sorted_candidates = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)

        if k == 'all':
            return sorted_candidates
        else:
            return sorted_candidates[:k]

    def predict_word_completion(self, prefix, k=5, ignore_punctuation=True):
        """
         Predict completions for a user input that may end with a partial word
         (e.g., "How are y") or a space (e.g., "How are "), or be entirely empty.

         - If the prefix ends with a partial word (no trailing space), this method
           attempts to complete that partial word.
         - If the prefix ends with a space (or is empty), it predicts the most
           likely next word.

         Parameters
         ----------
         prefix : str
             The user input text, which may:
              - End with a space, indicating the last token is complete.
              - End with a partial token (no trailing space).
              - Be entirely empty or whitespace.
         k : int, optional
             The maximum number of predicted completions to return. Defaults to 5.
         ignore_punctuation : bool, optional
             If True, punctuation tokens are excluded from predictions. Defaults to True.

         Returns
         -------
         List[Tuple[str, int]]
             A list of (word, score) tuples representing the top `k` completions or
             next words, where `score` is the frequency (or count) of the word in
             the model's n-gram distribution.

         Examples
         --------
         >>> # Instantiate an NGramModel with n=3 (trigram)
         >>> ngram_model = NGramModel(n=3)

         >>> # 1) Predict completions for a partial word:
         >>> prefix = "How are y"
         >>> completions = ngram_model.predict_word_completion(prefix, k=5)
         >>> print(completions)
         [('you', 95), ('young', 27), ...]

         >>> # 2) Predict the next word after a trailing space:
         >>> prefix = "How are "
         >>> completions = ngram_model.predict_word_completion(prefix, k=5)
         >>> print(completions)
         [('you', 350), ('we', 120), ...]

         >>> # 3) Predict from an empty prefix:
         >>> prefix = ""
         >>> completions = ngram_model.predict_word_completion(prefix, k=5)
         >>> print(completions)
         [('the', 1025), ('a', 940), ...]
         """
        # Trim trailing spaces, but check if it originally ended with space
        ended_with_space = (prefix.endswith(' ') or len(prefix.strip()) == 0)
        stripped_prefix = prefix.rstrip()

        if not stripped_prefix:
            # If the prefix is empty or all spaces -> just call predict_next with empty prefix
            return self.predict_next(
                prefix="",
                k=k,
                ignore_punctuation=ignore_punctuation
            )

        # Tokenize the stripped prefix
        tokens = nltk.word_tokenize(stripped_prefix)

        if ended_with_space:
            # If the user typed a space last, do a normal next-word prediction
            # because there's no partial word to complete
            return self.predict_next(
                prefix=stripped_prefix,
                k=k,
                ignore_punctuation=ignore_punctuation
            )
        else:
            # We assume the last token is a partial word to be completed
            partial_word = tokens[-1]
            # Build a prefix without the partial word
            prefix_without_partial = " ".join(tokens[:-1])

            # Get top predictions for the prefix without that partial
            raw_predictions = self.predict_next(
                prefix=prefix_without_partial,
                k='all',  # get all predictions, then filter
                ignore_punctuation=ignore_punctuation
            )

            # Filter so we only keep words starting with partial_word
            partial_lower = partial_word.lower()
            matching = [(w, score) for (w, score) in raw_predictions if w.startswith(partial_lower)]

            # If we already have at least k matching completions, just return the top k
            if len(matching) >= k:
                return matching[:k]

            # Otherwise, fill up the remainder from non-matching predictions
            leftover_needed = k - len(matching)
            not_matching = [(w, score) for (w, score) in raw_predictions if not w.startswith(partial_lower)]

            # The raw_predictions list should already be sorted by descending probability
            # or counts, so leftover `not_matching` is also effectively sorted the same way
            # (but if you want to be sure, re-sort):
            # not_matching = sorted(not_matching, key=lambda x: x[1], reverse=True)

            filler = not_matching[:leftover_needed]
            return matching + filler


reload_model = True
# Example usage
if __name__ == "__main__":
    # Load the example corpus (you can use any text corpus)

    # Combine sentences from multiple corpora

    # Create an instance of the NGramModel with n=3 (trigram)

    if reload_model:
        ngram_model = InterpolatedNGramModel()
        pickle.dump(ngram_model, open("ngram_model_interpolate.pkl", "wb"))
    else:
        ngram_model = pickle.load(open("ngram_model_interpolate.pkl", "rb"))

    # Predict the top k most likely next words for a given prefix
    # prefix = "the stock market"
    # top_k_predictions = ngram_model.predict_next(prefix, k=5, ignore_punctuation=True)
    # print(f"Next word predictions for the prefix '{' '.join(prefix)}':")
    # for word, count in top_k_predictions:
    #     print(f"{word}: {count}")

    #
    prefix = ""
    top_k_predictions = ngram_model.predict_next(prefix, k=5, ignore_punctuation=True)
    print(f"Next word predictions for the prefix '{' '.join(prefix)}':")
    for word, count in top_k_predictions:
        print(f"{word}: {count}")

    examples = [
        "How are y",         # partial final word
        "How are ",          # ends with space
        "",                  # empty prefix
        "The economy is bo",  # partial final word
        "But the brigadier dines on"
    ]
    for text in examples:
        start_time = time.perf_counter()
        completions = ngram_model.predict_word_completion(text, k=5, ignore_punctuation=True)
        print(f"Time spent predicting: {time.perf_counter() - start_time:.8f}s")
        print(f"Prefix: '{text}'")
        print("Predictions:")
        for word, score in completions:
            print(f"  {word} : {score}")
        print("---")
