import pickle
import string
import time
from timeit import timeit

import nltk
from nltk.util import ngrams
from collections import Counter, defaultdict
import ssl
from nltk.corpus import reuters, brown, gutenberg, inaugural


class NGramModel:
    def __init__(self, n):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Ensure necessary nltk data is downloaded
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('reuters')
        nltk.download('brown')
        nltk.download('gutenberg')
        nltk.download('inaugural')

        # n should be >= 2, but we now also handle i=1 for unigrams
        assert n >= 2, "n must be >= 2 (but we also include unigrams inside)."
        self.n = n

        # self.models[i] will map a prefix-tuple of length (i-1) to a Counter of next-token counts.
        # For i=1 (unigram), the prefix-tuple is always ().
        self.models = {i: defaultdict(Counter) for i in range(1, n + 1)}

        # Build a combined corpus from multiple sources
        combined_corpus = (
            list(reuters.sents()) +
            list(brown.sents()) +
            list(gutenberg.sents()) +
            list(inaugural.sents())
        )
        self.build_model(combined_corpus)

    def build_model(self, corpus):
        """
        Build individual n-gram count models for i in [1..n].
        For i=1, we store all unigrams under the prefix ().
        For i>1, prefix is of length (i-1).
        """
        for sentence in corpus:
            # Convert to lower case
            sentence = [word.lower() for word in sentence]

            for i in range(1, self.n + 1):
                # Create i-grams (with padding)
                n_grams = list(ngrams(
                    sentence,
                    i,
                    pad_left=True,
                    pad_right=True,
                    left_pad_symbol='<s>',
                    right_pad_symbol='</s>'
                ))
                # For i=1, prefix=() and suffix=ngram[0]
                # For i>1, prefix=ngram[:-1] and suffix=ngram[-1]
                for ng in n_grams:
                    prefix = ng[:-1]
                    suffix = ng[-1]
                    self.models[i][prefix][suffix] += 1

    def predict_next(self, prefix, k=5,
                     ignore_punctuation=False,
                     return_prob=False,
                     epsilon=1e-8):
        """
        Predict the next word(s) given a prefix using a weighted mixture of
        all valid n-gram orders (1..n). If `return_prob=True`, return
        probabilities. Otherwise, return integer 'scores' (pseudo-counts).
        """
        # Tokenize prefix and prepend start token
        tokens = nltk.word_tokenize(prefix.lower())
        tokens = ['<s>'] + tokens

        # Which n-gram models are valid for this prefix length?
        # For an i-gram, we need at least i-1 tokens in the prefix (0 if i=1).
        prefix_length = len(tokens)
        valid_ngram_orders = [
            i for i in range(1, self.n + 1)
            if prefix_length >= (i - 1)
        ]

        if not valid_ngram_orders:
            return []

        # Use uniform weights among all valid orders
        weight = 1.0 / len(valid_ngram_orders)

        next_word_probs = defaultdict(float)

        for i in valid_ngram_orders:
            prefix_tuple = tuple(tokens[-(i - 1):]) if i > 1 else ()
            counts_i = self.models[i][prefix_tuple]
            total_counts_i = sum(counts_i.values())
            if total_counts_i == 0:
                continue

            vocab_size_i = len(counts_i)
            for word, count in counts_i.items():
                if ignore_punctuation and word in string.punctuation:
                    continue
                # Compute smoothed probability
                p_i = (count + epsilon) / (total_counts_i + epsilon * vocab_size_i)
                # Accumulate in the mixture
                next_word_probs[word] += weight * p_i

        if not next_word_probs:
            return []

        # Sort by accumulated probability (descending)
        sorted_words = sorted(next_word_probs.items(), key=lambda x: x[1], reverse=True)

        # If k == 'all', we return all candidates
        if k == 'all':
            result = sorted_words
        else:
            result = sorted_words[:k]

        # If return_prob=True, return (word, probability).
        # If return_prob=False, return (word, integer_score).
        if return_prob:
            # Already in the form (word, probability)
            return result
        else:
            # Convert probability to an integer 'score' (like a pseudo-count).
            # Adjust the scaling (1e6 here) however you want.
            scored_result = [(w, int(round(p * 1e6))) for w, p in result]
            return scored_result

    def predict_word_completion(self, prefix, k=5, ignore_punctuation=True,
                                return_prob=False):
        """
        Predict completions for a partial final word (no trailing space),
        or next-word if prefix ends in space or is empty.
        """
        ended_with_space = prefix.endswith(' ') or len(prefix.strip()) == 0
        stripped_prefix = prefix.rstrip()

        # 1) If prefix is empty or all spaces, predict from ""
        if not stripped_prefix:
            return self.predict_next(
                prefix="",
                k=k,
                ignore_punctuation=ignore_punctuation,
                return_prob=return_prob
            )

        # 2) Otherwise, tokenize
        tokens = nltk.word_tokenize(stripped_prefix)

        if ended_with_space:
            # If the user typed a space last, do a normal next-word prediction
            return self.predict_next(
                prefix=stripped_prefix,
                k=k,
                ignore_punctuation=ignore_punctuation,
                return_prob=return_prob
            )
        else:
            # The last token is a partial word to complete
            partial_word = tokens[-1]
            prefix_without_partial = " ".join(tokens[:-1])

            # Get a distribution for the prefix (all words), then filter
            raw_predictions = self.predict_next(
                prefix=prefix_without_partial,
                k='all',
                ignore_punctuation=ignore_punctuation,
                return_prob=True  # temporarily get probabilities
            )

            partial_lower = partial_word.lower()

            # Filter to those starting with partial_lower
            filtered = [(w, p) for (w, p) in raw_predictions if w.startswith(partial_lower)]
            # Sort descending by probability
            filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
            if k != 'all':
                filtered_sorted = filtered_sorted[:k]

            if return_prob:
                # Keep as probabilities
                return filtered_sorted
            else:
                # Convert to integer scores
                return {(w, int(round(p * 1e6))) for (w, p) in filtered_sorted}

reload_model = False
# Example usage
if __name__ == "__main__":
    # Load the example corpus (you can use any text corpus)

    # Combine sentences from multiple corpora

    # Create an instance of the NGramModel with n=3 (trigram)

    if reload_model:
        ngram_model = NGramModel(n=3)
        pickle.dump(ngram_model, open("ngram_model.pkl", "wb"))
    else:
        ngram_model = pickle.load(open("ngram_model.pkl", "rb"))

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
        # "How are y",         # partial final word
        # "How are ",          # ends with space
        # "",                  # empty prefix
        # "The economy is bo",  # partial final word
        # "",
        # "A",
        # "How are c",
        # "Ap",
        "The name of the mailboxes means nothin",
        "This was a slightly differen",
        'The rustling prob',
        'And this is wha'
    ]
    for text in examples:
        start_time = time.perf_counter()
        completions = ngram_model.predict_word_completion(text, k=5, ignore_punctuation=True, return_prob=True)
        elapsed_time = time.perf_counter() - start_time

        print(f"Predicting using prefix: '{text}'")
        print(f"Took {elapsed_time} seconds")
        for word, score in completions:
            print(f"  {word} : {score}")
        print("---")
