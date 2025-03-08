import pickle
import string

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
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('reuters')
        nltk.download('brown')
        nltk.download('gutenberg')
        nltk.download('inaugural')

        assert n >= 2, "n must be greater than or equal to 2 (bigram model)"
        self.n = n
        self.models = {i: defaultdict(Counter) for i in range(2, n + 1)}

        combined_corpus = list(reuters.sents()) + list(brown.sents()) + list(gutenberg.sents()) + list(inaugural.sents())
        self.build_model(combined_corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            sentence = [word.lower() for word in sentence]  # Convert to lower case
            for i in range(2, self.n + 1):
                n_grams = list(ngrams(sentence, i, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
                for ngram in n_grams:
                    prefix = ngram[:-1]
                    suffix = ngram[-1]
                    self.models[i][prefix][suffix] += 1

    def predict_next(self, prefix, k=5, ignore_punctuation=False, return_prob=False, epsilon=1e-8):
        # Tokenize the prefix into words
        tokens = nltk.word_tokenize(prefix.lower())

        # Always add the start token to the prefix
        tokens = ['<s>'] + tokens

        # Determine the length of the prefix
        prefix_length = len(tokens)

        # Use the largest possible n-gram model based on the prefix length
        n = min(prefix_length + 1, self.n)

        # Select the appropriate prefix tuple
        prefix_tuple = tuple(tokens[-(n - 1):])

        next_words = self.models[n][prefix_tuple]

        if ignore_punctuation:
            # Filter out punctuation from the next words
            next_words = Counter({word: count for word, count in next_words.items() if word not in string.punctuation})

        if return_prob:
            # Calculate the total count and convert to probabilities
            total_count = sum(next_words.values())
            next_words = Counter({word: (count + epsilon) / (total_count + epsilon * len(next_words)) for word, count in
                                  next_words.items()})

        if k == 'all':
            return next_words

        return next_words.most_common(k)


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
            filtered = [
                (word, count)
                for word, count in raw_predictions.items()
                if word.startswith(partial_lower)
            ]

            # Sort by count desc, then return top k
            filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
            return filtered_sorted[:k]

reload_model = True
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
        "How are y",         # partial final word
        "How are ",          # ends with space
        "",                  # empty prefix
        "The economy is bo"  # partial final word
    ]
    for text in examples:
        completions = ngram_model.predict_word_completion(text, k=5, ignore_punctuation=True)
        print(f"Prefix: '{text}'")
        print("Predictions:")
        for word, score in completions:
            print(f"  {word} : {score}")
        print("---")
