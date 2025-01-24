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


# Example usage
if __name__ == "__main__":
    # Load the example corpus (you can use any text corpus)

    # Combine sentences from multiple corpora

    # Create an instance of the NGramModel with n=3 (trigram)
    ngram_model = NGramModel(n=3)

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
