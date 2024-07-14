import nltk
import unicodedata

from nltk.corpus import brown, reuters, gutenberg, wordnet
from nltk.tokenize import word_tokenize
from collections import Counter

import ssl

# Ensure necessary NLTK data is downloaded


def normalize_and_filter(word):
    # Normalize the word to NFKD form
    normalized_word = unicodedata.normalize('NFKD', word)
    # Encode to ASCII bytes, ignoring characters that can't be converted
    ascii_bytes = normalized_word.encode('ascii', 'ignore')
    # Decode back to a string
    ascii_word = ascii_bytes.decode('ascii')
    return ascii_word

class Vocab:
    """Creates a comprehensive vocabulary from multiple corpora and WordNet.


    """
    def __init__(self):
        # fix the SSL certificate issue, needed for downloading the NLTK data
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt')
        nltk.download('brown')
        nltk.download('reuters')
        nltk.download('gutenberg')
        nltk.download('wordnet')

        # Combine multiple corpora
        corpora = brown.words() + reuters.words() + gutenberg.words()

        # Tokenize and build the vocabulary
        tokens = [normalize_and_filter(word.lower()) for word in corpora if word.isalpha()]
        self.vocabulary = Counter(tokens)

        # Add words from WordNet
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                normalized_word = normalize_and_filter(lemma.name().lower())
                self.vocabulary[normalized_word] += 1

        # remove from the counter single-letter words
        # and words that contains punctuation
        for word in list(self.vocabulary.keys()):
            if len(word) == 1:
                self.vocabulary.pop(word)
            elif not word.isalpha():
                self.vocabulary.pop(word)

        # Optionally, lemmatize the tokens to reduce them to base forms
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Build the final vocabulary with lemmatized tokens
        self.lemmatized_vocabulary = Counter(lemmatized_tokens)

        self.vocab_list = list(self.vocabulary.keys())
        self.lemmatized_vocab_list = list(self.lemmatized_vocabulary.keys())

        # Display the most common words
        print(f"Loaded {len(tokens)} tokens from the copora, with a vocabulary size of {len(self.vocabulary)}, and lemmatized vocabulary size of {len(self.lemmatized_vocabulary)}")
        print(f"Most common words:")
        print(f"Original vocabulary: {self.vocabulary.most_common(10)}")
        print(f"Lemmatized vocabulary: {self.lemmatized_vocabulary.most_common(10)}")


if __name__ == "__main__":
    # Create an instance of the vocab class
    vocabulary = Vocab()