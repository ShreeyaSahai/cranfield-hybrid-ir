import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download NLTK resources if they are not already installed
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


class TextPreprocessor:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        """
        Basic text cleaning.

        Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Remove extra whitespace
        """

        text = text.lower()

        # Keep alphabetic characters and spaces
        text = re.sub(r"[^a-z\s]", " ", text)

        # Remove repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text):
        """
        Split text into individual tokens.
        """

        return text.split()

    def remove_stopwords(self, tokens):
        """
        Remove common English stopwords.
        """

        return [
            token
            for token in tokens
            if token not in self.stop_words
        ]

    def stem_tokens(self, tokens):
        """
        Apply Porter stemming.
        """

        return [
            self.stemmer.stem(token)
            for token in tokens
        ]

    def preprocess(self, text):
        """
        Complete preprocessing pipeline.

        raw text
            ↓
        cleaning
            ↓
        tokenization
            ↓
        stopword removal
            ↓
        stemming
            ↓
        processed tokens
        """

        text = self.clean_text(text)

        tokens = self.tokenize(text)

        tokens = self.remove_stopwords(tokens)

        tokens = self.stem_tokens(tokens)

        return tokens

if __name__ == "__main__":

    processor = TextPreprocessor()

    sample = """
    What similarity laws must be obeyed when constructing
    aeroelastic models of heated high speed aircraft?
    """

    print("Original:")
    print(sample)

    print("\nProcessed:")
    print(processor.preprocess(sample))