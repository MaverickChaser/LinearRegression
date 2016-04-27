import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.porter import PorterStemmer


class Vectorizer(object):
    stemmer = PorterStemmer()

    def __init__(self, max_features):
        self.vectorizer = TfidfVectorizer(tokenizer=Vectorizer.tokenize,
                                          max_features=max_features, stop_words='english')

    @staticmethod
    def tokenize(document):
        tokens = nltk.word_tokenize(document)
        return [Vectorizer.stemmer.stem(token) for token in tokens]

    def fit_transform(self, documents):
        documents = (document.lower().translate(None, string.punctuation) for document in documents)
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents):
        return self.vectorizer.transform(documents)
