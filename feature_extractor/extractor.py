import nltk
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(document):
    tokens = nltk.word_tokenize(document)
    return [stemmer.stem(token) for token in tokens]


def extract(documents, max_features=100):
    documents = (document.lower().translate(None, string.punctuation) for document in documents)
    tfidf = TfidfVectorizer(tokenizer=tokenize, max_features=max_features, stop_words='english')
    return tfidf.fit_transform(documents)
