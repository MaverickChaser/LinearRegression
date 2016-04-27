import data_io
import numpy as np
import pickle

import metrics
from regression import linear_regression as linreg
from feature_extractor.extractor import Vectorizer

import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer

MAX_DOCUMENTS = 20000
MAX_FEATURES = 100

CATEGORICAL_FIELDS = ['Title', 'ContractType', 'Company']


def form_input(data, feature_extractor, category_vectorizer, train=True):
    documents = data['FullDescription']
    values = []

    arrays = (data[category] for category in CATEGORICAL_FIELDS)
    iterable = zip(*arrays)
    #print(iterable)
    for v in iterable:
        d = {}
        for i, category in enumerate(CATEGORICAL_FIELDS):
            val = v[i]

            if pd.isnull(val):
                d.update({})
            else:
                d.update({category: val})
        values.append(d)
    #print(values)
    if train:
        X = feature_extractor.fit_transform(documents)
        category_vectorizer_matrix = category_vectorizer.fit_transform(values)
        print(category_vectorizer_matrix)
    else:
        X = feature_extractor.transform(documents)
        category_vectorizer_matrix = category_vectorizer.transform(values)

    X = hstack((X, category_vectorizer_matrix))
    print(X)
    return X





def main():
    print("Reading in the training data")
    data = data_io.get_train_df()
    print("Extracting features")
    feature_extractor = Vectorizer(MAX_FEATURES)
    category_vectorizer = DictVectorizer()


    #category_title = pd.get_dummies(train['Title'])
    #print (category_vectorizer.shape, X.shape)

    X = form_input(data, feature_extractor, category_vectorizer)
    #location = pd.get_dummies(train['LocationNormalized'])
    #X = hstack((X, location))
    #contract_time = pd.get_dummies(train['ContractTime'])
    #X = hstack((X, contract_time))
    #print(X)
    y = data["SalaryNormalized"]
    print("Training model")
    linreg.train(X, y)
    print("Making predictions")
    predictions = linreg.predict(X)
    mae_train = metrics.MAE(predictions, data["SalaryNormalized"])
    print('MAE train=%s', mae_train)


    print("Validating...")

    data = data_io.get_valid_df()
    X = form_input(data, feature_extractor, category_vectorizer, train=False)
    predictions = linreg.predict(X)
    data_io.write_submission(predictions)

    '''
    data = data[-MAX_DOCUMENTS / 20:]
    y = data["SalaryNormalized"]
    X = form_input(data, feature_extractor, category_vectorizer, train=False)
    predictions = linreg.predict(X)
    mae_train = metrics.MAE(predictions, data["SalaryNormalized"])
    print('MAE valid=%s', mae_train)
    '''


if __name__ == "__main__":
    main()
