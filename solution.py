import data_io
import numpy as np
import pickle

import metrics
from regression import linear_regression as linreg
from feature_extractor.extractor import FeatureExtractor

MAX_DOCUMENTS = 5000
MAX_FEATURES = 50

FIELDS = ['FullDescription', 'Title', 'LocationRaw']


def main():
    print("Reading in the training data")
    train = data_io.get_train_df()[:MAX_DOCUMENTS]
    print("Extracting features")
    feature_extractor = FeatureExtractor(MAX_FEATURES)
    documents = train['FullDescription']
    X = feature_extractor.extract(documents)
    y = train["SalaryNormalized"]
    print("Training model")
    linreg.train(X, y)
    print("Making predictions")
    predictions = linreg.predict(X)
    mae_train = metrics.MAE(predictions, train["SalaryNormalized"])
    print('MAE train=%s', mae_train)
    data_io.write_submission(predictions)

if __name__ == "__main__":
    main()
