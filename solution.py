import data_io
import numpy as np
import pickle

import metrics
from regression import linear_regression as linreg
from feature_extractor.extractor import extract

MAX_DOCUMENTS = 10000

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()[:MAX_DOCUMENTS]
    print("Extracting features")
    X = extract(train["FullDescription"], max_features=100)
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
