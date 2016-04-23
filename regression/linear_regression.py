from sklearn import linear_model

regr = linear_model.LassoCV()


def train(X, y):
    regr.fit(X, y)


def predict(X):
    return regr.predict(X)
