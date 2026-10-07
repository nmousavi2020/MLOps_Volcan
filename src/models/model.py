import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def create_gbrt(params):
    return GradientBoostingRegressor(**params)


def train_gbrt(X, y_log, params):
    model = create_gbrt(params)
    model.fit(X, y_log)
    return model


def predict_from_log(model, X, eps):
    return np.exp(model.predict(X)) - eps