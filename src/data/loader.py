import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(train_file, test_file):
    train = pd.read_csv(train_file, index_col=0)
    test = pd.read_csv(test_file, index_col=0)

    feature_cols = train.columns.drop(["Mass", "lon", "lat"])

    X = train[feature_cols].copy()
    y = train["Mass"].copy()

    coords_train = train[["lon", "lat"]].copy()

    X_test = test[feature_cols].copy()
    coords_test = test[["lon", "lat"]].copy()

    scaler = StandardScaler()

    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    return X, y, coords_train, X_test, coords_test