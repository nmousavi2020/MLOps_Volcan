import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
)


def compute_metrics(y_true, y_pred, eps):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r2_log = r2_score(
        np.log(y_true + eps),
        np.log(y_pred + eps),
    )
    med_rel_error = np.median(
        np.abs(y_true - y_pred) / (y_true + eps)
    )
    log_error = np.mean(
        np.abs(
            np.log(y_true + eps) -
            np.log(y_pred + eps)
        )
    )

    value_range = np.max(y_true) - np.min(y_true) + eps

    return {
        "R2": r2,
        "R2_log": r2_log,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MedRelError": med_rel_error,
        "LogError": log_error,
        "NormRMSE": rmse / value_range,
        "NormMAE": mae / value_range,
        "NormMedAE": medae / value_range,
    }