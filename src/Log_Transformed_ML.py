#
# Copyright (c) 2026 N. Mousavi, J. Fullea, and S. M. Mousavi
# All rights reserved.
#
# Description:
#     Machine-learning workflow for volcanic eruption mass estimation using
#     Gradient Boosting Regression Trees (GBRT), logarithmic target
#     transformation.
#
# Reference:
#     Mousavi, N., Fullea, J., & Mousavi, S. M. (2026).
#     A machine learning approach for volcanic eruption mass estimation.
#     Journal of Geophysical Research: Machine Learning and Computation, 3,
#     e2026JH001264.
#
# DOI:
#     https://doi.org/10.1029/2026JH001264
#
# Input files:
#     global.csv
#     gris_features.csv
#
# Outputs:
#     predictions/predicted_mass_with_uncertainty.csv
#     plots/errorbar_plot.png
#
# =============================================================================

print()
print("╭────────────────────────────────────────────────────────────────────╮")
print("│                     Log_Transformed_ML                             │")
print("│              GBRT + Log-Target Transformation                      │")
print("│           Quantile Regression + Predictive Modeling                │")
print("│                                                                    │")
print("│   Copyright © 2026 N. Mousavi, J. Fullea, and S. M. Mousavi        │")
print("│   For citation and reference information, please see the README.   │")
print("╰────────────────────────────────────────────────────────────────────╯")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 24
})

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

# ---------------------- CONFIG ----------------------
N_SPLITS = 2
EPS = 1e-6
EXTREME_THRESHOLD = 1.5  # Gt, mass above which quantile regression is applied
QUANTILES = [0.8]

TRAIN_FILE = "global.csv"
TEST_FILE = "gris_features.csv"

OUT_PLOT_DIR = "plots"
OUT_PRED_DIR = "predictions"
os.makedirs(OUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUT_PRED_DIR, exist_ok=True)

# ---------------------- GBRT PARAMETERS ----------------------
GBRT_PARAMS = dict(
    #loss="squared_error",
    loss="huber",
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=4,
    max_features=0.3,
    subsample=1.0,
    random_state=42,
)

QUANTILE_PARAMS = GBRT_PARAMS.copy()
QUANTILE_PARAMS.update({"loss": "quantile"})  # For quantile regression

# ---------------------- LOAD DATA ----------------------
def load_data():
    train = pd.read_csv(TRAIN_FILE, index_col=0)
    test = pd.read_csv(TEST_FILE, index_col=0)

    feature_cols = train.columns.drop(["Mass", "lon", "lat"])
    X = train[feature_cols].copy()
    y = train["Mass"].copy()
    coords_train = train[["lon", "lat"]].copy()
    X_test = test[feature_cols].copy()
    coords_test = test[["lon", "lat"]].copy()

    # Standardize features
    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    return X, y, coords_train, X_test, coords_test

# ---------------------- SAVE PREDICTIONS ----------------------
def save_predictions(coords, y_pred, filename):
    """
    coords  : DataFrame with 'lon' and 'lat' columns
    y_pred  : array-like, predicted values
    filename: str, file name to save CSV
    """
    df = coords.copy()
    df["predicted_mass"] = y_pred
    df.to_csv(os.path.join(OUT_PRED_DIR, filename), index=False)
    print(f"Predictions saved to {os.path.join(OUT_PRED_DIR, filename)}")
    
# ---------------------- METRICS ----------------------
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r2_log = r2_score(np.log(y_true + EPS), np.log(y_pred + EPS))
    med_rel_error = np.median(np.abs(y_true - y_pred) / (y_true + EPS))
    log_error = np.mean(np.abs(np.log(y_true + EPS) - np.log(y_pred + EPS)))

    return dict(
        R2=r2, R2_log=r2_log,
        RMSE=rmse, MAE=mae, MedAE=medae,
        MedRelError=med_rel_error, LogError=log_error,
        NormRMSE=rmse/(np.max(y_true)-np.min(y_true)+EPS),
        NormMAE=mae/(np.max(y_true)-np.min(y_true)+EPS),
        NormMedAE=medae/(np.max(y_true)-np.min(y_true)+EPS)
    )

# ---------------------- K-FOLD TRAINING ----------------------
def kfold_gbrt_log(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all, y_pred_all, metrics_all = [], [], []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nRunning Fold {i+1}/{n_splits}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Log-transform target
        y_train_log = np.log(y_train + EPS)

        # Train standard GBRT
        model = GradientBoostingRegressor(**GBRT_PARAMS)
        model.fit(X_train, y_train_log)

        # Predict
        y_pred_log = model.predict(X_test)
        y_pred = np.exp(y_pred_log) - EPS

        # ---------------------- Quantile for extremes ----------------------
        extreme_mask = y_test > EXTREME_THRESHOLD
        if extreme_mask.any():
            for q in QUANTILES:
                q_model = GradientBoostingRegressor(**QUANTILE_PARAMS, alpha=q)
                q_model.fit(X_train, y_train_log)
                y_pred_log_extreme = q_model.predict(X_test)
                y_pred_extreme = np.exp(y_pred_log_extreme) - EPS
                # Replace only extreme values
                y_pred[extreme_mask] = y_pred_extreme[extreme_mask]

        # ---------------------- Bias correction to match training mean ----------------------
        bias = np.mean(y_train) / np.mean(y_pred)
        y_pred *= bias

        # Metrics
        metrics = compute_metrics(y_test.values, y_pred)
        y_true_all.append(y_test.values)
        y_pred_all.append(y_pred)
        metrics_all.append(metrics)

        print(f"R2 (linear): {metrics['R2']:.4f}, R2 (log): {metrics['R2_log']:.4f}, MedRelErr: {metrics['MedRelError']:.3f}")

    return y_true_all, y_pred_all, metrics_all

# ----------------------
# PLOT
# ----------------------
def plot_obs_vs_pred(y_true_all, y_pred_all, metrics_all, filename="Observed_vs_Predicted_KFold_Log.png"):
    plt.figure(figsize=(7, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(y_true_all)))

    for i, (y_true, y_pred) in enumerate(zip(y_true_all, y_pred_all)):
        plt.scatter(y_true, y_pred, s=70, alpha=0.7, color=colors[i])

    # Ideal line
    max_val = max([np.max(y) for y in y_true_all])
    plt.plot([EPS, max_val], [EPS, max_val], "k--", lw=1)

    plt.xlabel("Observed MASS (Gt)")
    plt.ylabel("Predicted MASS (Gt)")

    # Log-log plot (important)
    plt.xscale("log")
    plt.yscale("log")

    plt.grid(True, which="both", ls="--", alpha=0.5)

    legend_labels = [
        f"Fold {i+1}\n"
        f"R²log = {m['R2_log']:.2f}\n"
        f"MedRelErr = {m['MedRelError']:.2f}\n"
        f"NormMedAE = {m['NormMedAE']:.3f}"
        for i, m in enumerate(metrics_all)
    ]

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[i], markersize=8,
                   label=legend_labels[i])
        for i in range(len(metrics_all))
    ]

    plt.legend(handles=handles, loc="lower right", fontsize=8)

    # Save
    path = os.path.join(OUT_PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Plot saved to {path}")

# ---------------------- HISTOGRAM ----------------------
def plot_histogram(values, title, filename):
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=50, color="k", alpha=0.7, edgecolor="k")
    plt.xlabel("Mass (Gt)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(linestyle="dotted", alpha=0.6)

    # Calculate mean and std
    mean_val = np.mean(values)
    std_val = np.std(values)

    # Annotate on plot
    plt.text(
        0.98, 0.95, f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_PLOT_DIR, filename), dpi=300)
    plt.close()
    print(f"Histogram saved to {filename}")

    # ---------------------- HISTOGRAM FOR SMALL AND LARGE VALUES ----------------------
def plot_small_large_histograms(values, threshold, title_prefix, filename_prefix):
    """
    values    : array-like, eruption mass values
    threshold : float, value separating small and large eruptions
    title_prefix : str, title prefix
    filename_prefix : str, file prefix for saving plots
    """
    small_values = values[values <= threshold]
    large_values = values[values > threshold]

    for vals, label in zip([small_values, large_values], ["Small", "Large"]):
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=50, color="k", alpha=0.7, edgecolor="k")
        plt.xlabel("Mass (Gt)")
        plt.ylabel("Frequency")
        plt.title(f"{title_prefix} ({label} eruptions)")
        plt.grid(linestyle="dotted", alpha=0.6)

        mean_val = np.mean(vals)
        std_val = np.std(vals)
        plt.text(
            0.98, 0.95, f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10, va="top", ha="right",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
        )

        plt.tight_layout()
        save_path = os.path.join(OUT_PLOT_DIR, f"{filename_prefix}_{label.lower()}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"{label} histogram saved to {save_path}")
        
# ---------------------- MAIN ----------------------
if __name__ == "__main__":

    X, y, coords_train, X_test, coords_test = load_data()

    # K-Fold Training + Prediction
    y_true_all, y_pred_all, metrics_all = kfold_gbrt_log(X, y, n_splits=N_SPLITS)

    plot_obs_vs_pred(y_true_all, y_pred_all, metrics_all)

    # Flatten predictions for histogram
    y_pred_flat = np.concatenate(y_pred_all)
    y_true_flat = np.concatenate(y_true_all)

    # ---------------------- TEST SET PREDICTION ----------------------
    # Train on full data
    y_train_log = np.log(y + EPS)
    final_model = GradientBoostingRegressor(**GBRT_PARAMS)
    final_model.fit(X, y_train_log)
    y_test_pred = np.exp(final_model.predict(X_test)) - EPS
    
    # Quantile correction for large eruptions
    extreme_train_mask = y > EXTREME_THRESHOLD
    X_train_extreme = X[extreme_train_mask]
    y_train_extreme_log = np.log(y[extreme_train_mask] + EPS)
    
    # Find which test samples are extreme
    extreme_mask_test = y_test_pred > EXTREME_THRESHOLD
    
    if extreme_mask_test.any():
        for q in QUANTILES:
            q_model = GradientBoostingRegressor(**QUANTILE_PARAMS, alpha=q)
            q_model.fit(X_train_extreme, y_train_extreme_log)  # train on extreme eruptions only
            y_test_pred_extreme = np.exp(q_model.predict(X_test[extreme_mask_test])) - EPS
            y_test_pred[extreme_mask_test] = y_test_pred_extreme

    # Save predictions
    save_predictions(coords_test, y_test_pred, "predicted_mass_test.csv")

    # Small vs Large
    threshold_mass = 1.5  # adjust based on your data
    plot_small_large_histograms(y_true_flat, threshold_mass, "Training Mass", "training_mass")
    plot_small_large_histograms(y_test_pred, threshold_mass, "Predicted Mass", "predicted_mass")
    

    print("\nAll done!")