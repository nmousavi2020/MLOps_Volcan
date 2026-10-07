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

#
# Copyright (c) 2026 N. Mousavi, J. Fullea, and S. M. Mousavi
# All rights reserved.
#

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

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from config import (
    EPS,
    EXTREME_THRESHOLD,
    N_SPLITS,
    QUANTILES,
    TEST_FILE,
    TRAIN_FILE,
)
from data.loader import load_data
from models.metrics import compute_metrics
from models.model import predict_from_log, train_gbrt
from utils.tracking import log_metrics, log_model, log_params, start_run

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
})


# ---------------------- OUTPUT DIRECTORIES ----------------------

OUT_PLOT_DIR = "plots"
OUT_PRED_DIR = "predictions"

os.makedirs(OUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUT_PRED_DIR, exist_ok=True)


# ---------------------- GBRT PARAMETERS ----------------------

GBRT_PARAMS = dict(
    loss="huber",
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=4,
    max_features=0.3,
    subsample=1.0,
    random_state=42,
)

QUANTILE_PARAMS = GBRT_PARAMS.copy()
QUANTILE_PARAMS.update({"loss": "quantile"})


# ---------------------- SAVE PREDICTIONS ----------------------

def save_predictions(coords, y_pred, filename):
    df = coords.copy()
    df["predicted_mass"] = y_pred

    path = os.path.join(OUT_PRED_DIR, filename)
    df.to_csv(path, index=False)

    print(f"Predictions saved to {path}")


# ---------------------- K-FOLD TRAINING ----------------------

def kfold_gbrt_log(X, y, n_splits=N_SPLITS):

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    y_true_all = []
    y_pred_all = []
    metrics_all = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):

        with start_run(f"fold_{i + 1}"):

            log_params(GBRT_PARAMS)

            print(f"\nRunning Fold {i + 1}/{n_splits}...")

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Log-transform target
            y_train_log = np.log(y_train + EPS)

            # Train standard GBRT
            model = train_gbrt(
                X_train,
                y_train_log,
                GBRT_PARAMS,
            )

            # Predict
            y_pred = predict_from_log(
                model,
                X_test,
                EPS,
            )

            # ---------------------- Quantile Regression ----------------------

            extreme_mask = y_test > EXTREME_THRESHOLD

            if extreme_mask.any():

                for q in QUANTILES:

                    q_model = GradientBoostingRegressor(
                        **QUANTILE_PARAMS,
                        alpha=q,
                    )

                    q_model.fit(
                        X_train,
                        y_train_log,
                    )

                    y_pred_log_extreme = q_model.predict(X_test)

                    y_pred_extreme = (
                        np.exp(y_pred_log_extreme) - EPS
                    )

                    y_pred[extreme_mask] = (
                        y_pred_extreme[extreme_mask]
                    )

            # ---------------------- Bias Correction ----------------------

            bias = np.mean(y_train) / np.mean(y_pred)
            y_pred *= bias

            # ---------------------- Metrics ----------------------

            metrics = compute_metrics(
                y_test.values,
                y_pred,
                EPS,
            )

            log_metrics(metrics)

            y_true_all.append(y_test.values)
            y_pred_all.append(y_pred)
            metrics_all.append(metrics)

            print(
                f"R2 (linear): {metrics['R2']:.4f}, "
                f"R2 (log): {metrics['R2_log']:.4f}, "
                f"MedRelErr: {metrics['MedRelError']:.3f}"
            )

    return (
        y_true_all,
        y_pred_all,
        metrics_all,
    )


# ---------------------- OBSERVED VS PREDICTED ----------------------

def plot_obs_vs_pred(
    y_true_all,
    y_pred_all,
    metrics_all,
    filename="Observed_vs_Predicted_KFold_Log.png",
):

    plt.figure(figsize=(7, 6))

    colors = plt.cm.viridis(
        np.linspace(0, 1, len(y_true_all))
    )

    for i, (y_true, y_pred) in enumerate(
        zip(y_true_all, y_pred_all)
    ):

        plt.scatter(
            y_true,
            y_pred,
            s=70,
            alpha=0.7,
            color=colors[i],
        )

    max_val = max(
        np.max(y)
        for y in y_true_all
    )

    plt.plot(
        [EPS, max_val],
        [EPS, max_val],
        "k--",
        lw=1,
    )

    plt.xlabel("Observed MASS (Gt)")
    plt.ylabel("Predicted MASS (Gt)")

    plt.xscale("log")
    plt.yscale("log")

    plt.grid(
        True,
        which="both",
        ls="--",
        alpha=0.5,
    )

    legend_labels = [
        f"Fold {i + 1}\n"
        f"R²log = {m['R2_log']:.2f}\n"
        f"MedRelErr = {m['MedRelError']:.2f}\n"
        f"NormMedAE = {m['NormMedAE']:.3f}"
        for i, m in enumerate(metrics_all)
    ]

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=8,
            label=legend_labels[i],
        )
        for i in range(len(metrics_all))
    ]

    plt.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
    )

    path = os.path.join(
        OUT_PLOT_DIR,
        filename,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Plot saved to {path}")


# ---------------------- SMALL / LARGE HISTOGRAMS ----------------------

def plot_small_large_histograms(
    values,
    threshold,
    title_prefix,
    filename_prefix,
):

    values = np.asarray(values)

    small_values = values[
        values <= threshold
    ]

    large_values = values[
        values > threshold
    ]

    for vals, label in zip(
        [small_values, large_values],
        ["Small", "Large"],
    ):

        if len(vals) == 0:
            continue

        plt.figure(figsize=(6, 4))

        plt.hist(
            vals,
            bins=50,
            color="k",
            alpha=0.7,
            edgecolor="k",
        )

        plt.xlabel("Mass (Gt)")
        plt.ylabel("Frequency")

        plt.title(
            f"{title_prefix} ({label} eruptions)"
        )

        plt.grid(
            linestyle="dotted",
            alpha=0.6,
        )

        mean_val = np.mean(vals)
        std_val = np.std(vals)

        plt.text(
            0.98,
            0.95,
            f"Mean: {mean_val:.2f}\n"
            f"Std: {std_val:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
            ),
        )

        plt.tight_layout()

        save_path = os.path.join(
            OUT_PLOT_DIR,
            f"{filename_prefix}_{label.lower()}.png",
        )

        plt.savefig(
            save_path,
            dpi=300,
        )

        plt.close()

        print(
            f"{label} histogram saved to {save_path}"
        )


# ---------------------- MAIN ----------------------

if __name__ == "__main__":

    # Load data
    X, y, coords_train, X_test, coords_test = load_data(
        TRAIN_FILE,
        TEST_FILE,
    )

    # ---------------------- K-FOLD ----------------------

    (
        y_true_all,
        y_pred_all,
        metrics_all,
    ) = kfold_gbrt_log(
        X,
        y,
        n_splits=N_SPLITS,
    )

    plot_obs_vs_pred(
        y_true_all,
        y_pred_all,
        metrics_all,
    )

    # Flatten predictions
    y_pred_flat = np.concatenate(
        y_pred_all
    )

    y_true_flat = np.concatenate(
        y_true_all
    )

    # ---------------------- FINAL MODEL ----------------------

    y_train_log = np.log(
        y + EPS
    )

    final_model = train_gbrt(
        X,
        y_train_log,
        GBRT_PARAMS,
    )

    log_model(final_model)

    y_test_pred = predict_from_log(
        final_model,
        X_test,
        EPS,
    )

    # ---------------------- QUANTILE MODEL ----------------------

    extreme_train_mask = (
        y > EXTREME_THRESHOLD
    )

    X_train_extreme = X[
        extreme_train_mask
    ]

    y_train_extreme_log = np.log(
        y[extreme_train_mask] + EPS
    )

    extreme_mask_test = (
        y_test_pred > EXTREME_THRESHOLD
    )

    if (
        extreme_mask_test.any()
        and len(X_train_extreme) > 0
    ):

        for q in QUANTILES:

            q_model = GradientBoostingRegressor(
                **QUANTILE_PARAMS,
                alpha=q,
            )

            q_model.fit(
                X_train_extreme,
                y_train_extreme_log,
            )

            y_test_pred_extreme = (
                np.exp(
                    q_model.predict(
                        X_test[extreme_mask_test]
                    )
                ) - EPS
            )

            y_test_pred[
                extreme_mask_test
            ] = y_test_pred_extreme

    # ---------------------- SAVE PREDICTIONS ----------------------

    save_predictions(
        coords_test,
        y_test_pred,
        "predicted_mass_test.csv",
    )

    # ---------------------- HISTOGRAMS ----------------------

    threshold_mass = EXTREME_THRESHOLD

    plot_small_large_histograms(
        y_true_flat,
        threshold_mass,
        "Training Mass",
        "training_mass",
    )

    plot_small_large_histograms(
        y_test_pred,
        threshold_mass,
        "Predicted Mass",
        "predicted_mass",
    )

    print("\nAll done!")