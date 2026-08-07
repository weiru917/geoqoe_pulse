from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from static_feature_registry import (
    APPROVED_STATIC_FEATURES,
    ID_COLUMN,
    LABEL_STATUS_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    SPATIAL_GROUP_COLUMN,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
    / "cell_static_qoe_baseline.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
)

FOLD_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "static_baseline_fold_metrics.csv"
)

OVERALL_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "static_baseline_overall_metrics.json"
)

OOF_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "static_baseline_oof_predictions.parquet"
)

RIDGE_COEFFICIENTS_PATH = (
    OUTPUT_DIRECTORY
    / "static_ridge_coefficients.csv"
)


NUMBER_OF_FOLDS = 5

# A modest amount of regularisation.
# We are not tuning this yet because Ridge is only a baseline.
RIDGE_ALPHA = 10.0


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict:
    """
    Calculate regression metrics.

    MAE:
        Average absolute prediction error in Mbps.

    RMSE:
        Penalises large errors more heavily than MAE.

    R²:
        Compares the model against predicting the overall mean.
        R² below zero means the model performed worse than that
        simple mean-based reference.

    Spearman:
        Measures whether cells are ranked in approximately the
        correct QoE order, even when exact Mbps values differ.
    """

    spearman_result = spearmanr(
        y_true,
        y_prediction,
    )

    return {
        "mae_mbps": float(
            mean_absolute_error(
                y_true,
                y_prediction,
            )
        ),
        "rmse_mbps": float(
            root_mean_squared_error(
                y_true,
                y_prediction,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_prediction,
            )
        ),
        "spearman_correlation": float(
            spearman_result.statistic
        ),
    }


def print_metrics(
    model_name: str,
    metrics: dict,
) -> None:
    """Print one set of regression metrics."""

    print(f"\n{model_name}")

    print(
        "MAE:",
        f"{metrics['mae_mbps']:.4f} Mbps",
    )

    print(
        "RMSE:",
        f"{metrics['rmse_mbps']:.4f} Mbps",
    )

    print(
        "R²:",
        f"{metrics['r2']:.4f}",
    )

    print(
        "Spearman correlation:",
        f"{metrics['spearman_correlation']:.4f}",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static Regression Baselines")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Static dataset not found:\n{DATA_PATH}"
        )

    required_columns = [
        ID_COLUMN,
        SPATIAL_GROUP_COLUMN,
        LABEL_STATUS_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        TARGET_COLUMN,
        *APPROVED_STATIC_FEATURES,
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    print(f"\nLoading real-labelled static data from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    labelled_df = df[
        df[LABEL_STATUS_COLUMN].astype(bool)
    ].copy()

    labelled_df = labelled_df.reset_index(drop=True)

    X = labelled_df[
        APPROVED_STATIC_FEATURES
    ].astype("float64")

    y = labelled_df[
        TARGET_COLUMN
    ].astype("float64")

    groups = labelled_df[
        SPATIAL_GROUP_COLUMN
    ].astype(str)

    print("\n--- Dataset summary ---")

    print(
        "Real-labelled cells:",
        f"{len(labelled_df):,}",
    )

    print(
        "Approved input features:",
        X.shape[1],
    )

    print(
        "Spatial parent groups:",
        f"{groups.nunique():,}",
    )

    print(
        "Target mean:",
        f"{y.mean():.4f} Mbps",
    )

    print(
        "Target median:",
        f"{y.median():.4f} Mbps",
    )

    print(
        "Target standard deviation:",
        f"{y.std():.4f} Mbps",
    )

    # Confirm all input values are usable.
    X_is_finite = bool(
        np.isfinite(
            X.to_numpy(copy=False)
        ).all()
    )

    y_is_finite = bool(
        np.isfinite(
            y.to_numpy(copy=False)
        ).all()
    )

    print("\n--- Numerical check ---")

    print(
        "All feature values are finite:",
        X_is_finite,
    )

    print(
        "All target values are finite:",
        y_is_finite,
    )

    if not X_is_finite or not y_is_finite:
        raise ValueError(
            "Features or target contain NaN or infinite values."
        )

    group_kfold = GroupKFold(
        n_splits=NUMBER_OF_FOLDS
    )

    # Each cell receives exactly one prediction from the fold
    # where it served as validation data.
    dummy_oof_prediction = np.full(
        shape=len(labelled_df),
        fill_value=np.nan,
        dtype="float64",
    )

    ridge_oof_prediction = np.full(
        shape=len(labelled_df),
        fill_value=np.nan,
        dtype="float64",
    )

    fold_assignment = np.full(
        shape=len(labelled_df),
        fill_value=-1,
        dtype="int8",
    )

    fold_metric_rows = []
    ridge_coefficient_rows = []

    total_start_time = time.perf_counter()

    print("\n--- Training five spatial folds ---")

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"FOLD {fold_number}")
        print("-" * 70)

        X_train = X.iloc[
            train_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_validation = y.iloc[
            validation_indices
        ]

        train_groups = set(
            groups.iloc[train_indices]
        )

        validation_groups = set(
            groups.iloc[validation_indices]
        )

        overlapping_groups = (
            train_groups.intersection(
                validation_groups
            )
        )

        print(
            "Training cells:",
            f"{len(train_indices):,}",
        )

        print(
            "Validation cells:",
            f"{len(validation_indices):,}",
        )

        print(
            "Training spatial groups:",
            f"{len(train_groups):,}",
        )

        print(
            "Validation spatial groups:",
            f"{len(validation_groups):,}",
        )

        print(
            "Overlapping spatial groups:",
            len(overlapping_groups),
        )

        if overlapping_groups:
            raise ValueError(
                f"Spatial leakage detected in fold {fold_number}."
            )

        # ====================================================
        # MODEL 1: MEDIAN DUMMY REGRESSOR
        # ====================================================

        dummy_model = DummyRegressor(
            strategy="median"
        )

        dummy_model.fit(
            X_train,
            y_train,
        )

        dummy_prediction = dummy_model.predict(
            X_validation
        )

        dummy_oof_prediction[
            validation_indices
        ] = dummy_prediction

        dummy_metrics = calculate_regression_metrics(
            y_true=y_validation.to_numpy(),
            y_prediction=dummy_prediction,
        )

        print_metrics(
            "Median dummy regressor",
            dummy_metrics,
        )

        # ====================================================
        # MODEL 2: RIDGE REGRESSION
        # ====================================================

        # StandardScaler learns the training-set mean and
        # standard deviation for every feature.
        #
        # Ridge then learns a regularised linear relationship.
        ridge_model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "ridge",
                    Ridge(
                        alpha=RIDGE_ALPHA,
                    ),
                ),
            ]
        )

        fold_start_time = time.perf_counter()

        ridge_model.fit(
            X_train,
            y_train,
        )

        ridge_training_seconds = (
            time.perf_counter()
            - fold_start_time
        )

        ridge_prediction = ridge_model.predict(
            X_validation
        )

        # Download speed cannot physically be negative.
        ridge_prediction = np.clip(
            ridge_prediction,
            a_min=0.0,
            a_max=None,
        )

        ridge_oof_prediction[
            validation_indices
        ] = ridge_prediction

        ridge_metrics = calculate_regression_metrics(
            y_true=y_validation.to_numpy(),
            y_prediction=ridge_prediction,
        )

        print_metrics(
            "Ridge Regression",
            ridge_metrics,
        )

        print(
            "Ridge training time:",
            f"{ridge_training_seconds:.4f} seconds",
        )

        print(
            "Ridge MAE improvement over dummy:",
            f"{dummy_metrics['mae_mbps'] - ridge_metrics['mae_mbps']:.4f} Mbps",
        )

        fold_assignment[
            validation_indices
        ] = fold_number

        # Save one row per model per fold.
        for model_name, metrics in [
            (
                "median_dummy",
                dummy_metrics,
            ),
            (
                "ridge",
                ridge_metrics,
            ),
        ]:
            fold_metric_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "training_cells": len(
                        train_indices
                    ),
                    "validation_cells": len(
                        validation_indices
                    ),
                    "training_groups": len(
                        train_groups
                    ),
                    "validation_groups": len(
                        validation_groups
                    ),
                    "mae_mbps": metrics[
                        "mae_mbps"
                    ],
                    "rmse_mbps": metrics[
                        "rmse_mbps"
                    ],
                    "r2": metrics["r2"],
                    "spearman_correlation": metrics[
                        "spearman_correlation"
                    ],
                }
            )

        # Retrieve coefficients learned by Ridge after scaling.
        ridge_estimator = (
            ridge_model.named_steps["ridge"]
        )

        for feature, coefficient in zip(
            APPROVED_STATIC_FEATURES,
            ridge_estimator.coef_,
        ):
            ridge_coefficient_rows.append(
                {
                    "fold": fold_number,
                    "feature": feature,
                    "standardized_coefficient": float(
                        coefficient
                    ),
                }
            )

    total_training_seconds = (
        time.perf_counter()
        - total_start_time
    )

    # --------------------------------------------------------
    # Verify that all rows received predictions.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RECONCILIATION")
    print("=" * 70)

    dummy_missing_predictions = int(
        np.isnan(
            dummy_oof_prediction
        ).sum()
    )

    ridge_missing_predictions = int(
        np.isnan(
            ridge_oof_prediction
        ).sum()
    )

    unassigned_folds = int(
        (fold_assignment == -1).sum()
    )

    print(
        "Missing dummy predictions:",
        dummy_missing_predictions,
    )

    print(
        "Missing Ridge predictions:",
        ridge_missing_predictions,
    )

    print(
        "Rows without fold assignment:",
        unassigned_folds,
    )

    if (
        dummy_missing_predictions > 0
        or ridge_missing_predictions > 0
        or unassigned_folds > 0
    ):
        raise ValueError(
            "One or more labelled cells did not receive "
            "an out-of-fold prediction."
        )

    # --------------------------------------------------------
    # Calculate overall OOF metrics.
    # --------------------------------------------------------

    y_array = y.to_numpy()

    dummy_overall_metrics = (
        calculate_regression_metrics(
            y_true=y_array,
            y_prediction=dummy_oof_prediction,
        )
    )

    ridge_overall_metrics = (
        calculate_regression_metrics(
            y_true=y_array,
            y_prediction=ridge_oof_prediction,
        )
    )

    print("\n--- Overall out-of-fold results ---")

    print_metrics(
        "Median dummy regressor",
        dummy_overall_metrics,
    )

    print_metrics(
        "Ridge Regression",
        ridge_overall_metrics,
    )

    mae_improvement = (
        dummy_overall_metrics["mae_mbps"]
        - ridge_overall_metrics["mae_mbps"]
    )

    rmse_improvement = (
        dummy_overall_metrics["rmse_mbps"]
        - ridge_overall_metrics["rmse_mbps"]
    )

    print("\n--- Overall comparison ---")

    print(
        "Ridge MAE improvement over dummy:",
        f"{mae_improvement:.4f} Mbps",
    )

    print(
        "Ridge RMSE improvement over dummy:",
        f"{rmse_improvement:.4f} Mbps",
    )

    print(
        "Total five-fold training time:",
        f"{total_training_seconds:.4f} seconds",
    )

    # --------------------------------------------------------
    # Summarise metrics across folds.
    # --------------------------------------------------------

    fold_metrics_df = pd.DataFrame(
        fold_metric_rows
    )

    print("\n--- Mean and variation across folds ---")

    fold_summary = (
        fold_metrics_df
        .groupby("model")
        .agg(
            mean_mae_mbps=(
                "mae_mbps",
                "mean",
            ),
            std_mae_mbps=(
                "mae_mbps",
                "std",
            ),
            mean_rmse_mbps=(
                "rmse_mbps",
                "mean",
            ),
            std_rmse_mbps=(
                "rmse_mbps",
                "std",
            ),
            mean_r2=(
                "r2",
                "mean",
            ),
            std_r2=(
                "r2",
                "std",
            ),
            mean_spearman=(
                "spearman_correlation",
                "mean",
            ),
            std_spearman=(
                "spearman_correlation",
                "std",
            ),
        )
    )

    print(fold_summary)

    # --------------------------------------------------------
    # Save OOF predictions and metrics.
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    oof_df = labelled_df[
        [
            ID_COLUMN,
            SPATIAL_GROUP_COLUMN,
            TARGET_COLUMN,
            SAMPLE_WEIGHT_COLUMN,
        ]
    ].copy()

    oof_df["spatial_cv_fold"] = (
        fold_assignment
    )

    oof_df[
        "dummy_prediction_mbps"
    ] = dummy_oof_prediction

    oof_df[
        "ridge_prediction_mbps"
    ] = ridge_oof_prediction

    oof_df[
        "dummy_absolute_error_mbps"
    ] = np.abs(
        oof_df[TARGET_COLUMN]
        - oof_df["dummy_prediction_mbps"]
    )

    oof_df[
        "ridge_absolute_error_mbps"
    ] = np.abs(
        oof_df[TARGET_COLUMN]
        - oof_df["ridge_prediction_mbps"]
    )

    oof_df.to_parquet(
        OOF_PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    fold_metrics_df.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    ridge_coefficients_df = pd.DataFrame(
        ridge_coefficient_rows
    )

    ridge_coefficients_df.to_csv(
        RIDGE_COEFFICIENTS_PATH,
        index=False,
    )

    overall_metrics = {
        "dataset": (
            "cell_static_qoe_baseline.parquet"
        ),
        "data_status": (
            "real quarterly Ookla targets"
        ),
        "target": TARGET_COLUMN,
        "approved_features": (
            APPROVED_STATIC_FEATURES
        ),
        "number_of_features": len(
            APPROVED_STATIC_FEATURES
        ),
        "labelled_cells": len(
            labelled_df
        ),
        "spatial_groups": int(
            groups.nunique()
        ),
        "cross_validation": (
            "5-fold GroupKFold by h3_parent_res6"
        ),
        "ridge_alpha": RIDGE_ALPHA,
        "total_training_seconds": float(
            total_training_seconds
        ),
        "median_dummy_oof": (
            dummy_overall_metrics
        ),
        "ridge_oof": (
            ridge_overall_metrics
        ),
        "ridge_mae_improvement_over_dummy_mbps": float(
            mae_improvement
        ),
        "ridge_rmse_improvement_over_dummy_mbps": float(
            rmse_improvement
        ),
    }

    with open(
        OVERALL_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall_metrics,
            file,
            indent=4,
        )

    print("\n--- Saved outputs ---")

    print(
        f"Fold metrics:\n{FOLD_METRICS_PATH}"
    )

    print(
        f"\nOverall metrics:\n"
        f"{OVERALL_METRICS_PATH}"
    )

    print(
        f"\nOut-of-fold predictions:\n"
        f"{OOF_PREDICTIONS_PATH}"
    )

    print(
        f"\nRidge coefficients:\n"
        f"{RIDGE_COEFFICIENTS_PATH}"
    )

    print("\nStatic baseline training completed.")
    print(
        "Every reported prediction was produced while "
        "its parent H3 region was excluded from training."
    )


if __name__ == "__main__":
    main()