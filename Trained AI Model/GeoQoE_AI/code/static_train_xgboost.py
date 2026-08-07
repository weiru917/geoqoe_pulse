from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold

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

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "static_xgboost"
)

BASELINE_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "static_baseline_overall_metrics.json"
)

FOLD_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_fold_metrics.csv"
)

OVERALL_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_overall_metrics.json"
)

OOF_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_oof_predictions.parquet"
)

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_feature_importance.csv"
)

RAW_FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_feature_importance_by_fold.csv"
)


NUMBER_OF_FOLDS = 5
RANDOM_SEED = 42

XGBOOST_PARAMETERS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.50,
    "reg_lambda": 5.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "importance_type": "gain",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def calculate_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict:
    """
    Calculate unweighted cell-level regression metrics.

    These are the primary metrics used to compare models.
    Every H3 cell contributes equally.
    """

    if (
        np.std(y_true) == 0
        or np.std(y_prediction) == 0
    ):
        spearman_value = float("nan")
    else:
        spearman_value = float(
            spearmanr(
                y_true,
                y_prediction,
            ).statistic
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
        "spearman_correlation": spearman_value,
    }


def make_training_weights(
    ookla_test_counts: pd.Series,
) -> tuple[np.ndarray, float]:
    """
    Create moderate confidence weights from Ookla test counts.

    1. Square root reduces the extreme difference between cells.
    2. The 95th-percentile cap prevents a few heavily measured
       cells from dominating training.
    3. Mean normalisation keeps the average weight equal to 1.
    """

    raw_weights = np.sqrt(
        np.clip(
            ookla_test_counts.to_numpy(
                dtype="float64"
            ),
            a_min=1.0,
            a_max=None,
        )
    )

    cap_value = float(
        np.quantile(
            raw_weights,
            0.95,
        )
    )

    capped_weights = np.minimum(
        raw_weights,
        cap_value,
    )

    normalised_weights = (
        capped_weights
        / capped_weights.mean()
    )

    return (
        normalised_weights.astype("float32"),
        cap_value,
    )


def print_metrics(
    model_name: str,
    metrics: dict,
) -> None:
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


def extract_gain_importance(
    model: xgb.XGBRegressor,
    fold_number: int,
    variant: str,
) -> list[dict]:
    """
    Extract gain-based feature importance from one fitted model.
    Features unused by a tree ensemble receive importance zero.
    """

    booster = model.get_booster()

    raw_importance = booster.get_score(
        importance_type="gain"
    )

    rows = []

    for feature in APPROVED_STATIC_FEATURES:
        rows.append(
            {
                "fold": fold_number,
                "variant": variant,
                "feature": feature,
                "gain_importance": float(
                    raw_importance.get(
                        feature,
                        0.0,
                    )
                ),
            }
        )

    return rows


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static XGBoost Spatial Cross-Validation")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Static dataset not found:\n{DATA_PATH}"
        )

    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Baseline metrics were not found. Run "
            "static_train_baselines.py first."
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

    print(f"\nLoading real-labelled data from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    labelled_df = df[
        df[LABEL_STATUS_COLUMN].astype(bool)
    ].copy()

    labelled_df = labelled_df.reset_index(
        drop=True
    )

    X = labelled_df[
        APPROVED_STATIC_FEATURES
    ].astype("float32")

    y = labelled_df[
        TARGET_COLUMN
    ].astype("float32")

    groups = labelled_df[
        SPATIAL_GROUP_COLUMN
    ].astype(str)

    test_counts = labelled_df[
        SAMPLE_WEIGHT_COLUMN
    ].astype("float32")

    print("\n--- Dataset summary ---")

    print(
        "Real-labelled cells:",
        f"{len(labelled_df):,}",
    )

    print(
        "Input features:",
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

    if not np.isfinite(
        X.to_numpy(copy=False)
    ).all():
        raise ValueError(
            "The feature matrix contains non-finite values."
        )

    if not np.isfinite(
        y.to_numpy(copy=False)
    ).all():
        raise ValueError(
            "The target contains non-finite values."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    variants = [
        "unweighted",
        "confidence_weighted",
    ]

    oof_predictions = {
        variant: np.full(
            shape=len(labelled_df),
            fill_value=np.nan,
            dtype="float64",
        )
        for variant in variants
    }

    fold_assignment = np.full(
        shape=len(labelled_df),
        fill_value=-1,
        dtype="int8",
    )

    fold_metric_rows = []
    feature_importance_rows = []

    group_kfold = GroupKFold(
        n_splits=NUMBER_OF_FOLDS
    )

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

        overlap = train_groups.intersection(
            validation_groups
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
            "Training groups:",
            len(train_groups),
        )

        print(
            "Validation groups:",
            len(validation_groups),
        )

        print(
            "Overlapping groups:",
            len(overlap),
        )

        if overlap:
            raise ValueError(
                f"Spatial leakage detected in fold {fold_number}."
            )

        training_weights, weight_cap = (
            make_training_weights(
                test_counts.iloc[
                    train_indices
                ]
            )
        )

        print(
            "Confidence-weight square-root cap:",
            f"{weight_cap:.4f}",
        )

        fold_assignment[
            validation_indices
        ] = fold_number

        for variant in variants:
            print(f"\nTraining variant: {variant}")

            model = xgb.XGBRegressor(
                **XGBOOST_PARAMETERS
            )

            start_time = time.perf_counter()

            if variant == "confidence_weighted":
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=training_weights,
                    verbose=False,
                )
            else:
                model.fit(
                    X_train,
                    y_train,
                    verbose=False,
                )

            training_seconds = (
                time.perf_counter()
                - start_time
            )

            prediction = model.predict(
                X_validation
            )

            # Download speed cannot be negative.
            prediction = np.clip(
                prediction,
                a_min=0.0,
                a_max=None,
            )

            oof_predictions[
                variant
            ][validation_indices] = prediction

            metrics = calculate_metrics(
                y_true=y_validation.to_numpy(),
                y_prediction=prediction,
            )

            print_metrics(
                f"XGBoost — {variant}",
                metrics,
            )

            print(
                "Training time:",
                f"{training_seconds:.4f} seconds",
            )

            fold_metric_rows.append(
                {
                    "fold": fold_number,
                    "variant": variant,
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
                    "weight_cap_sqrt_tests": (
                        weight_cap
                        if variant
                        == "confidence_weighted"
                        else np.nan
                    ),
                    "training_seconds": (
                        training_seconds
                    ),
                    **metrics,
                }
            )

            feature_importance_rows.extend(
                extract_gain_importance(
                    model=model,
                    fold_number=fold_number,
                    variant=variant,
                )
            )

            model_path = (
                MODEL_DIRECTORY
                / f"{variant}_fold_{fold_number}.json"
            )

            model.save_model(
                str(model_path)
            )

            print(
                "Saved model:",
                model_path.name,
            )

    total_training_seconds = (
        time.perf_counter()
        - total_start_time
    )

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RECONCILIATION")
    print("=" * 70)

    for variant in variants:
        missing_count = int(
            np.isnan(
                oof_predictions[variant]
            ).sum()
        )

        print(
            f"Missing {variant} predictions:",
            missing_count,
        )

        if missing_count > 0:
            raise ValueError(
                f"{variant} has missing OOF predictions."
            )

    unassigned_rows = int(
        (fold_assignment == -1).sum()
    )

    print(
        "Rows without fold assignment:",
        unassigned_rows,
    )

    if unassigned_rows > 0:
        raise ValueError(
            "Some rows did not receive a fold assignment."
        )

    y_array = y.to_numpy()

    overall_results = {}

    print("\n--- Overall out-of-fold results ---")

    for variant in variants:
        metrics = calculate_metrics(
            y_true=y_array,
            y_prediction=oof_predictions[
                variant
            ],
        )

        overall_results[variant] = metrics

        print_metrics(
            f"XGBoost — {variant}",
            metrics,
        )

    # --------------------------------------------------------
    # Compare against previously saved baselines.
    # --------------------------------------------------------

    with open(
        BASELINE_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        baseline_metrics = json.load(file)

    dummy_metrics = baseline_metrics[
        "median_dummy_oof"
    ]

    ridge_metrics = baseline_metrics[
        "ridge_oof"
    ]

    print("\n--- Comparison with baselines ---")

    print(
        "Dummy MAE:",
        f"{dummy_metrics['mae_mbps']:.4f} Mbps",
    )

    print(
        "Ridge MAE:",
        f"{ridge_metrics['mae_mbps']:.4f} Mbps",
    )

    for variant in variants:
        variant_mae = overall_results[
            variant
        ]["mae_mbps"]

        print(
            f"{variant} XGBoost MAE:",
            f"{variant_mae:.4f} Mbps",
        )

        print(
            "  Improvement over Ridge:",
            f"{ridge_metrics['mae_mbps'] - variant_mae:.4f} Mbps",
        )

    candidate_mae = {
        "median_dummy": float(
            dummy_metrics["mae_mbps"]
        ),
        "ridge": float(
            ridge_metrics["mae_mbps"]
        ),
        "xgboost_unweighted": float(
            overall_results[
                "unweighted"
            ]["mae_mbps"]
        ),
        "xgboost_confidence_weighted": float(
            overall_results[
                "confidence_weighted"
            ]["mae_mbps"]
        ),
    }

    provisional_winner = min(
        candidate_mae,
        key=candidate_mae.get,
    )

    print(
        "\nLowest OOF MAE model:",
        provisional_winner,
    )

    print(
        "Lowest OOF MAE:",
        f"{candidate_mae[provisional_winner]:.4f} Mbps",
    )

    # --------------------------------------------------------
    # Save fold metrics.
    # --------------------------------------------------------

    fold_metrics_df = pd.DataFrame(
        fold_metric_rows
    )

    fold_metrics_df.to_csv(
        FOLD_METRICS_PATH,
        index=False,
    )

    print("\n--- Mean and variation across folds ---")

    fold_summary = (
        fold_metrics_df
        .groupby("variant")
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
    # Save OOF predictions.
    # --------------------------------------------------------

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

    for variant in variants:
        prediction_column = (
            f"xgboost_{variant}_prediction_mbps"
        )

        error_column = (
            f"xgboost_{variant}_absolute_error_mbps"
        )

        oof_df[
            prediction_column
        ] = oof_predictions[variant]

        oof_df[
            error_column
        ] = np.abs(
            oof_df[TARGET_COLUMN]
            - oof_df[prediction_column]
        )

    oof_df.to_parquet(
        OOF_PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    # --------------------------------------------------------
    # Save feature importance.
    # --------------------------------------------------------

    raw_importance_df = pd.DataFrame(
        feature_importance_rows
    )

    raw_importance_df.to_csv(
        RAW_FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    importance_summary = (
        raw_importance_df
        .groupby(
            [
                "variant",
                "feature",
            ],
            as_index=False,
        )
        .agg(
            mean_gain_importance=(
                "gain_importance",
                "mean",
            ),
            std_gain_importance=(
                "gain_importance",
                "std",
            ),
        )
    )

    importance_summary[
        "gain_fraction_within_variant"
    ] = (
        importance_summary[
            "mean_gain_importance"
        ]
        / importance_summary
        .groupby("variant")[
            "mean_gain_importance"
        ]
        .transform("sum")
    )

    importance_summary = (
        importance_summary
        .sort_values(
            [
                "variant",
                "gain_fraction_within_variant",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    importance_summary.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save overall metadata and metrics.
    # --------------------------------------------------------

    output_metrics = {
        "dataset": (
            "cell_static_qoe_baseline.parquet"
        ),
        "data_status": (
            "real quarterly Ookla targets"
        ),
        "target": TARGET_COLUMN,
        "number_of_labelled_cells": len(
            labelled_df
        ),
        "number_of_spatial_groups": int(
            groups.nunique()
        ),
        "number_of_features": len(
            APPROVED_STATIC_FEATURES
        ),
        "features": APPROVED_STATIC_FEATURES,
        "cross_validation": (
            "5-fold GroupKFold by h3_parent_res6"
        ),
        "xgboost_parameters": (
            XGBOOST_PARAMETERS
        ),
        "confidence_weighting": {
            "used_as_feature": False,
            "weight_formula": (
                "sqrt(ookla_tests), capped at the "
                "training-fold 95th percentile and "
                "normalised to mean 1"
            ),
        },
        "baseline_results": {
            "median_dummy": dummy_metrics,
            "ridge": ridge_metrics,
        },
        "xgboost_results": overall_results,
        "candidate_oof_mae_mbps": (
            candidate_mae
        ),
        "provisional_lowest_mae_model": (
            provisional_winner
        ),
        "total_training_seconds": float(
            total_training_seconds
        ),
    }

    with open(
        OVERALL_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_metrics,
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
        f"\nOOF predictions:\n"
        f"{OOF_PREDICTIONS_PATH}"
    )

    print(
        f"\nFeature importance:\n"
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    print(
        f"\nFold models:\n"
        f"{MODEL_DIRECTORY}"
    )

    print(
        "\nStatic XGBoost cross-validation completed."
    )

    print(
        "Every reported prediction was produced while "
        "its parent-H3 spatial region was excluded from training."
    )


if __name__ == "__main__":
    main()