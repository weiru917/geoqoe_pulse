from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

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

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "static_xgboost"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
)

OOF_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_oof_predictions.parquet"
)

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIRECTORY
    / "static_xgboost_feature_importance.csv"
)

ALL_CELL_PARQUET_PATH = (
    OUTPUT_DIRECTORY
    / "static_all_cell_predictions.parquet"
)

ALL_CELL_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "static_all_cell_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "static_prediction_summary.json"
)


EXPECTED_MODEL_COUNT = 5

SELECTED_VARIANT = "confidence_weighted"

OOF_PREDICTION_COLUMN = (
    "xgboost_confidence_weighted_prediction_mbps"
)

OOF_ERROR_COLUMN = (
    "xgboost_confidence_weighted_absolute_error_mbps"
)


def calculate_oof_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict:
    """
    Recalculate the honest out-of-fold metrics for the
    selected model.

    These predictions were generated when each cell's
    parent H3 region was excluded from model training.
    """

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
    }


def assign_disagreement_level(
    standard_deviation: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.Series:
    """
    Convert ensemble standard deviation into three relative
    disagreement categories.

    These categories describe agreement among the five models.
    They are not calibrated probabilities or confidence intervals.
    """

    conditions = [
        standard_deviation <= lower_threshold,
        standard_deviation <= upper_threshold,
    ]

    choices = [
        "low",
        "moderate",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="high",
        ),
        index=standard_deviation.index,
        dtype="string",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static All-Cell Prediction Generation")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Static dataset not found:\n{DATA_PATH}"
        )

    if not OOF_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "XGBoost out-of-fold predictions were not found.\n"
            "Run static_train_xgboost.py first."
        )

    required_columns = [
        ID_COLUMN,
        SPATIAL_GROUP_COLUMN,
        LABEL_STATUS_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        TARGET_COLUMN,
        "ookla_rtt_ms",
        "qoe_risk_label",
        *APPROVED_STATIC_FEATURES,
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    print(f"\nLoading all grid cells from:\n{DATA_PATH}")

    all_cells_df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    all_cells_df = all_cells_df.reset_index(
        drop=True
    )

    X_all = all_cells_df[
        APPROVED_STATIC_FEATURES
    ].astype("float32")

    print("\n--- Dataset summary ---")

    print(
        "All grid cells:",
        f"{len(all_cells_df):,}",
    )

    labelled_count = int(
        all_cells_df[
            LABEL_STATUS_COLUMN
        ].astype(bool).sum()
    )

    unlabelled_count = (
        len(all_cells_df) - labelled_count
    )

    print(
        "Real Ookla-labelled cells:",
        f"{labelled_count:,}",
    )

    print(
        "Cells requiring AI estimation:",
        f"{unlabelled_count:,}",
    )

    print(
        "Input features:",
        X_all.shape[1],
    )

    if not np.isfinite(
        X_all.to_numpy(copy=False)
    ).all():
        raise ValueError(
            "The all-cell feature matrix contains "
            "NaN or infinite values."
        )

    # --------------------------------------------------------
    # Locate the five selected fold models.
    # --------------------------------------------------------

    model_paths = sorted(
        MODEL_DIRECTORY.glob(
            f"{SELECTED_VARIANT}_fold_*.json"
        )
    )

    print("\n--- Saved model check ---")

    print(
        "Selected model variant:",
        SELECTED_VARIANT,
    )

    print(
        "Fold models found:",
        len(model_paths),
    )

    for model_path in model_paths:
        print(f"  - {model_path.name}")

    if len(model_paths) != EXPECTED_MODEL_COUNT:
        raise FileNotFoundError(
            f"Expected {EXPECTED_MODEL_COUNT} "
            f"{SELECTED_VARIANT} fold models, "
            f"but found {len(model_paths)}."
        )

    # --------------------------------------------------------
    # Generate predictions from every fold model.
    # --------------------------------------------------------

    prediction_columns = []

    print("\n--- Generating fold-model predictions ---")

    for fold_number, model_path in enumerate(
        model_paths,
        start=1,
    ):
        print(
            f"Loading and predicting with fold model "
            f"{fold_number}..."
        )

        model = xgb.XGBRegressor()

        model.load_model(
            str(model_path)
        )

        prediction = model.predict(
            X_all
        )

        prediction = np.clip(
            prediction,
            a_min=0.0,
            a_max=None,
        )

        prediction_column = (
            f"fold_{fold_number}_prediction_mbps"
        )

        all_cells_df[
            prediction_column
        ] = prediction

        prediction_columns.append(
            prediction_column
        )

    print("All five fold-model predictions generated.")

    # --------------------------------------------------------
    # Build the ensemble estimate.
    # --------------------------------------------------------

    prediction_matrix = all_cells_df[
        prediction_columns
    ].to_numpy(
        dtype="float64"
    )

    all_cells_df[
        "ai_ensemble_estimate_mbps"
    ] = prediction_matrix.mean(
        axis=1
    )

    all_cells_df[
        "ensemble_std_mbps"
    ] = prediction_matrix.std(
        axis=1,
        ddof=0,
    )

    all_cells_df[
        "ensemble_min_mbps"
    ] = prediction_matrix.min(
        axis=1
    )

    all_cells_df[
        "ensemble_max_mbps"
    ] = prediction_matrix.max(
        axis=1
    )

    all_cells_df[
        "ensemble_range_mbps"
    ] = (
        all_cells_df["ensemble_max_mbps"]
        - all_cells_df["ensemble_min_mbps"]
    )

    # Use the unlabelled-cell disagreement distribution
    # to define relative low/moderate/high categories.
    unlabelled_mask = ~all_cells_df[
        LABEL_STATUS_COLUMN
    ].astype(bool)

    unlabelled_std = all_cells_df.loc[
        unlabelled_mask,
        "ensemble_std_mbps",
    ]

    lower_disagreement_threshold = float(
        unlabelled_std.quantile(1 / 3)
    )

    upper_disagreement_threshold = float(
        unlabelled_std.quantile(2 / 3)
    )

    all_cells_df[
        "ensemble_disagreement_level"
    ] = assign_disagreement_level(
        standard_deviation=all_cells_df[
            "ensemble_std_mbps"
        ],
        lower_threshold=(
            lower_disagreement_threshold
        ),
        upper_threshold=(
            upper_disagreement_threshold
        ),
    )

    # --------------------------------------------------------
    # Add honest OOF predictions for labelled cells.
    # --------------------------------------------------------

    print("\nLoading honest out-of-fold predictions...")

    oof_df = pd.read_parquet(
        OOF_PREDICTIONS_PATH
    )

    required_oof_columns = {
        ID_COLUMN,
        OOF_PREDICTION_COLUMN,
        OOF_ERROR_COLUMN,
        "spatial_cv_fold",
    }

    missing_oof_columns = (
        required_oof_columns
        .difference(oof_df.columns)
    )

    if missing_oof_columns:
        raise ValueError(
            "The OOF file is missing columns: "
            f"{sorted(missing_oof_columns)}"
        )

    oof_subset = oof_df[
        [
            ID_COLUMN,
            "spatial_cv_fold",
            OOF_PREDICTION_COLUMN,
            OOF_ERROR_COLUMN,
        ]
    ].copy()

    all_cells_df = all_cells_df.merge(
        oof_subset,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    all_cells_df = all_cells_df.rename(
        columns={
            OOF_PREDICTION_COLUMN: (
                "honest_oof_estimate_mbps"
            ),
            OOF_ERROR_COLUMN: (
                "honest_oof_absolute_error_mbps"
            ),
        }
    )

    # --------------------------------------------------------
    # Clearly identify observed and estimated cells.
    # --------------------------------------------------------

    labelled_mask = all_cells_df[
        LABEL_STATUS_COLUMN
    ].astype(bool)

    all_cells_df[
        "static_data_status"
    ] = np.where(
        labelled_mask,
        "observed_ookla",
        "ai_estimated",
    )

    # This field is designed for display:
    # - observed Ookla value when available;
    # - otherwise the AI ensemble estimate.
    all_cells_df[
        "display_dl_mbps"
    ] = np.where(
        labelled_mask,
        all_cells_df[TARGET_COLUMN],
        all_cells_df[
            "ai_ensemble_estimate_mbps"
        ],
    )

    all_cells_df[
        "model_variant"
    ] = (
        "confidence_weighted_xgboost_"
        "five_fold_spatial_ensemble"
    )

    all_cells_df[
        "prediction_scope"
    ] = (
        "quarterly_static_spatial_qoe"
    )

    # --------------------------------------------------------
    # Verify labelled and unlabelled handling.
    # --------------------------------------------------------

    labelled_output = all_cells_df[
        labelled_mask
    ].copy()

    unlabelled_output = all_cells_df[
        ~labelled_mask
    ].copy()

    labelled_missing_oof = int(
        labelled_output[
            "honest_oof_estimate_mbps"
        ].isna().sum()
    )

    unlabelled_with_oof = int(
        unlabelled_output[
            "honest_oof_estimate_mbps"
        ].notna().sum()
    )

    print("\n--- Output integrity check ---")

    print(
        "Labelled cells missing honest OOF predictions:",
        labelled_missing_oof,
    )

    print(
        "Unlabelled cells incorrectly containing OOF predictions:",
        unlabelled_with_oof,
    )

    if labelled_missing_oof > 0:
        raise ValueError(
            "Some real-labelled cells are missing "
            "their honest OOF estimates."
        )

    if unlabelled_with_oof > 0:
        raise ValueError(
            "Unlabelled cells should not contain "
            "out-of-fold evaluation predictions."
        )

    # --------------------------------------------------------
    # Recalculate OOF metrics for confirmation.
    # --------------------------------------------------------

    oof_metrics = calculate_oof_metrics(
        y_true=labelled_output[
            TARGET_COLUMN
        ].to_numpy(
            dtype="float64"
        ),
        y_prediction=labelled_output[
            "honest_oof_estimate_mbps"
        ].to_numpy(
            dtype="float64"
        ),
    )

    print("\n--- Confirmed honest OOF metrics ---")

    print(
        "MAE:",
        f"{oof_metrics['mae_mbps']:.4f} Mbps",
    )

    print(
        "RMSE:",
        f"{oof_metrics['rmse_mbps']:.4f} Mbps",
    )

    print(
        "R²:",
        f"{oof_metrics['r2']:.4f}",
    )

    # --------------------------------------------------------
    # Summarise unlabelled-cell estimates.
    # --------------------------------------------------------

    print("\n--- AI-estimated unlabelled cells ---")

    print(
        "Estimated cells:",
        f"{len(unlabelled_output):,}",
    )

    print("\nEstimated download-speed summary:")

    print(
        unlabelled_output[
            "ai_ensemble_estimate_mbps"
        ].describe()
    )

    print("\nEnsemble-disagreement summary:")

    print(
        unlabelled_output[
            "ensemble_std_mbps"
        ].describe()
    )

    print("\nRelative disagreement levels:")

    print(
        unlabelled_output[
            "ensemble_disagreement_level"
        ].value_counts()
    )

    print(
        "\nLow/moderate boundary:",
        f"{lower_disagreement_threshold:.4f} Mbps",
    )

    print(
        "Moderate/high boundary:",
        f"{upper_disagreement_threshold:.4f} Mbps",
    )

    # --------------------------------------------------------
    # Display feature importance for selected variant.
    # --------------------------------------------------------

    print("\n--- Selected-model feature importance ---")

    top_feature_rows = []

    if FEATURE_IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        )

        selected_importance = (
            importance_df[
                importance_df["variant"]
                == SELECTED_VARIANT
            ]
            .sort_values(
                "gain_fraction_within_variant",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        top_feature_rows = (
            selected_importance
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        for position, row in enumerate(
            top_feature_rows,
            start=1,
        ):
            print(
                f"{position:>2}. "
                f"{row['feature']:<42}"
                f"{row['gain_fraction_within_variant'] * 100:>8.3f}%"
            )
    else:
        print(
            "Feature-importance file not found; "
            "this display was skipped."
        )

    # --------------------------------------------------------
    # Save output files.
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_cells_df.to_parquet(
        ALL_CELL_PARQUET_PATH,
        index=False,
        compression="zstd",
    )

    # CSV is convenient for teammates and dashboard inspection.
    all_cells_df.to_csv(
        ALL_CELL_CSV_PATH,
        index=False,
    )

    summary = {
        "dataset": (
            "cell_static_qoe_baseline.parquet"
        ),
        "data_status": (
            "real quarterly Ookla targets with "
            "AI estimates for unlabelled cells"
        ),
        "selected_model": (
            "confidence_weighted XGBoost"
        ),
        "ensemble": (
            "mean of five spatial GroupKFold models"
        ),
        "model_feature_count": len(
            APPROVED_STATIC_FEATURES
        ),
        "model_features": (
            APPROVED_STATIC_FEATURES
        ),
        "total_cells": len(
            all_cells_df
        ),
        "observed_ookla_cells": len(
            labelled_output
        ),
        "ai_estimated_cells": len(
            unlabelled_output
        ),
        "honest_oof_metrics": (
            oof_metrics
        ),
        "ensemble_disagreement": {
            "interpretation": (
                "relative disagreement among five fold "
                "models; not a calibrated confidence interval"
            ),
            "low_moderate_boundary_mbps": (
                lower_disagreement_threshold
            ),
            "moderate_high_boundary_mbps": (
                upper_disagreement_threshold
            ),
            "unlabelled_mean_std_mbps": float(
                unlabelled_output[
                    "ensemble_std_mbps"
                ].mean()
            ),
            "unlabelled_median_std_mbps": float(
                unlabelled_output[
                    "ensemble_std_mbps"
                ].median()
            ),
        },
        "top_features": top_feature_rows,
        "important_usage_note": (
            "Use honest_oof_estimate_mbps for model "
            "evaluation on observed cells. Use "
            "ai_ensemble_estimate_mbps for unlabelled "
            "cell estimation. Do not evaluate using the "
            "all-model ensemble predictions on labelled cells."
        ),
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n--- Saved outputs ---")

    print(
        f"All-cell Parquet:\n"
        f"{ALL_CELL_PARQUET_PATH}"
    )

    print(
        f"\nAll-cell CSV:\n"
        f"{ALL_CELL_CSV_PATH}"
    )

    print(
        f"\nPrediction summary:\n"
        f"{SUMMARY_PATH}"
    )

    print("\nStatic all-cell prediction generation completed.")

    print(
        "\nImportant: ensemble_std_mbps describes "
        "model disagreement, not a guaranteed prediction "
        "error or statistical confidence interval."
    )


if __name__ == "__main__":
    main()