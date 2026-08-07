from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from static_feature_registry import (
    APPROVED_STATIC_FEATURES,
    ID_COLUMN,
    LABEL_STATUS_COLUMN,
    SPATIAL_GROUP_COLUMN,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
    / "static_all_cell_predictions.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
)

RELIABILITY_PARQUET_PATH = (
    OUTPUT_DIRECTORY
    / "static_all_cell_predictions_with_reliability.parquet"
)

RELIABILITY_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "static_all_cell_predictions_with_reliability.csv"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "static_prediction_reliability_report.json"
)

COVERAGE_FOLD_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "static_coverage_bias_fold_metrics.csv"
)

COVERAGE_FEATURES_PATH = (
    OUTPUT_DIRECTORY
    / "static_coverage_bias_feature_coefficients.csv"
)


NUMBER_OF_FOLDS = 5
RANDOM_SEED = 42


def percentile_rank(
    reference_values: np.ndarray,
    query_values: np.ndarray,
) -> np.ndarray:
    """
    Calculate where each query value lies relative to a
    reference distribution.

    A value near 1 means that the distance is unusually large
    compared with the labelled-cell reference distances.
    """

    sorted_reference = np.sort(reference_values)

    ranks = np.searchsorted(
        sorted_reference,
        query_values,
        side="right",
    )

    return ranks / len(sorted_reference)


def describe_series(
    series: pd.Series,
) -> dict:
    """Convert a numerical summary into JSON-friendly values."""

    description = series.describe()

    return {
        key: float(value)
        for key, value in description.items()
    }


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static Prediction Reliability Audit")
    print("=" * 70)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "All-cell prediction file not found:\n"
            f"{PREDICTIONS_PATH}\n\n"
            "Run static_generate_all_predictions.py first."
        )

    print(f"\nLoading all-cell predictions from:\n{PREDICTIONS_PATH}")

    df = pd.read_parquet(
        PREDICTIONS_PATH
    )

    required_columns = {
        ID_COLUMN,
        SPATIAL_GROUP_COLUMN,
        LABEL_STATUS_COLUMN,
        TARGET_COLUMN,
        "honest_oof_estimate_mbps",
        "ai_ensemble_estimate_mbps",
        "ensemble_std_mbps",
        "ensemble_disagreement_level",
        *APPROVED_STATIC_FEATURES,
    }

    missing_columns = required_columns.difference(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            "Prediction file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df.reset_index(drop=True)

    labelled_mask = df[
        LABEL_STATUS_COLUMN
    ].astype(bool)

    unlabelled_mask = ~labelled_mask

    labelled_df = df[
        labelled_mask
    ].copy()

    unlabelled_df = df[
        unlabelled_mask
    ].copy()

    print("\n--- Dataset summary ---")

    print(
        "Total cells:",
        f"{len(df):,}",
    )

    print(
        "Observed Ookla cells:",
        f"{len(labelled_df):,}",
    )

    print(
        "AI-estimated cells:",
        f"{len(unlabelled_df):,}",
    )

    X = df[
        APPROVED_STATIC_FEATURES
    ].astype("float64")

    X_labelled = X.loc[
        labelled_mask
    ]

    X_unlabelled = X.loc[
        unlabelled_mask
    ]

    if not np.isfinite(
        X.to_numpy(copy=False)
    ).all():
        raise ValueError(
            "The feature matrix contains NaN or infinite values."
        )

    # ========================================================
    # PART 1: FEATURE-SPACE SUPPORT
    # ========================================================
    #
    # The scaler is fitted only on labelled cells.
    #
    # We then calculate:
    # - for each labelled cell: distance to its closest OTHER
    #   labelled cell;
    # - for each unlabelled cell: distance to its closest
    #   labelled cell.
    #
    # This tells us whether an unlabelled cell resembles the
    # population used to train the QoE model.
    # ========================================================

    print("\n--- Feature-space support analysis ---")

    scaler = StandardScaler()

    X_labelled_scaled = scaler.fit_transform(
        X_labelled
    )

    X_unlabelled_scaled = scaler.transform(
        X_unlabelled
    )

    # Two neighbours are requested for labelled cells because
    # the nearest neighbour of each labelled cell is itself.
    labelled_neighbour_model = NearestNeighbors(
        n_neighbors=2,
        metric="euclidean",
    )

    labelled_neighbour_model.fit(
        X_labelled_scaled
    )

    labelled_distances = (
        labelled_neighbour_model
        .kneighbors(
            X_labelled_scaled,
            return_distance=True,
        )[0][:, 1]
    )

    # For unlabelled cells, the nearest labelled cell is enough.
    unlabelled_neighbour_model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    unlabelled_neighbour_model.fit(
        X_labelled_scaled
    )

    unlabelled_distances = (
        unlabelled_neighbour_model
        .kneighbors(
            X_unlabelled_scaled,
            return_distance=True,
        )[0][:, 0]
    )

    support_95_threshold = float(
        np.quantile(
            labelled_distances,
            0.95,
        )
    )

    support_99_threshold = float(
        np.quantile(
            labelled_distances,
            0.99,
        )
    )

    all_support_distances = np.full(
        shape=len(df),
        fill_value=np.nan,
        dtype="float64",
    )

    all_support_distances[
        labelled_mask.to_numpy()
    ] = labelled_distances

    all_support_distances[
        unlabelled_mask.to_numpy()
    ] = unlabelled_distances

    df[
        "nearest_labelled_feature_distance"
    ] = all_support_distances

    df[
        "feature_distance_reference_percentile"
    ] = percentile_rank(
        reference_values=labelled_distances,
        query_values=all_support_distances,
    )

    df[
        "feature_space_support_level"
    ] = np.select(
        [
            df[
                "nearest_labelled_feature_distance"
            ] <= support_95_threshold,

            df[
                "nearest_labelled_feature_distance"
            ] <= support_99_threshold,
        ],
        [
            "within_reference",
            "limited_reference",
        ],
        default="outside_reference",
    )

    unlabelled_support_counts = (
        df.loc[
            unlabelled_mask,
            "feature_space_support_level",
        ]
        .value_counts()
    )

    print(
        "95th-percentile labelled distance:",
        f"{support_95_threshold:.4f}",
    )

    print(
        "99th-percentile labelled distance:",
        f"{support_99_threshold:.4f}",
    )

    print("\nUnlabelled-cell support levels:")

    print(unlabelled_support_counts)

    # ========================================================
    # PART 2: COMBINE SUPPORT WITH MODEL DISAGREEMENT
    # ========================================================

    disagreement = (
        df["ensemble_disagreement_level"]
        .astype(str)
    )

    support = (
        df["feature_space_support_level"]
        .astype(str)
    )

    higher_reliability = (
        (support == "within_reference")
        & (disagreement == "low")
    )

    lower_reliability = (
        (support == "outside_reference")
        | (disagreement == "high")
    )

    df[
        "relative_prediction_reliability"
    ] = np.select(
        [
            higher_reliability,
            lower_reliability,
        ],
        [
            "higher",
            "lower",
        ],
        default="moderate",
    )

    print("\n--- Combined relative reliability ---")

    unlabelled_reliability_counts = (
        df.loc[
            unlabelled_mask,
            "relative_prediction_reliability",
        ]
        .value_counts()
    )

    print(unlabelled_reliability_counts)

    # ========================================================
    # PART 3: OOKLA-COVERAGE BIAS DIAGNOSTIC
    # ========================================================
    #
    # We ask whether GIS features can predict which cells have
    # a real Ookla label.
    #
    # High performance here means the labelled and unlabelled
    # cells are geographically/systematically different.
    #
    # This diagnostic model is NOT the QoE model and must never
    # be used as a QoE feature.
    # ========================================================

    print("\n--- Ookla coverage-bias diagnostic ---")

    coverage_target = labelled_mask.astype("int8")

    groups = df[
        SPATIAL_GROUP_COLUMN
    ].astype(str)

    coverage_oof_probability = np.full(
        shape=len(df),
        fill_value=np.nan,
        dtype="float64",
    )

    coverage_fold_rows = []

    coverage_cv = StratifiedGroupKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        coverage_cv.split(
            X,
            coverage_target,
            groups=groups,
        ),
        start=1,
    ):
        X_train = X.iloc[
            train_indices
        ]

        y_train = coverage_target.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_validation = coverage_target.iloc[
            validation_indices
        ]

        model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        probability = model.predict_proba(
            X_validation
        )[:, 1]

        coverage_oof_probability[
            validation_indices
        ] = probability

        if y_validation.nunique() == 2:
            fold_roc_auc = float(
                roc_auc_score(
                    y_validation,
                    probability,
                )
            )

            fold_pr_auc = float(
                average_precision_score(
                    y_validation,
                    probability,
                )
            )
        else:
            fold_roc_auc = float("nan")
            fold_pr_auc = float("nan")

        coverage_fold_rows.append(
            {
                "fold": fold_number,
                "training_cells": len(
                    train_indices
                ),
                "validation_cells": len(
                    validation_indices
                ),
                "validation_labelled_rate": float(
                    y_validation.mean()
                ),
                "roc_auc": fold_roc_auc,
                "pr_auc": fold_pr_auc,
            }
        )

        print(
            f"Fold {fold_number}: "
            f"ROC-AUC={fold_roc_auc:.4f}, "
            f"PR-AUC={fold_pr_auc:.4f}"
        )

    missing_coverage_predictions = int(
        np.isnan(
            coverage_oof_probability
        ).sum()
    )

    if missing_coverage_predictions > 0:
        raise ValueError(
            "Coverage diagnostic has missing "
            "out-of-fold predictions."
        )

    coverage_roc_auc = float(
        roc_auc_score(
            coverage_target,
            coverage_oof_probability,
        )
    )

    coverage_pr_auc = float(
        average_precision_score(
            coverage_target,
            coverage_oof_probability,
        )
    )

    coverage_prevalence = float(
        coverage_target.mean()
    )

    if coverage_roc_auc >= 0.80:
        coverage_bias_interpretation = "strong"
    elif coverage_roc_auc >= 0.65:
        coverage_bias_interpretation = "moderate"
    else:
        coverage_bias_interpretation = "limited"

    print("\nOverall coverage diagnostic:")

    print(
        "Labelled-cell prevalence:",
        f"{coverage_prevalence:.4f}",
    )

    print(
        "ROC-AUC:",
        f"{coverage_roc_auc:.4f}",
    )

    print(
        "PR-AUC:",
        f"{coverage_pr_auc:.4f}",
    )

    print(
        "Coverage-selection difference:",
        coverage_bias_interpretation,
    )

    df[
        "ookla_coverage_propensity_diagnostic"
    ] = coverage_oof_probability

    # Fit one final diagnostic model on all cells only to inspect
    # standardized coefficients. These coefficients are for
    # interpretation, not evaluation.
    full_coverage_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    full_coverage_model.fit(
        X,
        coverage_target,
    )

    coverage_coefficients = (
        full_coverage_model
        .named_steps["classifier"]
        .coef_[0]
    )

    coefficient_df = pd.DataFrame(
        {
            "feature": APPROVED_STATIC_FEATURES,
            "standardized_coefficient": (
                coverage_coefficients
            ),
        }
    )

    coefficient_df[
        "absolute_coefficient"
    ] = coefficient_df[
        "standardized_coefficient"
    ].abs()

    coefficient_df = (
        coefficient_df
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nTop GIS indicators of Ookla-label coverage:")

    for position, row in coefficient_df.head(
        10
    ).iterrows():
        print(
            f"{position + 1:>2}. "
            f"{row['feature']:<42}"
            f"coefficient="
            f"{row['standardized_coefficient']:>8.4f}"
        )

    # ========================================================
    # PART 4: DISTRIBUTION COMPARISON
    # ========================================================

    print("\n--- Observed and estimated distributions ---")

    print("\nObserved Ookla download speed:")

    print(
        labelled_df[
            TARGET_COLUMN
        ].describe()
    )

    print("\nHonest OOF estimates on labelled cells:")

    print(
        labelled_df[
            "honest_oof_estimate_mbps"
        ].describe()
    )

    print("\nAI ensemble estimates on unlabelled cells:")

    print(
        unlabelled_df[
            "ai_ensemble_estimate_mbps"
        ].describe()
    )

    observed_mean = float(
        labelled_df[
            TARGET_COLUMN
        ].mean()
    )

    unlabelled_estimated_mean = float(
        unlabelled_df[
            "ai_ensemble_estimate_mbps"
        ].mean()
    )

    mean_ratio = (
        unlabelled_estimated_mean
        / observed_mean
    )

    print(
        "\nUnlabelled-estimate mean / "
        "observed-target mean:",
        f"{mean_ratio:.4f}",
    )

    # ========================================================
    # PART 5: SAVE OUTPUTS
    # ========================================================

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        RELIABILITY_PARQUET_PATH,
        index=False,
        compression="zstd",
    )

    df.to_csv(
        RELIABILITY_CSV_PATH,
        index=False,
    )

    coverage_fold_df = pd.DataFrame(
        coverage_fold_rows
    )

    coverage_fold_df.to_csv(
        COVERAGE_FOLD_METRICS_PATH,
        index=False,
    )

    coefficient_df.to_csv(
        COVERAGE_FEATURES_PATH,
        index=False,
    )

    report = {
        "purpose": (
            "Reliability audit for static spatial QoE estimates"
        ),
        "total_cells": len(df),
        "observed_ookla_cells": len(
            labelled_df
        ),
        "ai_estimated_cells": len(
            unlabelled_df
        ),
        "feature_space_support": {
            "interpretation": (
                "Distance in standardized GIS feature space "
                "to the nearest real-labelled reference cell"
            ),
            "labelled_distance_95th_percentile": (
                support_95_threshold
            ),
            "labelled_distance_99th_percentile": (
                support_99_threshold
            ),
            "unlabelled_support_counts": {
                str(key): int(value)
                for key, value in (
                    unlabelled_support_counts.items()
                )
            },
        },
        "relative_reliability": {
            "interpretation": (
                "Combines feature-space support and ensemble "
                "model disagreement; not a calibrated accuracy "
                "probability or confidence interval"
            ),
            "unlabelled_counts": {
                str(key): int(value)
                for key, value in (
                    unlabelled_reliability_counts.items()
                )
            },
        },
        "coverage_bias_diagnostic": {
            "target": (
                "has_real_ookla_label"
            ),
            "used_as_qoe_feature": False,
            "cross_validation": (
                "5-fold StratifiedGroupKFold by "
                "h3_parent_res6"
            ),
            "labelled_prevalence": (
                coverage_prevalence
            ),
            "roc_auc": coverage_roc_auc,
            "pr_auc": coverage_pr_auc,
            "interpretation": (
                coverage_bias_interpretation
            ),
        },
        "distributions": {
            "observed_ookla_dl_mbps": (
                describe_series(
                    labelled_df[
                        TARGET_COLUMN
                    ]
                )
            ),
            "honest_oof_estimate_mbps": (
                describe_series(
                    labelled_df[
                        "honest_oof_estimate_mbps"
                    ]
                )
            ),
            "unlabelled_ai_estimate_mbps": (
                describe_series(
                    unlabelled_df[
                        "ai_ensemble_estimate_mbps"
                    ]
                )
            ),
            "unlabelled_to_observed_mean_ratio": (
                mean_ratio
            ),
        },
        "usage_note": (
            "Relative reliability indicates model support, "
            "not guaranteed prediction correctness. "
            "Cells marked lower reliability should be shown "
            "with a warning or reduced visual confidence."
        ),
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n--- Saved outputs ---")

    print(
        f"Reliability-enhanced Parquet:\n"
        f"{RELIABILITY_PARQUET_PATH}"
    )

    print(
        f"\nReliability-enhanced CSV:\n"
        f"{RELIABILITY_CSV_PATH}"
    )

    print(
        f"\nReliability report:\n"
        f"{REPORT_PATH}"
    )

    print(
        f"\nCoverage fold metrics:\n"
        f"{COVERAGE_FOLD_METRICS_PATH}"
    )

    print(
        f"\nCoverage diagnostic coefficients:\n"
        f"{COVERAGE_FEATURES_PATH}"
    )

    print("\nStatic prediction reliability audit completed.")

    print(
        "\nImportant: the reliability fields describe "
        "relative model support and agreement. They are "
        "not calibrated confidence probabilities."
    )


if __name__ == "__main__":
    main()