from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from static_feature_registry import (
    APPROVED_STATIC_FEATURES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
    / "cell_time_dataset.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
)

FEATURE_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "onset_feature_audit.csv"
)

REGISTRY_PATH = (
    OUTPUT_DIRECTORY
    / "onset_feature_registry.json"
)


# ============================================================
# CONTROL AND TARGET COLUMNS
# ============================================================

ID_COLUMN = "h3_index"
SPATIAL_GROUP_COLUMN = "h3_parent_res6"
TIMESTAMP_COLUMN = "timestamp"

CURRENT_STATE_COLUMN = "target_is_degraded"
NEXT_STATE_COLUMN = "target_is_degraded_next_1h"

ANCHOR_COLUMN = "qoe_level_is_ookla_anchored"
WARMUP_COLUMN = "in_warmup_window"

TARGET_COLUMN = "onset_target"


CONTROL_COLUMNS = [
    ID_COLUMN,
    SPATIAL_GROUP_COLUMN,
    TIMESTAMP_COLUMN,
    WARMUP_COLUMN,
    ANCHOR_COLUMN,
    CURRENT_STATE_COLUMN,
    NEXT_STATE_COLUMN,
]


# ============================================================
# FEATURE SET 1: TIME-ONLY BASELINE
# ============================================================

TIME_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_peak_period",
    "ridership_daily_surge_factor",
]


# ============================================================
# FEATURE SET 2: CURRENT AND HISTORICAL QOE
# ============================================================

CURRENT_QOE_FEATURES = [
    "dl_median",
    "ul_median",
    "rtt_median",
    "jitter_median",
    "loss_median",
]


HISTORICAL_QOE_FEATURES = [
    "lag_1h_dl_median",
    "lag_2h_dl_median",
    "lag_24h_dl_median",
    "rolling_mean_24h_dl_median",
    "rolling_std_24h_dl_median",

    "lag_1h_rtt_median",
    "lag_2h_rtt_median",
    "lag_24h_rtt_median",
    "rolling_mean_24h_rtt_median",
    "rolling_std_24h_rtt_median",
]


QOE_HISTORY_FEATURES = (
    CURRENT_QOE_FEATURES
    + HISTORICAL_QOE_FEATURES
)


# ============================================================
# FEATURE SET 3: FULL CAUSAL MODEL
# ============================================================

# Reuse the same 16 static GIS features already audited for
# the real static model.
STATIC_GIS_FEATURES = list(
    APPROVED_STATIC_FEATURES
)


DEMAND_CONTEXT_FEATURES = [
    # Composite of real static demand-related GIS variables.
    # It has no historical-QoE component.
    "demand_pressure_base",
]


SPATIAL_QOE_FEATURES = [
    # Same-hour neighbouring conditions are available at
    # predictor time t and are used to predict onset at t+1.
    "spatial_lag_dl_median",
    "spatial_var_dl_median",
    "spatial_lag_rtt_median",
    "spatial_var_rtt_median",
]


def unique_preserving_order(
    values: list[str],
) -> list[str]:
    """Remove duplicate names while preserving order."""

    return list(
        dict.fromkeys(values)
    )


TIME_ONLY_FEATURES = unique_preserving_order(
    TIME_FEATURES
)

QOE_HISTORY_ONLY_FEATURES = (
    unique_preserving_order(
        QOE_HISTORY_FEATURES
    )
)

FULL_CAUSAL_FEATURES = (
    unique_preserving_order(
        TIME_FEATURES
        + QOE_HISTORY_FEATURES
        + STATIC_GIS_FEATURES
        + DEMAND_CONTEXT_FEATURES
        + SPATIAL_QOE_FEATURES
    )
)


MODEL_FEATURE_SETS = {
    "time_only": TIME_ONLY_FEATURES,
    "qoe_history_only": QOE_HISTORY_ONLY_FEATURES,
    "full_causal": FULL_CAUSAL_FEATURES,
}


# ============================================================
# EXCLUDED COLUMNS
# ============================================================

EXCLUDED_SHORTCUT_OR_REDUNDANT_COLUMNS = {
    # Simulated measurement-volume variables.
    "test_sample_count": (
        "Simulated measurement volume; omitted from the "
        "primary model to reduce scenario-specific shortcuts."
    ),
    "unique_device_count": (
        "Approximately 70% of test_sample_count and therefore "
        "not an independent feature."
    ),
    "qoe_confidence_score": (
        "Deterministic transformation of test_sample_count."
    ),

    # Formula-derived duplicates.
    "dl_mean": "Formula-derived from dl_median.",
    "dl_std": "Formula-derived from dl_median.",
    "dl_p95": "Formula-derived from dl_median.",
    "ul_mean": "Formula-derived from ul_median.",
    "ul_std": "Formula-derived from ul_median.",
    "rtt_mean": "Formula-derived from rtt_median.",
    "rtt_std": "Formula-derived from rtt_median.",

    # Circular demand indices.
    "demand_pressure_index": (
        "Contains a historical-QoE component."
    ),
    "demand_pressure_index_pre_spatial": (
        "Contains a historical-QoE component."
    ),
    "spatial_lag_demand_pressure_index_pre_spatial": (
        "Derived from a circular demand-pressure index."
    ),
    "spatial_var_demand_pressure_index_pre_spatial": (
        "Derived from a circular demand-pressure index."
    ),

    # Undocumented fields.
    "qoe_trend_1h": (
        "Definition is undocumented in the feature dictionary."
    ),
    "spatial_rolling_mean_24h_dl_median": (
        "Definition is undocumented in the feature dictionary."
    ),

    # Highly engineered same-hour spatial statistics.
    "gi_star_zscore_rtt_median": (
        "Omitted from the first model to reduce redundant "
        "same-hour RTT transformations."
    ),
    "gi_star_pvalue_rtt_median": (
        "Omitted from the first model."
    ),
    "local_moran_i_rtt_median": (
        "Omitted from the first model."
    ),
    "local_moran_pvalue_rtt_median": (
        "Omitted from the first model."
    ),
    "historical_hotspot_freq_rtt_median": (
        "Omitted from the first model for simplicity."
    ),
    "hotspot_category_rtt_median_encoded": (
        "Omitted from the first model."
    ),
    "local_moran_quadrant_rtt_median_encoded": (
        "Omitted from the first model."
    ),
}


FORBIDDEN_MODEL_FEATURES = {
    *CONTROL_COLUMNS,
    TARGET_COLUMN,

    "row_id",
    "priority_base_score",
    "recommended_action",
    "likely_cause",

    "qoe_baseline_source",
    "tower_is_synthetic",
    "building_is_synthetic",
    "osm_is_synthetic",
    "population_is_synthetic",

    "ookla_quarter_test_count",
    "ookla_quarter_device_count",
}


TRAIN_TIMESTAMP_COUNT = 100
VALIDATION_TIMESTAMP_COUNT = 21
EXPECTED_TOTAL_TIMESTAMP_COUNT = 143


def assign_temporal_split(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Assign the existing chronological split.

    Train:
        First 100 predictor timestamps.

    Validation:
        Next 21 predictor timestamps.

    Test:
        Final 22 predictor timestamps.
    """

    unique_timestamps = pd.Index(
        sorted(
            df[TIMESTAMP_COLUMN].unique()
        )
    )

    if (
        len(unique_timestamps)
        != EXPECTED_TOTAL_TIMESTAMP_COUNT
    ):
        raise ValueError(
            "Expected 143 usable timestamps, but found "
            f"{len(unique_timestamps)}."
        )

    validation_end = (
        TRAIN_TIMESTAMP_COUNT
        + VALIDATION_TIMESTAMP_COUNT
    )

    train_times = set(
        unique_timestamps[
            :TRAIN_TIMESTAMP_COUNT
        ]
    )

    validation_times = set(
        unique_timestamps[
            TRAIN_TIMESTAMP_COUNT:
            validation_end
        ]
    )

    test_times = set(
        unique_timestamps[
            validation_end:
        ]
    )

    split = pd.Series(
        index=df.index,
        dtype="string",
    )

    split.loc[
        df[TIMESTAMP_COLUMN].isin(
            train_times
        )
    ] = "train"

    split.loc[
        df[TIMESTAMP_COLUMN].isin(
            validation_times
        )
    ] = "validation"

    split.loc[
        df[TIMESTAMP_COLUMN].isin(
            test_times
        )
    ] = "test"

    if split.isna().any():
        raise ValueError(
            "Some rows did not receive a temporal split."
        )

    return split


def is_numeric_or_boolean(
    series: pd.Series,
) -> bool:
    """Check whether a feature is usable as a numeric input."""

    return bool(
        pd.api.types.is_numeric_dtype(
            series.dtype
        )
        or pd.api.types.is_bool_dtype(
            series.dtype
        )
    )


def values_are_finite(
    series: pd.Series,
) -> bool:
    """Check all values for NaN or infinity."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype="float64"
    )

    return bool(
        np.isfinite(values).all()
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Onset Feature Registry Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found:\n{DATA_PATH}"
        )

    # --------------------------------------------------------
    # 1. Validate all selected feature names against schema.
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(
        DATA_PATH
    )

    parquet_columns = set(
        parquet_file.schema_arrow.names
    )

    selected_features = (
        unique_preserving_order(
            TIME_ONLY_FEATURES
            + QOE_HISTORY_ONLY_FEATURES
            + FULL_CAUSAL_FEATURES
        )
    )

    columns_to_load = (
        unique_preserving_order(
            CONTROL_COLUMNS
            + selected_features
        )
    )

    missing_columns = [
        column
        for column in columns_to_load
        if column not in parquet_columns
    ]

    print("\n--- Feature-set sizes ---")

    for set_name, features in (
        MODEL_FEATURE_SETS.items()
    ):
        print(
            f"{set_name:<20}"
            f"{len(features):>3} features"
        )

    print(
        "\nUnique selected source features:",
        len(selected_features),
    )

    if missing_columns:
        print("\nMissing selected columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "One or more selected features are absent."
        )

    print(
        "All selected features are present in the Parquet file."
    )

    # --------------------------------------------------------
    # 2. Check forbidden overlap.
    # --------------------------------------------------------

    forbidden_overlap = sorted(
        set(selected_features)
        .intersection(
            FORBIDDEN_MODEL_FEATURES
        )
    )

    print("\n--- Leakage and control-column check ---")

    if forbidden_overlap:
        for column in forbidden_overlap:
            print(f"  - {column}")

        raise ValueError(
            "Forbidden columns are present in the registry."
        )

    print(
        "No target, identifier, provenance or control column "
        "is included as a model feature."
    )

    # --------------------------------------------------------
    # 3. Load only selected and control columns.
    # --------------------------------------------------------

    print(
        f"\nLoading selected hourly columns from:\n{DATA_PATH}"
    )

    df = pd.read_parquet(
        DATA_PATH,
        columns=columns_to_load,
    )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="raise",
    )

    current_state = pd.to_numeric(
        df[CURRENT_STATE_COLUMN],
        errors="coerce",
    )

    next_state = pd.to_numeric(
        df[NEXT_STATE_COLUMN],
        errors="coerce",
    )

    onset_candidate_mask = (
        next_state.notna()
        & ~df[WARMUP_COLUMN].astype(bool)
        & df[ANCHOR_COLUMN].astype(bool)
        & (current_state == 0)
    )

    onset_df = df[
        onset_candidate_mask
    ].copy()

    onset_df[TARGET_COLUMN] = (
        pd.to_numeric(
            onset_df[NEXT_STATE_COLUMN],
            errors="raise",
        )
        .astype("int8")
    )

    onset_df[
        "temporal_split"
    ] = assign_temporal_split(
        onset_df
    )

    print("\n--- Onset dataset summary ---")

    print(
        "Candidate rows:",
        f"{len(onset_df):,}",
    )

    print(
        "Positive onsets:",
        f"{int(onset_df[TARGET_COLUMN].sum()):,}",
    )

    print(
        "Unique timestamps:",
        onset_df[TIMESTAMP_COLUMN].nunique(),
    )

    print(
        "Unique anchored cells:",
        f"{onset_df[ID_COLUMN].nunique():,}",
    )

    # --------------------------------------------------------
    # 4. Split-level reconciliation.
    # --------------------------------------------------------

    print("\n--- Locked temporal split ---")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_df = onset_df[
            onset_df["temporal_split"]
            == split_name
        ]

        positives = int(
            split_df[TARGET_COLUMN].sum()
        )

        print(f"\n{split_name.upper()}")

        print(
            "Rows:",
            f"{len(split_df):,}",
        )

        print(
            "Positive onsets:",
            f"{positives:,}",
        )

        print(
            "Positive rate:",
            f"{positives / len(split_df) * 100:.4f}%",
        )

        print(
            "Timestamps:",
            split_df[
                TIMESTAMP_COLUMN
            ].nunique(),
        )

    # --------------------------------------------------------
    # 5. Audit every selected feature.
    # --------------------------------------------------------

    print("\n--- Selected-feature audit ---")

    audit_rows = []

    non_numeric_features = []
    training_missing_features = []
    training_non_finite_features = []
    training_constant_features = []

    for feature in selected_features:
        series = onset_df[feature]

        numeric = is_numeric_or_boolean(
            series
        )

        train_series = onset_df.loc[
            onset_df["temporal_split"]
            == "train",
            feature,
        ]

        validation_series = onset_df.loc[
            onset_df["temporal_split"]
            == "validation",
            feature,
        ]

        test_series = onset_df.loc[
            onset_df["temporal_split"]
            == "test",
            feature,
        ]

        train_missing = int(
            train_series.isna().sum()
        )

        validation_missing = int(
            validation_series.isna().sum()
        )

        test_missing = int(
            test_series.isna().sum()
        )

        train_unique = int(
            train_series.nunique(
                dropna=True
            )
        )

        if numeric:
            train_finite = values_are_finite(
                train_series
            )

            validation_finite = (
                values_are_finite(
                    validation_series
                )
            )

            test_finite = values_are_finite(
                test_series
            )
        else:
            train_finite = False
            validation_finite = False
            test_finite = False

        print(
            f"{feature:<42}"
            f"dtype={str(series.dtype):<10} "
            f"train_missing={train_missing:>5} "
            f"train_unique={train_unique:>6} "
            f"train_finite={train_finite}"
        )

        audit_rows.append(
            {
                "feature": feature,
                "dtype": str(
                    series.dtype
                ),
                "numeric_or_boolean": numeric,
                "train_missing": train_missing,
                "validation_missing": (
                    validation_missing
                ),
                "test_missing": test_missing,
                "train_unique_values": (
                    train_unique
                ),
                "train_finite": train_finite,
                "validation_finite": (
                    validation_finite
                ),
                "test_finite": test_finite,
                "used_in_time_only": (
                    feature
                    in TIME_ONLY_FEATURES
                ),
                "used_in_qoe_history_only": (
                    feature
                    in QOE_HISTORY_ONLY_FEATURES
                ),
                "used_in_full_causal": (
                    feature
                    in FULL_CAUSAL_FEATURES
                ),
            }
        )

        if not numeric:
            non_numeric_features.append(
                feature
            )

        if train_missing > 0:
            training_missing_features.append(
                feature
            )

        if numeric and not train_finite:
            training_non_finite_features.append(
                feature
            )

        if train_unique <= 1:
            training_constant_features.append(
                feature
            )

    # --------------------------------------------------------
    # 6. Check for near-duplicate selected features.
    # --------------------------------------------------------

    print(
        "\n--- Near-duplicate correlation warning check ---"
    )

    training_df = onset_df[
        onset_df["temporal_split"]
        == "train"
    ]

    correlation_matrix = (
        training_df[
            FULL_CAUSAL_FEATURES
        ]
        .astype("float64")
        .corr()
    )

    high_correlation_pairs = []

    for left_index, left_feature in enumerate(
        FULL_CAUSAL_FEATURES
    ):
        for right_feature in (
            FULL_CAUSAL_FEATURES[
                left_index + 1:
            ]
        ):
            correlation = correlation_matrix.loc[
                left_feature,
                right_feature,
            ]

            if (
                pd.notna(correlation)
                and abs(correlation) >= 0.995
            ):
                high_correlation_pairs.append(
                    {
                        "left_feature": left_feature,
                        "right_feature": right_feature,
                        "correlation": float(
                            correlation
                        ),
                    }
                )

    if not high_correlation_pairs:
        print(
            "No selected feature pairs have "
            "|correlation| >= 0.995."
        )
    else:
        print(
            "Near-duplicate selected feature pairs found:"
        )

        for pair in high_correlation_pairs:
            print(
                f"  {pair['left_feature']} <-> "
                f"{pair['right_feature']}: "
                f"{pair['correlation']:.5f}"
            )

        print(
            "This is a warning only. Time-lagged QoE "
            "variables may naturally be highly correlated."
        )

    # --------------------------------------------------------
    # 7. Save audit and registry.
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        FEATURE_AUDIT_PATH,
        index=False,
    )

    split_counts = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_df = onset_df[
            onset_df["temporal_split"]
            == split_name
        ]

        split_counts[split_name] = {
            "rows": len(split_df),
            "positive_onsets": int(
                split_df[
                    TARGET_COLUMN
                ].sum()
            ),
            "timestamps": int(
                split_df[
                    TIMESTAMP_COLUMN
                ].nunique()
            ),
        }

    registry = {
        "dataset": (
            "cell_time_dataset.parquet"
        ),
        "data_status": (
            "Ookla-anchored simulated hourly scenario"
        ),
        "task": (
            "Among anchored currently normal cells, "
            "predict degradation onset in the next hour"
        ),
        "target": TARGET_COLUMN,
        "feature_sets": (
            MODEL_FEATURE_SETS
        ),
        "feature_set_counts": {
            name: len(features)
            for name, features
            in MODEL_FEATURE_SETS.items()
        },
        "excluded_shortcut_or_redundant_columns": (
            EXCLUDED_SHORTCUT_OR_REDUNDANT_COLUMNS
        ),
        "temporal_split": split_counts,
        "high_correlation_pairs": (
            high_correlation_pairs
        ),
        "test_usage_rule": (
            "The test split must not be used for feature "
            "selection, model selection or threshold tuning."
        ),
    }

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # 8. Final result.
    # --------------------------------------------------------

    split_positive_check = all(
        split_counts[split_name][
            "positive_onsets"
        ] > 0
        for split_name in [
            "train",
            "validation",
            "test",
        ]
    )

    checks_passed = (
        not missing_columns
        and not forbidden_overlap
        and not non_numeric_features
        and not training_missing_features
        and not training_non_finite_features
        and not training_constant_features
        and split_positive_check
    )

    print("\n--- Final result ---")

    if checks_passed:
        print(
            "Onset feature registry check PASSED."
        )

        print(
            f"Time-only baseline: "
            f"{len(TIME_ONLY_FEATURES)} features"
        )

        print(
            f"QoE-history baseline: "
            f"{len(QOE_HISTORY_ONLY_FEATURES)} features"
        )

        print(
            f"Full causal model: "
            f"{len(FULL_CAUSAL_FEATURES)} features"
        )

        print(
            "The model-training workflow can begin."
        )
    else:
        print(
            "Onset feature registry check FAILED."
        )

        if non_numeric_features:
            print(
                "Non-numeric features:",
                non_numeric_features,
            )

        if training_missing_features:
            print(
                "Features with training missing values:",
                training_missing_features,
            )

        if training_non_finite_features:
            print(
                "Features with non-finite training values:",
                training_non_finite_features,
            )

        if training_constant_features:
            print(
                "Constant training features:",
                training_constant_features,
            )

    print("\n--- Saved outputs ---")

    print(
        f"Feature audit:\n{FEATURE_AUDIT_PATH}"
    )

    print(
        f"\nFeature registry:\n{REGISTRY_PATH}"
    )


if __name__ == "__main__":
    main()