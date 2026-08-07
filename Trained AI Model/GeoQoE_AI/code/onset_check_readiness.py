from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
)

HOURLY_DATA_PATH = (
    DATA_DIRECTORY
    / "cell_time_dataset.parquet"
)

FEATURE_DICTIONARY_PATH = (
    DATA_DIRECTORY
    / "feature_dictionary.xlsx"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
)

CANDIDATE_FEATURES_PATH = (
    OUTPUT_DIRECTORY
    / "onset_dictionary_feature_candidates.csv"
)

SPLIT_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "onset_temporal_split_summary.csv"
)

READINESS_REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "onset_readiness_report.json"
)


# Existing chronological development split:
#
# 143 usable predictor timestamps
# - first 100: training
# - next 21: validation
# - final 22: test
EXPECTED_TRAIN_TIMESTAMPS = 100
EXPECTED_VALIDATION_TIMESTAMPS = 21


MANDATORY_COLUMNS = [
    "h3_index",
    "h3_parent_res6",
    "timestamp",
    "in_warmup_window",
    "qoe_level_is_ookla_anchored",
    "target_is_degraded",
    "target_is_degraded_next_1h",
]


# These should not enter the onset model even when a dictionary
# accidentally marks them as available.
AUTOMATIC_EXCLUSION_COLUMNS = {
    "row_id",
    "h3_index",
    "h3_parent_res6",
    "timestamp",
    "in_warmup_window",
    "target_is_degraded",
    "target_is_degraded_next_1h",
    "qoe_level_is_ookla_anchored",
    "qoe_baseline_source",
    "priority_base_score",
    "recommended_action",
    "likely_cause",
    "ookla_quarter_test_count",
    "ookla_quarter_device_count",
}


# These demand-pressure variables contain historical QoE
# components or are derived from another QoE-linked index.
# They are avoided so the model interpretation remains clearer.
CIRCULAR_DEMAND_COLUMNS = {
    "demand_pressure_index",
    "demand_pressure_index_pre_spatial",
    "spatial_lag_demand_pressure_index_pre_spatial",
    "spatial_var_demand_pressure_index_pre_spatial",
}


# These are exact formula transformations of other same-hour
# measurements. They create artificial feature richness without
# adding independent information.
FORMULA_DERIVED_DUPLICATES = {
    "dl_mean",
    "dl_std",
    "dl_p95",
    "ul_mean",
    "ul_std",
    "ul_p95",
    "rtt_mean",
    "rtt_std",
    "rtt_p95",
}


def normalise_boolean_flag(series: pd.Series) -> pd.Series:
    """
    Convert workbook boolean values such as True, TRUE, 1,
    or yes into a consistent Boolean Series.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def classify_dictionary_candidate(
    column: str,
) -> str:
    """
    Assign an initial review status.

    This is an audit classification only. The final feature
    registry will be created in the next checkpoint.
    """

    lowered = column.lower()

    if (
        column in AUTOMATIC_EXCLUSION_COLUMNS
        or lowered.startswith("target_")
    ):
        return "exclude_target_id_or_control"

    if (
        column in CIRCULAR_DEMAND_COLUMNS
    ):
        return "exclude_circular_demand_index"

    if (
        column in FORMULA_DERIVED_DUPLICATES
    ):
        return "exclude_formula_duplicate"

    if (
        lowered.startswith("ookla_")
        or "synthetic" in lowered
        or "source" in lowered
        or "anchored" in lowered
        or "provenance" in lowered
    ):
        return "exclude_provenance_or_ookla_status"

    return "review_for_final_registry"


def build_split_summary(
    onset_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Divide the onset candidates chronologically.

    The final test period must remain untouched until all model
    and threshold decisions have been completed.
    """

    unique_timestamps = pd.Index(
        sorted(
            onset_df["timestamp"].unique()
        )
    )

    number_of_timestamps = len(
        unique_timestamps
    )

    if number_of_timestamps < 3:
        raise ValueError(
            "Too few usable timestamps for temporal splitting."
        )

    if number_of_timestamps == 143:
        training_count = (
            EXPECTED_TRAIN_TIMESTAMPS
        )

        validation_count = (
            EXPECTED_VALIDATION_TIMESTAMPS
        )
    else:
        print(
            "\nWARNING: The usable timestamp count is not 143."
        )

        print(
            "A general 70% / 15% / 15% chronological split "
            "will be used for this audit."
        )

        training_count = int(
            number_of_timestamps * 0.70
        )

        validation_count = int(
            number_of_timestamps * 0.15
        )

    validation_end = (
        training_count
        + validation_count
    )

    training_times = set(
        unique_timestamps[
            :training_count
        ]
    )

    validation_times = set(
        unique_timestamps[
            training_count:validation_end
        ]
    )

    test_times = set(
        unique_timestamps[
            validation_end:
        ]
    )

    split_lookup = {}

    split_lookup.update(
        {
            timestamp: "train"
            for timestamp in training_times
        }
    )

    split_lookup.update(
        {
            timestamp: "validation"
            for timestamp in validation_times
        }
    )

    split_lookup.update(
        {
            timestamp: "test"
            for timestamp in test_times
        }
    )

    onset_df = onset_df.copy()

    onset_df[
        "temporal_split"
    ] = onset_df[
        "timestamp"
    ].map(split_lookup)

    if onset_df["temporal_split"].isna().any():
        raise ValueError(
            "Some onset rows did not receive a temporal split."
        )

    summary_rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_df = onset_df[
            onset_df["temporal_split"]
            == split_name
        ]

        positive_count = int(
            split_df[
                "onset_target"
            ].sum()
        )

        negative_count = (
            len(split_df)
            - positive_count
        )

        positive_rate = (
            positive_count
            / len(split_df)
            if len(split_df) > 0
            else 0.0
        )

        summary_rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "positive_onsets": positive_count,
                "negative_rows": negative_count,
                "positive_rate": positive_rate,
                "unique_timestamps": (
                    split_df[
                        "timestamp"
                    ].nunique()
                ),
                "unique_h3_cells": (
                    split_df[
                        "h3_index"
                    ].nunique()
                ),
                "spatial_parent_groups": (
                    split_df[
                        "h3_parent_res6"
                    ].nunique()
                ),
                "first_timestamp": (
                    split_df[
                        "timestamp"
                    ].min()
                ),
                "last_timestamp": (
                    split_df[
                        "timestamp"
                    ].max()
                ),
            }
        )

    split_summary_df = pd.DataFrame(
        summary_rows
    )

    split_metadata = {
        "usable_timestamps": (
            number_of_timestamps
        ),
        "training_timestamps": len(
            training_times
        ),
        "validation_timestamps": len(
            validation_times
        ),
        "test_timestamps": len(
            test_times
        ),
    }

    return (
        split_summary_df,
        split_metadata,
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Hourly Onset Readiness Audit")
    print("=" * 70)

    if not HOURLY_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found:\n"
            f"{HOURLY_DATA_PATH}"
        )

    if not FEATURE_DICTIONARY_PATH.exists():
        raise FileNotFoundError(
            f"Feature dictionary not found:\n"
            f"{FEATURE_DICTIONARY_PATH}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Inspect the Parquet schema without loading all columns.
    # ========================================================

    print(
        f"\nInspecting hourly Parquet schema:\n"
        f"{HOURLY_DATA_PATH}"
    )

    parquet_file = pq.ParquetFile(
        HOURLY_DATA_PATH
    )

    schema = parquet_file.schema_arrow

    parquet_columns = schema.names

    print("\n--- Hourly Parquet schema ---")

    print(
        "Parquet rows:",
        f"{parquet_file.metadata.num_rows:,}",
    )

    print(
        "Parquet columns:",
        len(parquet_columns),
    )

    missing_mandatory_columns = [
        column
        for column in MANDATORY_COLUMNS
        if column not in parquet_columns
    ]

    if missing_mandatory_columns:
        raise ValueError(
            "Required hourly columns are missing: "
            f"{missing_mandatory_columns}"
        )

    print(
        "All mandatory onset-audit columns are present."
    )

    # ========================================================
    # 2. Inspect dictionary-approved next-hour features.
    # ========================================================

    print("\n--- Feature dictionary audit ---")

    dictionary_df = pd.read_excel(
        FEATURE_DICTIONARY_PATH,
        sheet_name="Sheet1",
    )

    required_dictionary_columns = {
        "file",
        "column",
        "dtype",
        "pct_non_null",
        "is_feature_for_next_hour_target",
        "leakage_risk_note",
        "description",
    }

    missing_dictionary_columns = (
        required_dictionary_columns
        .difference(
            dictionary_df.columns
        )
    )

    if missing_dictionary_columns:
        raise ValueError(
            "Feature dictionary is missing columns: "
            f"{sorted(missing_dictionary_columns)}"
        )

    hourly_dictionary_df = dictionary_df[
        dictionary_df["file"]
        .astype(str)
        .str.strip()
        .eq("cell_time_dataset")
    ].copy()

    next_hour_flag = normalise_boolean_flag(
        hourly_dictionary_df[
            "is_feature_for_next_hour_target"
        ]
    )

    candidate_df = hourly_dictionary_df[
        next_hour_flag
    ].copy()

    candidate_df[
        "present_in_parquet"
    ] = candidate_df[
        "column"
    ].isin(parquet_columns)

    candidate_df[
        "initial_review_status"
    ] = candidate_df[
        "column"
    ].map(
        classify_dictionary_candidate
    )

    print(
        "Dictionary rows for hourly dataset:",
        len(hourly_dictionary_df),
    )

    print(
        "Dictionary-approved next-hour candidates:",
        len(candidate_df),
    )

    print(
        "Candidates present in Parquet:",
        int(
            candidate_df[
                "present_in_parquet"
            ].sum()
        ),
    )

    print(
        "Candidates requiring final review:",
        int(
            (
                candidate_df[
                    "initial_review_status"
                ]
                == "review_for_final_registry"
            ).sum()
        ),
    )

    print(
        "\n--- Dictionary-approved next-hour columns ---"
    )

    for _, row in candidate_df.iterrows():
        column = str(
            row["column"]
        )

        dtype = str(
            row["dtype"]
        )

        non_null = row[
            "pct_non_null"
        ]

        status = row[
            "initial_review_status"
        ]

        print(
            f"\n{column}"
        )

        print(
            f"  dtype: {dtype}"
        )

        print(
            f"  non-null: {non_null}%"
        )

        print(
            f"  initial status: {status}"
        )

        leakage_note = row[
            "leakage_risk_note"
        ]

        if pd.notna(leakage_note):
            print(
                f"  leakage note: {leakage_note}"
            )

        description = row[
            "description"
        ]

        if pd.notna(description):
            print(
                f"  description: {description}"
            )

    candidate_df.to_csv(
        CANDIDATE_FEATURES_PATH,
        index=False,
    )

    # ========================================================
    # 3. Load only the columns needed to audit onset labels.
    # ========================================================

    print(
        "\n--- Loading minimal hourly onset columns ---"
    )

    hourly_df = pd.read_parquet(
        HOURLY_DATA_PATH,
        columns=MANDATORY_COLUMNS,
    )

    hourly_df[
        "timestamp"
    ] = pd.to_datetime(
        hourly_df["timestamp"],
        errors="raise",
    )

    current_state = pd.to_numeric(
        hourly_df[
            "target_is_degraded"
        ],
        errors="coerce",
    )

    next_state = pd.to_numeric(
        hourly_df[
            "target_is_degraded_next_1h"
        ],
        errors="coerce",
    )

    warmup_mask = hourly_df[
        "in_warmup_window"
    ].astype(bool)

    anchored_mask = hourly_df[
        "qoe_level_is_ookla_anchored"
    ].astype(bool)

    valid_next_target = next_state.notna()

    currently_normal = (
        current_state == 0
    )

    onset_candidate_mask = (
        valid_next_target
        & ~warmup_mask
        & anchored_mask
        & currently_normal
    )

    onset_df = hourly_df[
        onset_candidate_mask
    ].copy()

    onset_df[
        "onset_target"
    ] = (
        pd.to_numeric(
            onset_df[
                "target_is_degraded_next_1h"
            ],
            errors="raise",
        )
        .astype("int8")
    )

    total_positive_onsets = int(
        onset_df[
            "onset_target"
        ].sum()
    )

    print(
        "Hourly panel rows:",
        f"{len(hourly_df):,}",
    )

    print(
        "Anchored, currently normal onset candidates:",
        f"{len(onset_df):,}",
    )

    print(
        "Positive next-hour onsets:",
        f"{total_positive_onsets:,}",
    )

    print(
        "Overall onset rate:",
        f"{total_positive_onsets / len(onset_df) * 100:.4f}%",
    )

    print(
        "Usable predictor timestamps:",
        onset_df[
            "timestamp"
        ].nunique(),
    )

    # ========================================================
    # 4. Lock chronological train/validation/test periods.
    # ========================================================

    (
        split_summary_df,
        split_metadata,
    ) = build_split_summary(
        onset_df
    )

    print("\n--- Chronological split summary ---")

    for _, row in split_summary_df.iterrows():
        print(
            f"\n{row['split'].upper()}"
        )

        print(
            "Rows:",
            f"{int(row['rows']):,}",
        )

        print(
            "Positive onsets:",
            f"{int(row['positive_onsets']):,}",
        )

        print(
            "Positive rate:",
            f"{float(row['positive_rate']) * 100:.4f}%",
        )

        print(
            "Timestamps:",
            int(
                row[
                    "unique_timestamps"
                ]
            ),
        )

        print(
            "Unique cells:",
            f"{int(row['unique_h3_cells']):,}",
        )

        print(
            "Spatial parent groups:",
            int(
                row[
                    "spatial_parent_groups"
                ]
            ),
        )

        print(
            "Period:",
            row[
                "first_timestamp"
            ],
            "to",
            row[
                "last_timestamp"
            ],
        )

    split_summary_df.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # 5. Inspect temporal concentration of positive onsets.
    # ========================================================

    positive_onsets_df = onset_df[
        onset_df[
            "onset_target"
        ] == 1
    ].copy()

    positive_onsets_df[
        "hour_of_day"
    ] = positive_onsets_df[
        "timestamp"
    ].dt.hour

    positive_onsets_df[
        "calendar_date"
    ] = positive_onsets_df[
        "timestamp"
    ].dt.date.astype(str)

    onset_by_hour = (
        positive_onsets_df
        .groupby("hour_of_day")
        .size()
        .sort_values(
            ascending=False
        )
    )

    onset_by_date = (
        positive_onsets_df
        .groupby("calendar_date")
        .size()
        .sort_index()
    )

    print("\n--- Positive onsets by hour of day ---")

    print(onset_by_hour)

    print("\n--- Positive onsets by date ---")

    print(onset_by_date)

    top_two_hour_count = int(
        onset_by_hour
        .head(2)
        .sum()
    )

    top_two_hour_share = (
        top_two_hour_count
        / total_positive_onsets
        if total_positive_onsets > 0
        else 0.0
    )

    print(
        "\nShare of onsets occurring in the two "
        "most common hours:",
        f"{top_two_hour_share * 100:.2f}%",
    )

    # ========================================================
    # 6. Final readiness checks.
    # ========================================================

    split_has_positive_events = bool(
        (
            split_summary_df[
                "positive_onsets"
            ] > 0
        ).all()
    )

    correct_total_onsets = (
        int(
            split_summary_df[
                "positive_onsets"
            ].sum()
        )
        == total_positive_onsets
    )

    all_dictionary_candidates_present = bool(
        candidate_df[
            "present_in_parquet"
        ].all()
    )

    readiness_passed = (
        len(onset_df) > 0
        and total_positive_onsets > 0
        and split_has_positive_events
        and correct_total_onsets
        and all_dictionary_candidates_present
    )

    print("\n--- Final readiness result ---")

    if readiness_passed:
        print(
            "Hourly onset readiness audit PASSED."
        )

        print(
            "Training, validation, and test periods all "
            "contain positive onset events."
        )
    else:
        print(
            "Hourly onset readiness audit FAILED."
        )

        print(
            "Split has positive events:",
            split_has_positive_events,
        )

        print(
            "Split onset total reconciles:",
            correct_total_onsets,
        )

        print(
            "All dictionary candidates exist:",
            all_dictionary_candidates_present,
        )

    report = {
        "dataset": (
            "cell_time_dataset.parquet"
        ),
        "data_status": (
            "Ookla-anchored simulated hourly scenario"
        ),
        "task": (
            "Among anchored cells currently normal, "
            "predict degradation onset in the next hour"
        ),
        "candidate_rows": len(
            onset_df
        ),
        "positive_onsets": (
            total_positive_onsets
        ),
        "positive_rate": float(
            total_positive_onsets
            / len(onset_df)
        ),
        "temporal_split": (
            split_metadata
        ),
        "split_positive_events": {
            str(row["split"]): int(
                row[
                    "positive_onsets"
                ]
            )
            for _, row in (
                split_summary_df.iterrows()
            )
        },
        "dictionary_approved_candidates": len(
            candidate_df
        ),
        "dictionary_candidates_for_review": int(
            (
                candidate_df[
                    "initial_review_status"
                ]
                == "review_for_final_registry"
            ).sum()
        ),
        "top_two_onset_hour_share": float(
            top_two_hour_share
        ),
        "readiness_passed": (
            readiness_passed
        ),
        "important_limitation": (
            "The hourly panel and onset labels are simulated. "
            "The test period evaluates generalisation within "
            "the simulation, not performance on observed "
            "hourly telecom telemetry."
        ),
    }

    with open(
        READINESS_REPORT_PATH,
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
        f"Dictionary candidate audit:\n"
        f"{CANDIDATE_FEATURES_PATH}"
    )

    print(
        f"\nTemporal split summary:\n"
        f"{SPLIT_SUMMARY_PATH}"
    )

    print(
        f"\nReadiness report:\n"
        f"{READINESS_REPORT_PATH}"
    )

    print(
        "\nDo not use the test period for feature selection, "
        "model selection, or threshold tuning."
    )


if __name__ == "__main__":
    main()