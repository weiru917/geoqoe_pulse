from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
)

METADATA_PATH = DATA_DIRECTORY / "metadata.json"

STATIC_PATH = (
    DATA_DIRECTORY
    / "cell_static_qoe_baseline.parquet"
)

HOURLY_PATH = (
    DATA_DIRECTORY
    / "cell_time_dataset.parquet"
)

REQUIRED_FILES = [
    "cell_static_features.parquet",
    "cell_static_qoe_baseline.parquet",
    "cell_time_dataset.parquet",
    "feature_dictionary.xlsx",
    "metadata.json",
    "qoe_targets.parquet",
]


def print_separator(title: str) -> None:
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — New Dataset Package Inspection")
    print("=" * 70)

    print(f"\nExpected data folder:\n{DATA_DIRECTORY}")

    # ---------------------------------------------------------
    # 1. Check that all package files exist.
    # ---------------------------------------------------------

    missing_files = [
        filename
        for filename in REQUIRED_FILES
        if not (DATA_DIRECTORY / filename).exists()
    ]

    print_separator("PACKAGE FILE CHECK")

    if missing_files:
        print("Missing files:")

        for filename in missing_files:
            print(f"  - {filename}")

        raise FileNotFoundError(
            "The new dataset package is incomplete."
        )

    print("All six required files are present.")

    for filename in REQUIRED_FILES:
        file_path = DATA_DIRECTORY / filename
        size_mb = file_path.stat().st_size / (1024 ** 2)

        print(
            f"{filename:<42}"
            f"{size_mb:>10.2f} MB"
        )

    # ---------------------------------------------------------
    # 2. Read pipeline metadata.
    # ---------------------------------------------------------

    print_separator("PIPELINE METADATA")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    print(
        "Pipeline version:",
        metadata.get("pipeline_version"),
    )

    print(
        "Generated at:",
        metadata.get("generated_at_utc"),
    )

    print(
        "H3 resolution:",
        metadata.get("h3_resolution"),
    )

    print(
        "Study area:",
        metadata.get("study_area"),
    )

    row_counts = metadata.get(
        "row_counts",
        {},
    )

    print("\nMetadata row counts:")

    for name, value in row_counts.items():
        print(f"  {name}: {value:,}")

    disclosure = metadata.get(
        "simulation_disclosure",
        {},
    )

    print("\nHourly panel is simulated:")
    print(
        disclosure.get(
            "panel_qoe_hourly_shape_is_simulated"
        )
    )

    print(
        "Recommended defensible dataset:",
        disclosure.get(
            "recommended_defensible_dataset"
        ),
    )

    # ---------------------------------------------------------
    # 3. Inspect the real static Ookla dataset.
    # ---------------------------------------------------------

    print_separator("REAL STATIC QOE DATASET")

    static_columns = [
        "h3_index",
        "h3_parent_res6",
        "has_real_ookla_label",
        "ookla_dl_mbps",
        "ookla_rtt_ms",
        "ookla_tests",
        "qoe_risk_label",
    ]

    static_df = pd.read_parquet(
        STATIC_PATH,
        columns=static_columns,
    )

    labelled_df = static_df[
        static_df["has_real_ookla_label"]
    ].copy()

    unlabelled_df = static_df[
        ~static_df["has_real_ookla_label"]
    ].copy()

    print(
        "Total grid cells:",
        f"{len(static_df):,}",
    )

    print(
        "Real Ookla-labelled cells:",
        f"{len(labelled_df):,}",
    )

    print(
        "Unlabelled cells:",
        f"{len(unlabelled_df):,}",
    )

    print(
        "Real-labelled proportion:",
        f"{len(labelled_df) / len(static_df) * 100:.2f}%",
    )

    print(
        "Spatial parent groups:",
        f"{static_df['h3_parent_res6'].nunique():,}",
    )

    print("\nReal QoE risk-label distribution:")

    print(
        labelled_df["qoe_risk_label"]
        .value_counts(dropna=False)
    )

    print("\nReal download-speed target summary:")

    print(
        labelled_df["ookla_dl_mbps"]
        .describe()
    )

    print("\nReal Ookla test-count summary:")

    print(
        labelled_df["ookla_tests"]
        .describe()
    )

    # ---------------------------------------------------------
    # 4. Inspect the simulated hourly panel.
    # ---------------------------------------------------------

    print_separator("SIMULATED HOURLY DATASET")

    hourly_columns = [
        "h3_index",
        "h3_parent_res6",
        "timestamp",
        "in_warmup_window",
        "qoe_level_is_ookla_anchored",
        "target_is_degraded",
        "target_is_degraded_next_1h",
        "ridership_daily_surge_factor",
    ]

    hourly_df = pd.read_parquet(
        HOURLY_PATH,
        columns=hourly_columns,
    )

    hourly_df["timestamp"] = pd.to_datetime(
        hourly_df["timestamp"],
        errors="raise",
    )

    print(
        "Hourly rows:",
        f"{len(hourly_df):,}",
    )

    print(
        "Unique H3 cells:",
        f"{hourly_df['h3_index'].nunique():,}",
    )

    print(
        "Unique timestamps:",
        f"{hourly_df['timestamp'].nunique():,}",
    )

    print(
        "Earliest timestamp:",
        hourly_df["timestamp"].min(),
    )

    print(
        "Latest timestamp:",
        hourly_df["timestamp"].max(),
    )

    print(
        "Spatial parent groups:",
        f"{hourly_df['h3_parent_res6'].nunique():,}",
    )

    anchored_cells = hourly_df.loc[
        hourly_df["qoe_level_is_ookla_anchored"],
        "h3_index",
    ].nunique()

    print(
        "Ookla-anchored H3 cells:",
        f"{anchored_cells:,}",
    )

    print(
        "Distinct ridership factors:",
        hourly_df[
            "ridership_daily_surge_factor"
        ].nunique(),
    )

    # ---------------------------------------------------------
    # 5. Check usable forecasting rows and transitions.
    # ---------------------------------------------------------

    usable_hourly_df = hourly_df[
        hourly_df[
            "target_is_degraded_next_1h"
        ].notna()
    ].copy()

    clean_hourly_df = usable_hourly_df[
        usable_hourly_df["in_warmup_window"] == 0
    ].copy()

    current_state = clean_hourly_df[
        "target_is_degraded"
    ].astype("int8")

    next_state = clean_hourly_df[
        "target_is_degraded_next_1h"
    ].astype("int8")

    normal_to_normal = int(
        (
            (current_state == 0)
            & (next_state == 0)
        ).sum()
    )

    normal_to_degraded = int(
        (
            (current_state == 0)
            & (next_state == 1)
        ).sum()
    )

    degraded_to_normal = int(
        (
            (current_state == 1)
            & (next_state == 0)
        ).sum()
    )

    degraded_to_degraded = int(
        (
            (current_state == 1)
            & (next_state == 1)
        ).sum()
    )

    print_separator("CLEAN HOURLY TRANSITIONS")

    print(
        "Usable rows after warm-up:",
        f"{len(clean_hourly_df):,}",
    )

    print(
        "Normal -> Normal:",
        f"{normal_to_normal:,}",
    )

    print(
        "Normal -> Degraded:",
        f"{normal_to_degraded:,}",
    )

    print(
        "Degraded -> Normal:",
        f"{degraded_to_normal:,}",
    )

    print(
        "Degraded -> Degraded:",
        f"{degraded_to_degraded:,}",
    )

    # ---------------------------------------------------------
    # 6. Count onset-model candidates.
    # ---------------------------------------------------------

    anchored_clean_df = clean_hourly_df[
        clean_hourly_df[
            "qoe_level_is_ookla_anchored"
        ]
    ].copy()

    onset_candidates = anchored_clean_df[
        anchored_clean_df[
            "target_is_degraded"
        ] == 0
    ].copy()

    onset_count = int(
        (
            onset_candidates[
                "target_is_degraded_next_1h"
            ] == 1
        ).sum()
    )

    print_separator("ONSET-MODEL CANDIDATES")

    print(
        "Anchored, currently normal rows:",
        f"{len(onset_candidates):,}",
    )

    print(
        "Next-hour degradation onsets:",
        f"{onset_count:,}",
    )

    if len(onset_candidates) > 0:
        onset_rate = (
            onset_count
            / len(onset_candidates)
        )

        print(
            "Onset positive rate:",
            f"{onset_rate * 100:.4f}%",
        )

    # ---------------------------------------------------------
    # 7. Final consistency check.
    # ---------------------------------------------------------

    expected_static_rows = row_counts.get(
        "static_qoe_baseline_rows"
    )

    expected_hourly_rows = row_counts.get(
        "panel_rows"
    )

    expected_real_labels = row_counts.get(
        "static_qoe_baseline_real_labels"
    )

    checks_passed = (
        len(static_df) == expected_static_rows
        and len(hourly_df) == expected_hourly_rows
        and len(labelled_df) == expected_real_labels
        and static_df["h3_index"].is_unique
        and hourly_df[
            ["h3_index", "timestamp"]
        ].duplicated().sum() == 0
    )

    print_separator("FINAL RESULT")

    if checks_passed:
        print("New dataset package inspection PASSED.")
        print(
            "The files are ready for the new modelling workflow."
        )
    else:
        print("New dataset package inspection FAILED.")
        print(
            "One or more row-count, uniqueness, "
            "or metadata checks did not match."
        )


if __name__ == "__main__":
    main()