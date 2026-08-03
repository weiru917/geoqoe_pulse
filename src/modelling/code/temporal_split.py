from pathlib import Path

import numpy as np
import pandas as pd

from feature_registry import APPROVED_FEATURES, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)


def create_temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide the dataset chronologically using complete forecast timestamps.

    Earlier forecast periods go into training.
    Later periods go into validation and testing.
    """

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError(
            "Train and validation fractions must be positive."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "Train fraction plus validation fraction must be less than 1."
        )

    forecast_times = np.sort(
        df["forecast_timestamp"].dropna().unique()
    )

    number_of_times = len(forecast_times)

    if number_of_times < 10:
        raise ValueError(
            "Too few forecast timestamps for temporal splitting."
        )

    train_end = int(number_of_times * train_fraction)

    validation_end = int(
        number_of_times
        * (train_fraction + validation_fraction)
    )

    train_times = forecast_times[:train_end]

    validation_times = forecast_times[
        train_end:validation_end
    ]

    test_times = forecast_times[validation_end:]

    train_df = df[
        df["forecast_timestamp"].isin(train_times)
    ].copy()

    validation_df = df[
        df["forecast_timestamp"].isin(validation_times)
    ].copy()

    test_df = df[
        df["forecast_timestamp"].isin(test_times)
    ].copy()

    return train_df, validation_df, test_df


def print_split_summary(
    name: str,
    split_df: pd.DataFrame,
) -> None:
    """Print basic information for one dataset split."""

    target_counts = (
        split_df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(f"\n--- {name} ---")

    print(
        "Rows:",
        f"{len(split_df):,}",
    )

    print(
        "Unique H3 cells:",
        f"{split_df['h3_index'].nunique():,}",
    )

    print(
        "Unique predictor timestamps:",
        f"{split_df['timestamp'].nunique():,}",
    )

    print(
        "Unique forecast timestamps:",
        f"{split_df['forecast_timestamp'].nunique():,}",
    )

    print(
        "First predictor timestamp:",
        split_df["timestamp"].min(),
    )

    print(
        "Last predictor timestamp:",
        split_df["timestamp"].max(),
    )

    print(
        "First forecast timestamp:",
        split_df["forecast_timestamp"].min(),
    )

    print(
        "Last forecast timestamp:",
        split_df["forecast_timestamp"].max(),
    )

    print("Target counts:")
    print(target_counts)

    print(
        "Positive degradation rate:",
        f"{split_df[TARGET_COLUMN].mean() * 100:.4f}%",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Temporal Split Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    required_columns = [
        "h3_index",
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        TARGET_COLUMN,
        *APPROVED_FEATURES,
    ]

    # Remove any repeated names while preserving order.
    required_columns = list(dict.fromkeys(required_columns))

    print(f"\nLoading selected columns from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df["forecast_timestamp"] = pd.to_datetime(
        df["forecast_timestamp"],
        errors="raise",
    )

    # Use only rows with a complete 24-hour history.
    clean_df = df[
        df["in_warmup_window"] == 0
    ].copy()

    clean_df = clean_df.sort_values(
        ["forecast_timestamp", "h3_index"]
    ).reset_index(drop=True)

    print("\nClean dataset rows:", f"{len(clean_df):,}")
    print(
        "Available forecast timestamps:",
        f"{clean_df['forecast_timestamp'].nunique():,}",
    )

    train_df, validation_df, test_df = create_temporal_split(
        clean_df,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    print_split_summary(
        "TRAINING SET",
        train_df,
    )

    print_split_summary(
        "VALIDATION SET",
        validation_df,
    )

    print_split_summary(
        "TEST SET",
        test_df,
    )

    # Check that forecast timestamps never overlap between sets.
    train_times = set(train_df["forecast_timestamp"].unique())
    validation_times = set(
        validation_df["forecast_timestamp"].unique()
    )
    test_times = set(test_df["forecast_timestamp"].unique())

    train_validation_overlap = (
        train_times.intersection(validation_times)
    )

    train_test_overlap = (
        train_times.intersection(test_times)
    )

    validation_test_overlap = (
        validation_times.intersection(test_times)
    )

    print("\n--- Overlap checks ---")

    print(
        "Training/validation overlapping forecast timestamps:",
        len(train_validation_overlap),
    )

    print(
        "Training/test overlapping forecast timestamps:",
        len(train_test_overlap),
    )

    print(
        "Validation/test overlapping forecast timestamps:",
        len(validation_test_overlap),
    )

    total_rows_after_split = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print("\n--- Row reconciliation ---")

    print(
        "Clean dataset rows:",
        f"{len(clean_df):,}",
    )

    print(
        "Rows across all three splits:",
        f"{total_rows_after_split:,}",
    )

    checks_passed = (
        len(train_df) > 0
        and len(validation_df) > 0
        and len(test_df) > 0
        and train_df[TARGET_COLUMN].nunique() == 2
        and validation_df[TARGET_COLUMN].nunique() == 2
        and test_df[TARGET_COLUMN].nunique() == 2
        and len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
        and total_rows_after_split == len(clean_df)
    )

    print("\n--- Final result ---")

    if checks_passed:
        print("Temporal split check PASSED.")
        print(
            "Training, validation, and test sets are "
            "chronologically separate."
        )
    else:
        print("Temporal split check FAILED.")
        print(
            "One split may be empty, contain only one target class, "
            "overlap another split, or lose rows."
        )


if __name__ == "__main__":
    main()