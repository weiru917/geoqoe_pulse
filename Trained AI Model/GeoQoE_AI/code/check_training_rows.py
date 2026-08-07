from pathlib import Path

import pandas as pd

from feature_registry import APPROVED_FEATURES, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Clean Training Rows Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    # These columns are needed for this check.
    required_columns = [
        "h3_index",
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        TARGET_COLUMN,
        *APPROVED_FEATURES,
    ]

    # Remove duplicate column names while preserving their order.
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

    print("\nDataset loaded successfully.")

    print("\n--- Before warm-up removal ---")
    print("Rows:", f"{len(df):,}")
    print(
        "Unique H3 cells:",
        f"{df['h3_index'].nunique():,}",
    )
    print(
        "Unique predictor timestamps:",
        f"{df['timestamp'].nunique():,}",
    )

    print("\nWarm-up flag distribution:")
    print(
        df["in_warmup_window"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # Keep only rows with a complete 24-hour history.
    clean_df = df[
        df["in_warmup_window"] == 0
    ].copy()

    print("\n--- After warm-up removal ---")
    print("Rows:", f"{len(clean_df):,}")
    print(
        "Rows removed:",
        f"{len(df) - len(clean_df):,}",
    )
    print(
        "Unique H3 cells:",
        f"{clean_df['h3_index'].nunique():,}",
    )
    print(
        "Unique predictor timestamps:",
        f"{clean_df['timestamp'].nunique():,}",
    )
    print(
        "Earliest predictor timestamp:",
        clean_df["timestamp"].min(),
    )
    print(
        "Latest predictor timestamp:",
        clean_df["timestamp"].max(),
    )
    print(
        "Earliest forecast timestamp:",
        clean_df["forecast_timestamp"].min(),
    )
    print(
        "Latest forecast timestamp:",
        clean_df["forecast_timestamp"].max(),
    )

    # Check missing feature values after removing warm-up rows.
    feature_missing_counts = (
        clean_df[APPROVED_FEATURES]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_features = feature_missing_counts[
        feature_missing_counts > 0
    ]

    rows_with_missing_features = int(
        clean_df[APPROVED_FEATURES]
        .isna()
        .any(axis=1)
        .sum()
    )

    print("\n--- Missing-feature check ---")

    if missing_features.empty:
        print(
            "No missing values remain in the "
            "39 approved features."
        )
    else:
        print("Approved features with missing values:")

        for column, count in missing_features.items():
            print(f"  {column}: {count:,}")

    print(
        "Rows containing at least one missing approved feature:",
        f"{rows_with_missing_features:,}",
    )

    target_missing = int(
        clean_df[TARGET_COLUMN]
        .isna()
        .sum()
    )

    print(
        "Rows with missing target:",
        f"{target_missing:,}",
    )

    print("\n--- Clean target distribution ---")

    target_counts = (
        clean_df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    print(target_counts)

    positive_rate = clean_df[
        TARGET_COLUMN
    ].mean()

    print(
        "Positive degradation rate:",
        f"{positive_rate * 100:.4f}%",
    )

    observations_per_cell = (
        clean_df.groupby("h3_index")["timestamp"]
        .nunique()
    )

    print("\n--- Clean observations per H3 cell ---")
    print(observations_per_cell.describe())

    checks_passed = (
        len(clean_df) > 0
        and clean_df["h3_index"].nunique() > 0
        and clean_df["timestamp"].nunique() > 1
        and rows_with_missing_features == 0
        and target_missing == 0
        and clean_df[TARGET_COLUMN].nunique() == 2
    )

    print("\n--- Final result ---")

    if checks_passed:
        print("Clean training-row check PASSED.")
        print(
            "The rows after the 24-hour warm-up period "
            "are ready for train/validation/test splitting."
        )
    else:
        print("Clean training-row check FAILED.")


if __name__ == "__main__":
    main()