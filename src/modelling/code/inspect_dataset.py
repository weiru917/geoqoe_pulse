from pathlib import Path

import pandas as pd


# Locate the main GeoQoE_AI project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the current development dataset.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "cell_time_dataset_development.parquet"
)


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Development Dataset Inspection")
    print("=" * 70)

    print(f"\nExpected dataset location:\n{DATA_PATH}")

    # Stop with a clear message if the file is missing.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "\nDataset not found.\n"
            "Check that cell_time_dataset_development.parquet is placed inside:\n"
            f"{DATA_PATH.parent}"
        )

    print("\nLoading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print("Dataset loaded successfully.")

    print("\n--- Basic dimensions ---")
    print(f"Number of rows: {len(df):,}")
    print(f"Number of columns: {len(df.columns):,}")

    print("\n--- Column names ---")
    for number, column in enumerate(df.columns, start=1):
        print(f"{number:>3}. {column}")

    print("\n--- First five rows ---")
    print(df.head())

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    missing_fraction = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
    )

    missing_columns = missing_fraction[
        missing_fraction > 0
    ]

    if missing_columns.empty:
        print("No missing values found.")
    else:
        print("Columns containing missing values:")
        for column, fraction in missing_columns.items():
            print(
                f"{column}: "
                f"{fraction * 100:.2f}% missing"
            )

    required_columns = {
        "h3_index",
        "timestamp",
        "target_is_degraded",
    }

    missing_required = required_columns.difference(df.columns)

    print("\n--- Required-column check ---")

    if missing_required:
        print(
            "Missing required columns:",
            sorted(missing_required),
        )
    else:
        print("All required columns are present.")

        # Convert timestamps safely.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        print("\n--- Spatial-temporal structure ---")
        print(
            "Unique H3 cells:",
            f"{df['h3_index'].nunique():,}",
        )
        print(
            "Unique timestamps:",
            f"{df['timestamp'].nunique():,}",
        )
        print(
            "Earliest timestamp:",
            df["timestamp"].min(),
        )
        print(
            "Latest timestamp:",
            df["timestamp"].max(),
        )

        duplicate_count = df.duplicated(
            subset=["h3_index", "timestamp"]
        ).sum()

        print(
            "Duplicate cell-time rows:",
            f"{duplicate_count:,}",
        )

        print("\n--- Degradation target ---")
        target_counts = df[
            "target_is_degraded"
        ].value_counts(
            dropna=False
        )

        print(target_counts)

        positive_rate = df[
            "target_is_degraded"
        ].mean()

        print(
            "Positive degradation rate:",
            f"{positive_rate * 100:.4f}%",
        )

        observations_per_cell = (
            df.groupby("h3_index")["timestamp"]
            .nunique()
        )

        print("\n--- Timestamps per H3 cell ---")
        print(observations_per_cell.describe())

    print("\nInspection completed.")


if __name__ == "__main__":
    main()