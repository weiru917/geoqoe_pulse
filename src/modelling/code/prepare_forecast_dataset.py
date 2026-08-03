from pathlib import Path

import pandas as pd


# Main project folder:
# C:\Users\yunti\OneDrive\Desktop\GeoQoE_AI
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "cell_time_dataset_development.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Next-Hour Forecast Dataset Preparation")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    print(f"\nLoading:\n{INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "h3_index",
        "timestamp",
        "target_is_degraded",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Ensure timestamp is interpreted correctly.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    # The rows must be arranged chronologically inside each H3 cell.
    df = df.sort_values(
        ["h3_index", "timestamp"]
    ).reset_index(drop=True)

    print("\nCreating next-hour timestamp and target...")

    # Find the next available timestamp inside the same H3 cell.
    df["next_timestamp"] = (
        df.groupby("h3_index")["timestamp"]
        .shift(-1)
    )

    # Obtain the degradation state of the same cell at the next timestamp.
    df["target_is_degraded_next_1h"] = (
        df.groupby("h3_index")["target_is_degraded"]
        .shift(-1)
    )

    # Calculate the actual time difference.
    df["time_gap_to_target"] = (
        df["next_timestamp"] - df["timestamp"]
    )

    # Keep only rows where the target is exactly one hour later.
    valid_next_hour = (
        df["time_gap_to_target"]
        == pd.Timedelta(hours=1)
    )

    invalid_gap_count = int(
        (
            df["next_timestamp"].notna()
            & ~valid_next_hour
        ).sum()
    )

    print(
        "Rows with a next timestamp that is not exactly one hour later:",
        f"{invalid_gap_count:,}",
    )

    forecast_df = df.loc[valid_next_hour].copy()

    # Rename the next timestamp to make its purpose clearer.
    forecast_df["forecast_timestamp"] = (
        forecast_df["next_timestamp"]
    )

    forecast_df[
        "target_is_degraded_next_1h"
    ] = forecast_df[
        "target_is_degraded_next_1h"
    ].astype("int8")

    # Temporary helper columns are no longer needed.
    forecast_df = forecast_df.drop(
        columns=[
            "next_timestamp",
            "time_gap_to_target",
        ]
    )

    print("\n--- Forecast dataset summary ---")
    print(
        "Original rows:",
        f"{len(df):,}",
    )
    print(
        "Forecast rows:",
        f"{len(forecast_df):,}",
    )
    print(
        "Rows removed:",
        f"{len(df) - len(forecast_df):,}",
    )

    print(
        "Unique H3 cells:",
        f"{forecast_df['h3_index'].nunique():,}",
    )
    print(
        "Unique predictor timestamps:",
        f"{forecast_df['timestamp'].nunique():,}",
    )
    print(
        "Unique forecast timestamps:",
        f"{forecast_df['forecast_timestamp'].nunique():,}",
    )

    print(
        "Earliest predictor timestamp:",
        forecast_df["timestamp"].min(),
    )
    print(
        "Latest predictor timestamp:",
        forecast_df["timestamp"].max(),
    )
    print(
        "Earliest forecast timestamp:",
        forecast_df["forecast_timestamp"].min(),
    )
    print(
        "Latest forecast timestamp:",
        forecast_df["forecast_timestamp"].max(),
    )

    print("\n--- Next-hour target distribution ---")

    target_counts = forecast_df[
        "target_is_degraded_next_1h"
    ].value_counts().sort_index()

    print(target_counts)

    positive_rate = forecast_df[
        "target_is_degraded_next_1h"
    ].mean()

    print(
        "Next-hour positive rate:",
        f"{positive_rate * 100:.4f}%",
    )

    # Confirm every retained row has a forecast exactly one hour later.
    exact_gap_check = (
        forecast_df["forecast_timestamp"]
        - forecast_df["timestamp"]
    ).eq(pd.Timedelta(hours=1)).all()

    print(
        "All forecast targets exactly one hour ahead:",
        exact_gap_check,
    )

    # Save the prepared development dataset.
    forecast_df.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="zstd",
    )

    print(f"\nSaved forecast dataset to:\n{OUTPUT_PATH}")
    print("\nForecast dataset preparation completed.")


if __name__ == "__main__":
    main()