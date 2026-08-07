from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)


# These are the features approved for our first development model.
#
# They are grouped so that we can later test which types of information
# actually improve the forecast.

TIME_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_peak_period",
]

CURRENT_QOE_FEATURES = [
    "dl_median",
    "rtt_median",
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
    "qoe_trend_1h",
]

STATIC_GIS_FEATURES = [
    "tower_count",
    "tower_density_per_km2",
    "dist_to_nearest_tower_m",
    "building_count",
    "built_up_ratio",
    "building_density_per_km2",
    "poi_education_count",
    "poi_healthcare_count",
    "poi_retail_food_count",
    "poi_office_count",
    "poi_recreation_count",
    "poi_transport_stop_count",
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "dist_to_nearest_transit_m",
    "population_night",
    "population_density_per_km2",
]

SPATIAL_QOE_FEATURES = [
    "spatial_lag_dl_median",
    "spatial_var_dl_median",
    "spatial_lag_rtt_median",
    "spatial_var_rtt_median",
    "spatial_rolling_mean_24h_dl_median",
]


APPROVED_FEATURES = (
    TIME_FEATURES
    + CURRENT_QOE_FEATURES
    + HISTORICAL_QOE_FEATURES
    + STATIC_GIS_FEATURES
    + SPATIAL_QOE_FEATURES
)


TARGET_COLUMN = "target_is_degraded_next_1h"


# These columns are deliberately not used in the first model.
#
# Examples:
# - IDs and timestamps
# - current and future target variables
# - custom QoE score and MOS
# - demand index that is currently being revised
# - categorical hotspot strings
# - synthetic/confidence indicators that are better used for evaluation
EXCLUDED_COLUMNS = [
    "row_id",
    "h3_index",
    "timestamp",
    "forecast_timestamp",
    "target_is_degraded",
    "target_is_degraded_next_1h",
    "target_qoe_score",
    "target_mos",
    "target_degradation_class",
    "demand_pressure_base",
    "demand_pressure_index_pre_spatial",
    "demand_pressure_index",
    "hotspot_category_rtt_median",
    "local_moran_quadrant_rtt_median",
    "qoe_is_synthetic",
    "qoe_confidence_score",
    "test_sample_count",
    "unique_device_count",
]


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Feature Registry Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    print(f"\nReading dataset columns from:\n{DATA_PATH}")

    # Reading only the schema is enough for this check.
    df = pd.read_parquet(DATA_PATH)

    dataset_columns = set(df.columns)
    approved_set = set(APPROVED_FEATURES)

    missing_features = [
        feature
        for feature in APPROVED_FEATURES
        if feature not in dataset_columns
    ]

    duplicate_features = [
        feature
        for feature in APPROVED_FEATURES
        if APPROVED_FEATURES.count(feature) > 1
    ]

    print("\n--- Feature-group counts ---")
    print("Time features:", len(TIME_FEATURES))
    print("Current QoE features:", len(CURRENT_QOE_FEATURES))
    print("Historical QoE features:", len(HISTORICAL_QOE_FEATURES))
    print("Static GIS features:", len(STATIC_GIS_FEATURES))
    print("Spatial QoE features:", len(SPATIAL_QOE_FEATURES))
    print("Total approved features:", len(APPROVED_FEATURES))

    print("\n--- Required target check ---")
    if TARGET_COLUMN in dataset_columns:
        print(f"Target is present: {TARGET_COLUMN}")
    else:
        print(f"Target is missing: {TARGET_COLUMN}")

    print("\n--- Approved-feature check ---")
    if missing_features:
        print("Missing approved features:")
        for feature in missing_features:
            print(f"  - {feature}")
    else:
        print("All approved features are present.")

    print("\n--- Duplicate-feature check ---")
    if duplicate_features:
        print("Duplicate feature names found:")
        for feature in sorted(set(duplicate_features)):
            print(f"  - {feature}")
    else:
        print("No duplicate feature names found.")

    print("\n--- Data type check ---")

    non_numeric_features = []

    for feature in APPROVED_FEATURES:
        if feature in df.columns:
            dtype = df[feature].dtype
            is_numeric = pd.api.types.is_numeric_dtype(dtype)

            print(
                f"{feature:<45} "
                f"{str(dtype):<15} "
                f"{'numeric' if is_numeric else 'NON-NUMERIC'}"
            )

            if not is_numeric:
                non_numeric_features.append(feature)

    print("\n--- Final result ---")

    checks_passed = (
        TARGET_COLUMN in dataset_columns
        and not missing_features
        and not duplicate_features
        and not non_numeric_features
    )

    if checks_passed:
        print("Feature registry check PASSED.")
        print(
            "The first development model can use "
            f"{len(APPROVED_FEATURES)} approved numeric features."
        )
    else:
        print("Feature registry check FAILED.")

        if non_numeric_features:
            print(
                "Non-numeric approved features:",
                non_numeric_features,
            )


if __name__ == "__main__":
    main()