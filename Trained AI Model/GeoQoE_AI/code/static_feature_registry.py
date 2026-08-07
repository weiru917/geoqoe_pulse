from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
    / "cell_static_qoe_baseline.parquet"
)


# ============================================================
# MODEL TASK
# ============================================================

# We will train a regression model to estimate real
# Ookla download throughput in Mbps.
TARGET_COLUMN = "ookla_dl_mbps"

# This column identifies cells that contain a real Ookla target.
LABEL_STATUS_COLUMN = "has_real_ookla_label"

# This column defines larger spatial groups.
# It will be used for spatial cross-validation,
# but it must not be passed into the model as a feature.
SPATIAL_GROUP_COLUMN = "h3_parent_res6"

# Test count will later be used only as an optional
# confidence/sample weight. It is not a model feature.
SAMPLE_WEIGHT_COLUMN = "ookla_tests"

ID_COLUMN = "h3_index"


# ============================================================
# APPROVED STATIC GIS FEATURES
# ============================================================

INFRASTRUCTURE_FEATURES = [
    "tower_density_per_km2",
    "dist_to_nearest_tower_m",
]

BUILDING_FEATURES = [
    "building_density_per_km2",
    "built_up_ratio",
    "avg_building_footprint_m2",
]

POI_FEATURES = [
    "poi_education_count",
    "poi_healthcare_count",
    "poi_retail_food_count",
    "poi_office_count",
    "poi_recreation_count",
    "poi_transport_stop_count",
]

ACCESSIBILITY_FEATURES = [
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "dist_to_nearest_transit_m",
]

POPULATION_ACTIVITY_FEATURES = [
    "population_density_per_km2",

    # Although the current column is named population_day,
    # it should be interpreted cautiously as a daytime
    # activity/exposure proxy rather than verified daytime
    # residential population.
    "population_day",
]


APPROVED_STATIC_FEATURES = (
    INFRASTRUCTURE_FEATURES
    + BUILDING_FEATURES
    + POI_FEATURES
    + ACCESSIBILITY_FEATURES
    + POPULATION_ACTIVITY_FEATURES
)


# These candidates are intentionally excluded from the
# first model because another approved feature already
# expresses approximately the same concept.
REDUNDANT_CANDIDATES_EXCLUDED = {
    "opencellid_record_count": (
        "Almost perfectly correlated with "
        "tower_density_per_km2; retain the "
        "area-normalized density feature."
    ),
    "tower_count": (
        "Use tower_density_per_km2 instead."
    ),
    "building_count": (
        "Use building_density_per_km2 instead."
    ),
    "total_building_area_m2": (
        "Use built_up_ratio instead."
    ),
    "road_length_m": (
        "Use road_density_km_per_km2 instead."
    ),
    "intersection_count": (
        "Use intersection_density_per_km2 instead."
    ),
    "population_night": (
        "Use population_density_per_km2 instead."
    ),
}


# These columns must never enter the static model.
FORBIDDEN_MODEL_FEATURES = {
    ID_COLUMN,
    SPATIAL_GROUP_COLUMN,
    LABEL_STATUS_COLUMN,
    TARGET_COLUMN,
    "ookla_rtt_ms",
    "ookla_tests",
    "ookla_devices",
    "ookla_quarter_test_count",
    "ookla_quarter_device_count",
    "qoe_risk_label",
    "qoe_level_is_ookla_anchored",
    "qoe_baseline_source",
    "tower_is_synthetic",
    "tower_source",
    "building_is_synthetic",
    "osm_is_synthetic",
    "population_is_synthetic",
}


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static Feature Registry Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Static dataset not found:\n{DATA_PATH}"
        )

    columns_to_load = [
        ID_COLUMN,
        SPATIAL_GROUP_COLUMN,
        LABEL_STATUS_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        TARGET_COLUMN,
        *APPROVED_STATIC_FEATURES,
    ]

    # Remove accidental repeated names while preserving order.
    columns_to_load = list(
        dict.fromkeys(columns_to_load)
    )

    print(f"\nLoading required columns from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=columns_to_load,
    )

    # The model is trained only using cells that have
    # real Ookla labels.
    labelled_df = df[
        df[LABEL_STATUS_COLUMN].astype(bool)
    ].copy()

    print("\n--- Dataset summary ---")
    print("All grid cells:", f"{len(df):,}")
    print(
        "Real-labelled cells:",
        f"{len(labelled_df):,}",
    )
    print(
        "Spatial parent groups in labelled data:",
        f"{labelled_df[SPATIAL_GROUP_COLUMN].nunique():,}",
    )

    print("\n--- Feature-group counts ---")
    print(
        "Infrastructure features:",
        len(INFRASTRUCTURE_FEATURES),
    )
    print(
        "Building features:",
        len(BUILDING_FEATURES),
    )
    print(
        "POI features:",
        len(POI_FEATURES),
    )
    print(
        "Accessibility features:",
        len(ACCESSIBILITY_FEATURES),
    )
    print(
        "Population/activity features:",
        len(POPULATION_ACTIVITY_FEATURES),
    )
    print(
        "Total approved static features:",
        len(APPROVED_STATIC_FEATURES),
    )

    # --------------------------------------------------------
    # 1. Check for duplicate feature names.
    # --------------------------------------------------------

    duplicated_features = sorted(
        {
            feature
            for feature in APPROVED_STATIC_FEATURES
            if APPROVED_STATIC_FEATURES.count(feature) > 1
        }
    )

    print("\n--- Duplicate-feature check ---")

    if duplicated_features:
        print("Duplicate approved feature names found:")

        for feature in duplicated_features:
            print(f"  - {feature}")
    else:
        print("No duplicate approved feature names found.")

    # --------------------------------------------------------
    # 2. Check that required columns exist.
    # --------------------------------------------------------

    dataset_columns = set(df.columns)

    missing_features = [
        feature
        for feature in APPROVED_STATIC_FEATURES
        if feature not in dataset_columns
    ]

    print("\n--- Approved-feature presence check ---")

    if missing_features:
        print("Missing approved features:")

        for feature in missing_features:
            print(f"  - {feature}")
    else:
        print("All approved static features are present.")

    # --------------------------------------------------------
    # 3. Check for forbidden overlap.
    # --------------------------------------------------------

    forbidden_overlap = sorted(
        set(APPROVED_STATIC_FEATURES)
        .intersection(FORBIDDEN_MODEL_FEATURES)
    )

    print("\n--- Leakage and provenance check ---")

    if forbidden_overlap:
        print(
            "Forbidden columns found inside the "
            "approved feature registry:"
        )

        for feature in forbidden_overlap:
            print(f"  - {feature}")
    else:
        print(
            "No identifiers, targets, provenance fields, "
            "or Ookla observation fields are model features."
        )

    # --------------------------------------------------------
    # 4. Inspect every approved feature.
    # --------------------------------------------------------

    print("\n--- Approved-feature data checks ---")

    non_numeric_features = []
    missing_value_features = []
    constant_features = []
    non_finite_features = []

    for feature in APPROVED_STATIC_FEATURES:
        if feature not in labelled_df.columns:
            continue

        series = labelled_df[feature]

        is_numeric = pd.api.types.is_numeric_dtype(
            series.dtype
        )

        missing_count = int(series.isna().sum())

        unique_count = int(
            series.nunique(dropna=True)
        )

        if is_numeric:
            finite_check = bool(
                np.isfinite(
                    series.to_numpy(
                        dtype="float64",
                        copy=False,
                    )
                ).all()
            )
        else:
            finite_check = False

        print(
            f"{feature:<42}"
            f"dtype={str(series.dtype):<10} "
            f"missing={missing_count:>4} "
            f"unique={unique_count:>5} "
            f"finite={finite_check}"
        )

        if not is_numeric:
            non_numeric_features.append(feature)

        if missing_count > 0:
            missing_value_features.append(feature)

        if unique_count <= 1:
            constant_features.append(feature)

        if is_numeric and not finite_check:
            non_finite_features.append(feature)

    # --------------------------------------------------------
    # 5. Check target.
    # --------------------------------------------------------

    print("\n--- Regression-target check ---")

    target = labelled_df[TARGET_COLUMN]

    target_is_numeric = pd.api.types.is_numeric_dtype(
        target.dtype
    )

    target_missing_count = int(
        target.isna().sum()
    )

    target_is_finite = bool(
        np.isfinite(
            target.to_numpy(
                dtype="float64",
                copy=False,
            )
        ).all()
    )

    print("Target column:", TARGET_COLUMN)
    print("Target is numeric:", target_is_numeric)
    print(
        "Target missing values:",
        f"{target_missing_count:,}",
    )
    print(
        "All target values are finite:",
        target_is_finite,
    )
    print(
        "Target minimum:",
        f"{target.min():.4f} Mbps",
    )
    print(
        "Target median:",
        f"{target.median():.4f} Mbps",
    )
    print(
        "Target maximum:",
        f"{target.max():.4f} Mbps",
    )

    # --------------------------------------------------------
    # 6. Check spatial groups.
    # --------------------------------------------------------

    print("\n--- Spatial-group check ---")

    group_sizes = (
        labelled_df
        .groupby(SPATIAL_GROUP_COLUMN)
        .size()
    )

    print(
        "Number of spatial groups:",
        f"{len(group_sizes):,}",
    )
    print(
        "Smallest labelled group:",
        f"{group_sizes.min():,} cells",
    )
    print(
        "Median labelled group:",
        f"{group_sizes.median():.1f} cells",
    )
    print(
        "Largest labelled group:",
        f"{group_sizes.max():,} cells",
    )

    groups_with_fewer_than_five_cells = int(
        (group_sizes < 5).sum()
    )

    print(
        "Groups with fewer than 5 labelled cells:",
        groups_with_fewer_than_five_cells,
    )

    # --------------------------------------------------------
    # 7. Check potential high correlations.
    # --------------------------------------------------------

    print("\n--- High-correlation warning check ---")

    correlation_matrix = (
        labelled_df[APPROVED_STATIC_FEATURES]
        .corr()
    )

    high_correlation_pairs = []

    for left_index, left_feature in enumerate(
        APPROVED_STATIC_FEATURES
    ):
        for right_feature in APPROVED_STATIC_FEATURES[
            left_index + 1:
        ]:
            correlation = float(
                correlation_matrix.loc[
                    left_feature,
                    right_feature,
                ]
            )

            if abs(correlation) >= 0.95:
                high_correlation_pairs.append(
                    (
                        left_feature,
                        right_feature,
                        correlation,
                    )
                )

    if not high_correlation_pairs:
        print(
            "No approved feature pairs have "
            "|correlation| >= 0.95."
        )
    else:
        print(
            "Highly correlated approved feature pairs "
            "were found:"
        )

        for (
            left_feature,
            right_feature,
            correlation,
        ) in high_correlation_pairs:
            print(
                f"  {left_feature} <-> "
                f"{right_feature}: "
                f"{correlation:.4f}"
            )

        print(
            "This is a warning only. We will decide whether "
            "to remove any pair before model training."
        )

    # --------------------------------------------------------
    # 8. Display deliberately excluded candidates.
    # --------------------------------------------------------

    print("\n--- Deliberately excluded redundant candidates ---")

    for feature, reason in (
        REDUNDANT_CANDIDATES_EXCLUDED.items()
    ):
        print(f"{feature}:")
        print(f"  {reason}")

    # --------------------------------------------------------
    # 9. Final result.
    # --------------------------------------------------------

    checks_passed = (
        len(labelled_df) > 0
        and len(APPROVED_STATIC_FEATURES) > 0
        and not duplicated_features
        and not missing_features
        and not forbidden_overlap
        and not non_numeric_features
        and not missing_value_features
        and not constant_features
        and not non_finite_features
        and target_is_numeric
        and target_missing_count == 0
        and target_is_finite
        and labelled_df[
            SPATIAL_GROUP_COLUMN
        ].nunique() >= 5
    )

    print("\n--- Final result ---")

    if checks_passed:
        print("Static feature registry check PASSED.")
        print(
            "The real-data regression workflow can use "
            f"{len(APPROVED_STATIC_FEATURES)} approved GIS features."
        )
        print(
            "h3_parent_res6 will be used only for spatial grouping."
        )
        print(
            "ookla_tests will be used later only as an "
            "optional confidence/sample weight."
        )
    else:
        print("Static feature registry check FAILED.")

        if non_numeric_features:
            print(
                "Non-numeric approved features:",
                non_numeric_features,
            )

        if missing_value_features:
            print(
                "Approved features with missing values:",
                missing_value_features,
            )

        if constant_features:
            print(
                "Constant approved features:",
                constant_features,
            )

        if non_finite_features:
            print(
                "Approved features with non-finite values:",
                non_finite_features,
            )


if __name__ == "__main__":
    main()