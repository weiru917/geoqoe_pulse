from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# 1. Locate project
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    if (candidate / "data" / "outputs").exists():
        ROOT = candidate
        break

if ROOT is None:
    raise RuntimeError("Could not locate repository root.")

DATA_PATH = ROOT / "data" / "outputs" / "cell_quarter_dataset.parquet"

OUTPUT_DIR = ROOT / "outputs" / "finals_quarterly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)


# ============================================================
# 2. Fixed modelling task
# ============================================================

TARGET = "target_is_below_area_norm_next_q"

eligible = df[
    (df["target_next_q_valid"] == 1)
    & (df["is_below_area_norm"] == 0)
].copy()


# ============================================================
# 3. QoE / performance-history features
# ============================================================

QOE_HISTORY_FEATURES = [

    # Current real measurements
    "ookla_dl_mbps",
    "ookla_ul_mbps",
    "ookla_rtt_ms",

    # Current relative position
    "dl_percentile_in_quarter",

    # Historical download behaviour
    "lag_1q_dl",
    "lag_4q_dl",
    "rolling_mean_4q_dl",
    "rolling_std_4q_dl",

    # Historical latency behaviour
    "lag_1q_rtt",
    "lag_4q_rtt",
    "rolling_mean_4q_rtt",
    "rolling_std_4q_rtt",

    # Long-term history
    "recurrence_to_date",
    "quarters_observed_to_date",
    "below_area_norm_rate_to_date",

    # Missing-history indicators
    "lag_1q_dl_valid",
    "lag_4q_dl_valid",
]


# ============================================================
# 4. Dynamic activity features
# ============================================================

ACTIVITY_FEATURES = [

    # Quarterly nighttime-light activity
    "viirs_rad_quarter",
    "viirs_rad_yoy_log_ratio",
    "viirs_yoy_is_valid",

    # Official population change
    "district_population",
    "district_population_yoy_growth_pct",
    "district_population_growth_since_2020_pct",

    # Local population
    "population_density_per_km2",
    "population_day",
]


# ============================================================
# 5. Static geographic / infrastructure context
# ============================================================

GEOGRAPHIC_FEATURES = [

    # Tower context
    "tower_density_per_km2",
    "dist_to_nearest_tower_m",

    # Buildings
    "building_density_per_km2",
    "built_up_ratio",
    "avg_building_footprint_m2",

    # Activity / POIs
    "poi_education_count",
    "poi_healthcare_count",
    "poi_retail_food_count",
    "poi_office_count",
    "poi_recreation_count",
    "poi_transport_stop_count",

    # Accessibility
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "dist_to_nearest_transit_m",

    # Land cover
    "landcover_built_up_pct",
    "landcover_tree_cover_pct",
    "landcover_water_pct",

    # Terrain
    "elevation_mean_m",
    "terrain_ruggedness_m",
]


# ============================================================
# 6. Final model groups
# ============================================================

CONTEXT_FEATURES = (
    ACTIVITY_FEATURES
    + GEOGRAPHIC_FEATURES
)

FULL_FEATURES = (
    QOE_HISTORY_FEATURES
    + CONTEXT_FEATURES
)

FEATURE_GROUPS = {
    "qoe_history": QOE_HISTORY_FEATURES,
    "context_only": CONTEXT_FEATURES,
    "full": FULL_FEATURES,
}


# ============================================================
# 7. Explicit forbidden columns
# ============================================================

FORBIDDEN_EXACT = {

    # Future answers
    "target_is_degraded_next_q",
    "target_is_below_area_norm_next_q",
    "target_dl_mbps_next_q",
    "target_next_q_valid",

    # IDs / labels
    "h3_index",
    "h3_parent_res6",
    "quarter_label",

    # Sparse Sunway case-study fields
    "sunway_n_records",
    "sunway_median_dl_mbps",
    "sunway_p90_dl_mbps",
    "sunway_median_ul_mbps",
    "sunway_median_rsrp_dbm",
    "sunway_median_snr_db",
    "sunway_median_ping_ms",
    "sunway_median_jitter_ms",
    "sunway_median_packet_loss_pct",

    # Dashboard/application outputs
    "app_voice_call_ok",
    "app_web_browsing_ok",
    "app_video_call_ok",
    "app_video_streaming_ok",
    "app_online_gaming_ok",
    "app_user_impact_score",
    "app_n_evaluable",
    "app_n_failing",
    "app_worst_failing",
}


# ============================================================
# 8. Audit every selected feature
# ============================================================

print("=" * 80)
print("GeoQoE Pulse Finals — Feature Registry")
print("=" * 80)

print(f"\nEligible modelling rows: {len(eligible):,}")
print(f"Positive events:         {int(eligible[TARGET].sum()):,}")
print(f"Event rate:              {eligible[TARGET].mean():.4%}")


audit_rows = []

for group_name, features in FEATURE_GROUPS.items():

    print("\n" + "=" * 80)
    print(group_name.upper())
    print("=" * 80)

    print(f"Feature count: {len(features)}")

    duplicates = [
        f for f in features
        if features.count(f) > 1
    ]

    if duplicates:
        raise RuntimeError(
            f"Duplicate features in {group_name}: "
            f"{sorted(set(duplicates))}"
        )

    for feature in features:

        if feature not in df.columns:
            raise RuntimeError(
                f"Selected feature missing from dataset: {feature}"
            )

        if feature in FORBIDDEN_EXACT:
            raise RuntimeError(
                f"FORBIDDEN feature accidentally selected: {feature}"
            )

        future_words = [
            "next_q",
            "future",
            "target_",
        ]

        if any(
            word in feature.lower()
            for word in future_words
        ):
            raise RuntimeError(
                f"Potential future leakage selected: {feature}"
            )

        series = eligible[feature]

        numeric = pd.api.types.is_numeric_dtype(series)

        if not numeric:
            raise RuntimeError(
                f"Feature is not numeric: {feature}"
            )

        missing_pct = series.isna().mean() * 100
        unique_values = series.nunique(dropna=True)

        finite_pct = (
            np.isfinite(series.dropna())
            .mean() * 100
            if series.notna().any()
            else 0
        )

        print(
            f"{feature:45s} "
            f"missing={missing_pct:6.2f}% "
            f"unique={unique_values:6d} "
            f"finite={finite_pct:6.2f}%"
        )

        audit_rows.append({
            "group": group_name,
            "feature": feature,
            "missing_pct": missing_pct,
            "unique_values": unique_values,
            "finite_pct": finite_pct,
        })


# ============================================================
# 9. Check extreme correlations
# ============================================================

print("\n" + "=" * 80)
print("HIGH-CORRELATION CHECK — FULL MODEL")
print("=" * 80)

corr = eligible[FULL_FEATURES].corr().abs()

pairs = []

for i, col1 in enumerate(corr.columns):
    for j in range(i + 1, len(corr.columns)):
        col2 = corr.columns[j]

        value = corr.iloc[i, j]

        if pd.notna(value) and value >= 0.95:
            pairs.append((col1, col2, value))

if not pairs:
    print("No feature pairs with |correlation| >= 0.95.")
else:
    for a, b, value in sorted(
        pairs,
        key=lambda x: x[2],
        reverse=True
    ):
        print(f"{value:.4f}  {a}  <->  {b}")


# ============================================================
# 10. Save feature registry
# ============================================================

registry = {
    "task": (
        "Among currently normal cells, predict whether "
        "the cell falls into the bottom 25% of Klang Valley "
        "download performance next quarter."
    ),
    "target": TARGET,
    "eligibility": {
        "target_next_q_valid": 1,
        "is_below_area_norm": 0,
    },
    "feature_groups": FEATURE_GROUPS,
    "forbidden_exact": sorted(FORBIDDEN_EXACT),
}

with open(
    OUTPUT_DIR / "03_feature_registry.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        registry,
        f,
        indent=2,
    )

pd.DataFrame(audit_rows).to_csv(
    OUTPUT_DIR / "03_feature_audit.csv",
    index=False,
)


print("\n" + "=" * 80)
print("CHECKPOINT 3 COMPLETE")
print("=" * 80)

print("\nSaved:")
print(OUTPUT_DIR / "03_feature_registry.json")
print(OUTPUT_DIR / "03_feature_audit.csv")

print("\nNo AI model has been trained yet.")