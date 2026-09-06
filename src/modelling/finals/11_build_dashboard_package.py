from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# 1. Locate repository
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None

for candidate in [
    SCRIPT_PATH.parent,
    *SCRIPT_PATH.parents,
]:
    if (candidate / "data" / "outputs").exists():
        ROOT = candidate
        break

if ROOT is None:
    raise RuntimeError(
        "Could not locate repository root."
    )


DATA_PATH = (
    ROOT
    / "data"
    / "outputs"
    / "cell_quarter_dataset.parquet"
)

GEOMETRY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "grid_geometry.geojson"
)

PREDICTION_PATH = (
    ROOT
    / "outputs"
    / "finals_quarterly"
    / "09_final_test_predictions.parquet"
)

EXPLANATION_PATH = (
    ROOT
    / "outputs"
    / "finals_quarterly"
    / "10_final_test_local_explanations.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "finals_dashboard"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Load source data
# ============================================================

df = pd.read_parquet(
    DATA_PATH
)

pred = pd.read_parquet(
    PREDICTION_PATH
)

explanations = pd.read_parquet(
    EXPLANATION_PATH
)


print("=" * 80)
print(
    "GeoQoE Pulse Finals — "
    "Dashboard Package Builder"
)
print("=" * 80)

print(
    f"\nFinal-test predictions: "
    f"{len(pred):,}"
)

print(
    f"Local explanations:     "
    f"{len(explanations):,}"
)


# ============================================================
# 3. Current-quarter contextual data
# ============================================================

context_columns = [

    "h3_index",

    "quarter_index",
    "quarter_label",

    # Network
    "ookla_dl_mbps",
    "ookla_ul_mbps",
    "ookla_rtt_ms",
    "ookla_tests",
    "dl_percentile_in_quarter",
    "below_area_norm_rate_to_date",
    "rolling_mean_4q_dl",
    "lag_1q_dl",

    # Activity / population
    "viirs_rad_quarter",
    "viirs_rad_yoy_log_ratio",
    "district_population",
    "population_density_per_km2",
    "population_day",

    # Geography / infrastructure
    "tower_density_per_km2",
    "dist_to_nearest_tower_m",
    "building_density_per_km2",
    "built_up_ratio",
    "landcover_built_up_pct",
    "road_density_km_per_km2",
    "intersection_density_per_km2",
    "dist_to_nearest_transit_m",
]


current = df[
    df["quarter_index"] == 8104
][context_columns].copy()


# ============================================================
# 4. Merge prediction + explanations + context
# ============================================================

dashboard = pred[
    [
        "h3_index",
        "quarter_index",
        "quarter_label",
        "onset_target",
        "xgb_full",
        "risk_rank",
        "risk_band",
    ]
].copy()


dashboard = dashboard.merge(

    explanations[
        [
            "h3_index",

            "risk_driver_1_label",
            "risk_driver_1_value",
            "risk_driver_1_family",

            "risk_driver_2_label",
            "risk_driver_2_value",
            "risk_driver_2_family",

            "risk_driver_3_label",
            "risk_driver_3_value",
            "risk_driver_3_family",

            "protective_1_label",
            "protective_1_value",
            "protective_1_family",

            "protective_2_label",
            "protective_2_value",
            "protective_2_family",
        ]
    ],

    on="h3_index",
    how="left",
)


dashboard = dashboard.merge(

    current,

    on=[
        "h3_index",
        "quarter_index",
        "quarter_label",
    ],

    how="left",
)


# ============================================================
# 5. Human-readable status fields
# ============================================================

dashboard[
    "current_status"
] = "currently_not_underperforming"


dashboard[
    "forecast_for_quarter"
] = "2026Q2"


dashboard[
    "actual_next_quarter_status"
] = np.where(

    dashboard[
        "onset_target"
    ] == 1,

    "underperforming",

    "not_underperforming",
)


# ============================================================
# 6. Time Machine correctness category
#
# "Flagged" = High or Elevated
#            = Top 200
# ============================================================

def correctness_category(row):

    flagged = (
        row["risk_rank"] <= 200
    )

    actual = (
        row["onset_target"] == 1
    )

    if flagged and actual:
        return "correctly_anticipated"

    if flagged and not actual:
        return "false_alert"

    if (not flagged) and actual:
        return "missed"

    return "correctly_lower_risk"


dashboard[
    "time_machine_result"
] = dashboard.apply(
    correctness_category,
    axis=1,
)


# ============================================================
# 7. Relative risk position
#
# NOT a probability.
# Example:
# 99 = higher ranked than ~99% of eligible cells.
# ============================================================

n_cells = len(
    dashboard
)


dashboard[
    "risk_percentile"
] = (

    100.0

    * (
        n_cells
        - dashboard["risk_rank"]
    )

    / max(
        n_cells - 1,
        1,
    )
)


dashboard[
    "risk_percentile"
] = dashboard[
    "risk_percentile"
].round(1)


# ============================================================
# 8. Measurement-support label
#
# Based on training distribution observed in Checkpoint 7:
#
# <20     = limited support
# 20–149  = typical support
# >=150   = strong support
#
# This is relative measurement support,
# NOT a guarantee of accuracy.
# ============================================================

def measurement_support(
    tests
):

    if pd.isna(tests):
        return "unknown"

    if tests < 20:
        return "limited"

    if tests < 150:
        return "typical"

    return "strong"


dashboard[
    "measurement_support"
] = (

    dashboard[
        "ookla_tests"
    ]

    .apply(
        measurement_support
    )
)


# ============================================================
# 9. Friendly risk labels
# ============================================================

RISK_LABELS = {

    "high":
        "High emerging risk",

    "elevated":
        "Elevated emerging risk",

    "lower":
        "Lower emerging risk",
}


dashboard[
    "risk_band_label"
] = (

    dashboard[
        "risk_band"
    ]

    .map(
        RISK_LABELS
    )
)


# ============================================================
# 10. Internal score warning
# ============================================================

dashboard[
    "score_is_probability"
] = False


dashboard[
    "interpretation_note"
] = (
    "Model contributors explain which inputs "
    "influenced the emerging-risk ranking. "
    "They do not establish the cause of "
    "future network degradation."
)


# ============================================================
# 11. Console summary
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "DASHBOARD RISK SUMMARY"
)

print("=" * 80)


summary = (

    dashboard

    .groupby(
        "risk_band"
    )

    .agg(

        cells=(
            "h3_index",
            "count",
        ),

        actual_events=(
            "onset_target",
            "sum",
        ),

        actual_event_rate=(
            "onset_target",
            "mean",
        ),
    )

    .reset_index()
)


print(
    summary.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "TIME MACHINE RESULT SUMMARY"
)

print("=" * 80)


time_summary = (

    dashboard[
        "time_machine_result"
    ]

    .value_counts()

    .rename_axis(
        "result"
    )

    .reset_index(
        name="cells"
    )
)


print(
    time_summary.to_string(
        index=False
    )
)


print(
    "\nMeasurement support:"
)

print(
    dashboard[
        "measurement_support"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 12. Save tabular dashboard payload
# ============================================================

CSV_PATH = (
    OUTPUT_DIR
    / "2026Q1_to_2026Q2_cells.csv"
)


PARQUET_PATH = (
    OUTPUT_DIR
    / "2026Q1_to_2026Q2_cells.parquet"
)


dashboard.to_csv(
    CSV_PATH,
    index=False,
)


dashboard.to_parquet(
    PARQUET_PATH,
    index=False,
)


# ============================================================
# 13. Merge with grid geometry
#
# Use Python's built-in JSON library so we do not
# require geopandas.
# ============================================================

if not GEOMETRY_PATH.exists():

    raise FileNotFoundError(
        f"Geometry file not found: "
        f"{GEOMETRY_PATH}"
    )


with open(
    GEOMETRY_PATH,
    "r",
    encoding="utf-8",
) as f:

    geojson = json.load(f)


lookup = (

    dashboard

    .set_index(
        "h3_index"
    )

    .replace(
        {
            np.nan: None
        }
    )

    .to_dict(
        orient="index"
    )
)


matched = 0
output_features = []


for feature in geojson[
    "features"
]:

    properties = feature.get(
        "properties",
        {}
    )

    h3 = properties.get(
        "h3_index"
    )


    if h3 not in lookup:
        continue


    dashboard_properties = (
        lookup[h3]
    )


    clean_properties = {}

    for key, value in (
        dashboard_properties.items()
    ):

        if isinstance(
            value,
            np.generic,
        ):
            value = value.item()

        if pd.isna(value):
            value = None

        clean_properties[
            key
        ] = value


    clean_properties[
        "h3_index"
    ] = h3


    output_features.append({

        "type":
            "Feature",

        "geometry":
            feature.get(
                "geometry"
            ),

        "properties":
            clean_properties,
    })


    matched += 1


output_geojson = {

    "type":
        "FeatureCollection",

    "features":
        output_features,
}


GEOJSON_PATH = (
    OUTPUT_DIR
    / "2026Q1_to_2026Q2_cells.geojson"
)


with open(
    GEOJSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output_geojson,
        f,
        ensure_ascii=False,
    )


# ============================================================
# 14. Save frontend metadata
# ============================================================

metadata = {

    "package_version":
        "1.0.0",

    "model_version":
        "finals_quarterly_onset_v1.0.0",

    "predictor_quarter":
        "2026Q1",

    "forecast_quarter":
        "2026Q2",

    "eligible_cells":
        int(
            len(
                dashboard
            )
        ),

    "risk_policy": {

        "high":
            "Top 50 currently-normal cells",

        "elevated":
            "Ranks 51–200",

        "lower":
            "Ranks 201 and below",
    },

    "risk_score_note":
        (
            "The XGBoost output is used for "
            "relative ranking and must not "
            "be displayed as a calibrated "
            "probability."
        ),

    "measurement_support_note":
        (
            "Measurement support is based on "
            "the number of Ookla tests contributing "
            "to the cell-quarter measurement. "
            "It is a relative evidence indicator, "
            "not an accuracy guarantee."
        ),

    "explanation_note":
        (
            "SHAP contributors explain model "
            "behaviour and must not be described "
            "as proven causes of degradation."
        ),

    "time_machine_categories": {

        "correctly_anticipated":
            (
                "Top-200 flagged cell that "
                "actually became underperforming"
            ),

        "false_alert":
            (
                "Top-200 flagged cell that "
                "did not become underperforming"
            ),

        "missed":
            (
                "Lower-risk cell that later "
                "became underperforming"
            ),

        "correctly_lower_risk":
            (
                "Lower-risk cell that remained "
                "outside the underperforming group"
            ),
    },
}


METADATA_PATH = (
    OUTPUT_DIR
    / "dashboard_metadata.json"
)


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 15. Verify geometry coverage
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GEOMETRY MERGE"
)

print("=" * 80)


print(
    f"Dashboard cells: "
    f"{len(dashboard):,}"
)

print(
    f"Geometry matched: "
    f"{matched:,}"
)

print(
    f"Geometry missing: "
    f"{len(dashboard) - matched:,}"
)


# ============================================================
# 16. Finish
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CHECKPOINT 11 COMPLETE"
)

print("=" * 80)


print(
    "\nFrontend package:"
)

print(
    CSV_PATH
)

print(
    PARQUET_PATH
)

print(
    GEOJSON_PATH
)

print(
    METADATA_PATH
)