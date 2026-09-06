from pathlib import Path
import json

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier


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

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "finals_quarterly"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "finals_quarterly"
    / "final_xgb_full_v1.0.0.json"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "09_final_test_predictions.parquet"
)


# ============================================================
# 2. Load data / registry
# ============================================================

df = pd.read_parquet(DATA_PATH)

with open(
    OUTPUT_DIR / "03_feature_registry.json",
    "r",
    encoding="utf-8",
) as f:
    registry = json.load(f)


FEATURES = (
    registry[
        "feature_groups"
    ]["full"]
)

QOE_FEATURES = set(
    registry[
        "feature_groups"
    ]["qoe_history"]
)


# ============================================================
# 3. Define interpretable feature families
# ============================================================

ACTIVITY_FEATURES = {

    "viirs_rad_quarter",
    "viirs_rad_yoy_log_ratio",
    "viirs_yoy_is_valid",

    "district_population",
    "district_population_yoy_growth_pct",
    "district_population_growth_since_2020_pct",

    "population_density_per_km2",
    "population_day",
}


def feature_family(feature):

    if feature in QOE_FEATURES:
        return "QoE / performance history"

    if feature in ACTIVITY_FEATURES:
        return "Activity / population context"

    return "Geographic / infrastructure context"


# ============================================================
# 4. Friendly feature names
# ============================================================

FEATURE_LABELS = {

    # QoE / history
    "ookla_dl_mbps":
        "Current download speed",

    "ookla_ul_mbps":
        "Current upload speed",

    "ookla_rtt_ms":
        "Current latency",

    "dl_percentile_in_quarter":
        "Current citywide download percentile",

    "lag_1q_dl":
        "Previous-quarter download speed",

    "lag_4q_dl":
        "Download speed one year ago",

    "rolling_mean_4q_dl":
        "Recent average download speed",

    "rolling_std_4q_dl":
        "Recent download-speed variability",

    "lag_1q_rtt":
        "Previous-quarter latency",

    "lag_4q_rtt":
        "Latency one year ago",

    "rolling_mean_4q_rtt":
        "Recent average latency",

    "rolling_std_4q_rtt":
        "Recent latency variability",

    "recurrence_to_date":
        "Past underperformance recurrence",

    "quarters_observed_to_date":
        "Amount of historical observation",

    "below_area_norm_rate_to_date":
        "Historical underperformance rate",

    "lag_1q_dl_valid":
        "Availability of previous-quarter history",

    "lag_4q_dl_valid":
        "Availability of one-year history",

    # Activity / population
    "viirs_rad_quarter":
        "Nighttime-light activity",

    "viirs_rad_yoy_log_ratio":
        "Year-on-year activity change",

    "viirs_yoy_is_valid":
        "Availability of activity-change history",

    "district_population":
        "District population",

    "district_population_yoy_growth_pct":
        "District population growth",

    "district_population_growth_since_2020_pct":
        "District population growth since 2020",

    "population_density_per_km2":
        "Population density",

    "population_day":
        "Estimated local population",

    # Infrastructure / geography
    "tower_density_per_km2":
        "Nearby tower density",

    "dist_to_nearest_tower_m":
        "Distance to nearest tower",

    "building_density_per_km2":
        "Building density",

    "built_up_ratio":
        "Built-up intensity",

    "avg_building_footprint_m2":
        "Average building footprint",

    "poi_education_count":
        "Education facilities",

    "poi_healthcare_count":
        "Healthcare facilities",

    "poi_retail_food_count":
        "Retail and food activity",

    "poi_office_count":
        "Office activity",

    "poi_recreation_count":
        "Recreation activity",

    "poi_transport_stop_count":
        "Transport stops",

    "road_density_km_per_km2":
        "Road density",

    "intersection_density_per_km2":
        "Road-intersection density",

    "dist_to_nearest_transit_m":
        "Distance to public transport",

    "landcover_built_up_pct":
        "Built-up land coverage",

    "landcover_tree_cover_pct":
        "Tree-cover percentage",

    "landcover_water_pct":
        "Water-cover percentage",

    "elevation_mean_m":
        "Average elevation",

    "terrain_ruggedness_m":
        "Terrain ruggedness",
}


# ============================================================
# 5. Reconstruct FINAL TEST population
# ============================================================

test = df[
    (df["quarter_index"] == 8104)
    & (df["target_next_q_valid"] == 1)
    & (df["is_below_area_norm"] == 0)
].copy()


print("=" * 80)
print(
    "GeoQoE Pulse Finals — "
    "Final Model Interpretation"
)
print("=" * 80)

print(
    f"\nCells to explain: "
    f"{len(test):,}"
)

print(
    "Predictor quarter: 2026Q1"
)

print(
    "Outcome quarter:   2026Q2"
)


# ============================================================
# 6. Load frozen FINAL model
# ============================================================

model = XGBClassifier()

model.load_model(
    MODEL_PATH
)

booster = model.get_booster()


# ============================================================
# 7. Verify feature order
# ============================================================

model_features = booster.feature_names

if (
    model_features is not None
    and model_features != FEATURES
):
    raise RuntimeError(
        "Feature order in saved model "
        "does not match feature registry."
    )


# ============================================================
# 8. Recalculate risk scores
# ============================================================

risk_scores = model.predict_proba(
    test[FEATURES]
)[:, 1]


# ============================================================
# 9. Compare with Checkpoint 9 saved predictions
# ============================================================

saved_predictions = pd.read_parquet(
    PREDICTION_PATH
)


comparison = (
    test[
        [
            "h3_index",
            "quarter_index",
        ]
    ]
    .copy()
)

comparison[
    "recalculated_score"
] = risk_scores


comparison = comparison.merge(

    saved_predictions[
        [
            "h3_index",
            "quarter_index",
            "xgb_full",
        ]
    ],

    on=[
        "h3_index",
        "quarter_index",
    ],

    how="left",
)


max_score_difference = (
    comparison[
        "recalculated_score"
    ]
    - comparison[
        "xgb_full"
    ]
).abs().max()


print(
    "\nMaximum difference from "
    "Checkpoint 9 prediction:"
)

print(
    f"{max_score_difference:.12f}"
)


if max_score_difference > 1e-6:

    raise RuntimeError(
        "Final model predictions do not "
        "match Checkpoint 9."
    )


# ============================================================
# 10. Calculate SHAP contributions
#
# XGBoost pred_contribs gives one contribution
# per feature + one final bias column.
#
# Contributions are in raw model-output space
# (log odds), NOT probability percentage points.
# ============================================================

dmatrix = xgb.DMatrix(

    test[FEATURES],

    feature_names=FEATURES,
)


contributions = booster.predict(

    dmatrix,

    pred_contribs=True,
)


expected_columns = (
    len(FEATURES) + 1
)

if contributions.shape[1] != expected_columns:

    raise RuntimeError(
        "Unexpected SHAP contribution shape."
    )


shap_values = contributions[
    :, :-1
]

base_values = contributions[
    :, -1
]


# ============================================================
# 11. SHAP mathematical sanity check
# ============================================================

raw_margin = (
    shap_values.sum(axis=1)
    + base_values
)


reconstructed_probability = (
    1.0
    / (
        1.0
        + np.exp(
            -raw_margin
        )
    )
)


max_shap_probability_difference = (
    np.abs(
        reconstructed_probability
        - risk_scores
    )
    .max()
)


print(
    "\nMaximum SHAP reconstruction difference:"
)

print(
    f"{max_shap_probability_difference:.12f}"
)


if max_shap_probability_difference > 1e-5:

    raise RuntimeError(
        "SHAP contributions do not "
        "reconstruct model predictions."
    )


# ============================================================
# 12. Global SHAP importance
#
# Mean absolute SHAP:
# how strongly a feature tends to influence predictions.
#
# NOT causal importance.
# ============================================================

global_rows = []


for index, feature in enumerate(
    FEATURES
):

    feature_shap = (
        shap_values[
            :, index
        ]
    )

    global_rows.append({

        "feature":
            feature,

        "friendly_name":
            FEATURE_LABELS.get(
                feature,
                feature,
            ),

        "family":
            feature_family(
                feature
            ),

        "mean_abs_shap":
            float(
                np.mean(
                    np.abs(
                        feature_shap
                    )
                )
            ),

        "mean_shap":
            float(
                np.mean(
                    feature_shap
                )
            ),
    })


global_df = (
    pd.DataFrame(
        global_rows
    )
    .sort_values(
        "mean_abs_shap",
        ascending=False,
    )
)


total_global_importance = (
    global_df[
        "mean_abs_shap"
    ].sum()
)


global_df[
    "importance_pct"
] = (

    global_df[
        "mean_abs_shap"
    ]

    / total_global_importance

    * 100.0
)


# ============================================================
# 13. Importance by broad information family
# ============================================================

family_df = (

    global_df

    .groupby(
        "family"
    )

    .agg(
        total_mean_abs_shap=(
            "mean_abs_shap",
            "sum",
        )
    )

    .reset_index()
)


family_total = (
    family_df[
        "total_mean_abs_shap"
    ].sum()
)


family_df[
    "importance_share_pct"
] = (

    family_df[
        "total_mean_abs_shap"
    ]

    / family_total

    * 100.0
)


family_df = (
    family_df
    .sort_values(
        "importance_share_pct",
        ascending=False,
    )
)


# ============================================================
# 14. High-risk-only global importance
# ============================================================

risk_rank = (
    pd.Series(
        risk_scores,
        index=test.index,
    )
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)


high_mask = (
    risk_rank <= 50
).values


high_rows = []


for index, feature in enumerate(
    FEATURES
):

    high_rows.append({

        "feature":
            feature,

        "friendly_name":
            FEATURE_LABELS.get(
                feature,
                feature,
            ),

        "family":
            feature_family(
                feature
            ),

        "mean_abs_shap_high_risk":
            float(
                np.mean(
                    np.abs(
                        shap_values[
                            high_mask,
                            index,
                        ]
                    )
                )
            ),
    })


high_importance_df = (

    pd.DataFrame(
        high_rows
    )

    .sort_values(
        "mean_abs_shap_high_risk",
        ascending=False,
    )
)


# ============================================================
# 15. Helper for readable values
# ============================================================

def format_feature_value(
    feature,
    value,
):

    if pd.isna(value):
        return "not available"

    if feature in {
        "ookla_dl_mbps",
        "ookla_ul_mbps",
        "lag_1q_dl",
        "lag_4q_dl",
        "rolling_mean_4q_dl",
        "rolling_std_4q_dl",
    }:
        return f"{value:.1f} Mbps"

    if feature in {
        "ookla_rtt_ms",
        "lag_1q_rtt",
        "lag_4q_rtt",
        "rolling_mean_4q_rtt",
        "rolling_std_4q_rtt",
    }:
        return f"{value:.1f} ms"

    if feature == "dl_percentile_in_quarter":
        return f"{value:.1f} percentile"

    if feature in {
        "district_population_yoy_growth_pct",
        "district_population_growth_since_2020_pct",
        "landcover_built_up_pct",
        "landcover_tree_cover_pct",
        "landcover_water_pct",
    }:
        return f"{value:.1f}%"

    if feature in {
        "dist_to_nearest_tower_m",
        "dist_to_nearest_transit_m",
        "elevation_mean_m",
        "terrain_ruggedness_m",
    }:
        return f"{value:.1f} m"

    if feature in {
        "lag_1q_dl_valid",
        "lag_4q_dl_valid",
        "viirs_yoy_is_valid",
    }:
        return (
            "available"
            if value >= 0.5
            else "not available"
        )

    if abs(value) >= 1000:
        return f"{value:,.0f}"

    return f"{value:.2f}"


# ============================================================
# 16. Generate local explanations
#
# Positive SHAP:
# pushes model toward HIGHER emerging risk.
#
# Negative SHAP:
# pushes model toward LOWER emerging risk.
# ============================================================

saved_lookup = (

    saved_predictions

    .set_index(
        "h3_index"
    )
)


local_rows = []


for row_position, (
    dataframe_index,
    row
) in enumerate(
    test.iterrows()
):

    shap_row = (
        shap_values[
            row_position
        ]
    )


    feature_contributions = [

        (
            FEATURES[i],
            float(
                shap_row[i]
            ),
        )

        for i in range(
            len(FEATURES)
        )
    ]


    positive = sorted(

        [
            item
            for item
            in feature_contributions
            if item[1] > 0
        ],

        key=lambda x: x[1],

        reverse=True,
    )


    negative = sorted(

        [
            item
            for item
            in feature_contributions
            if item[1] < 0
        ],

        key=lambda x: x[1],
    )


    h3 = row[
        "h3_index"
    ]


    saved_row = (
        saved_lookup.loc[h3]
    )


    output = {

        "h3_index":
            h3,

        "quarter_label":
            row[
                "quarter_label"
            ],

        "emerging_risk_score":
            float(
                risk_scores[
                    row_position
                ]
            ),

        "risk_rank":
            int(
                saved_row[
                    "risk_rank"
                ]
            ),

        "risk_band":
            saved_row[
                "risk_band"
            ],

        "actual_next_quarter_event":
            int(
                row[
                    "target_is_below_area_norm_next_q"
                ]
            ),

        "ookla_tests":
            int(
                row[
                    "ookla_tests"
                ]
            ),

        "current_dl_mbps":
            float(
                row[
                    "ookla_dl_mbps"
                ]
            ),

        "current_ul_mbps":
            float(
                row[
                    "ookla_ul_mbps"
                ]
            ),

        "current_rtt_ms":
            float(
                row[
                    "ookla_rtt_ms"
                ]
            ),
    }


    # --------------------------------------------------------
    # Top 3 risk-increasing contributors
    # --------------------------------------------------------

    for rank in range(3):

        if rank < len(positive):

            feature, shap_value = (
                positive[rank]
            )

            raw_value = row[
                feature
            ]

            friendly = (
                FEATURE_LABELS.get(
                    feature,
                    feature,
                )
            )

            formatted_value = (
                format_feature_value(
                    feature,
                    raw_value,
                )
            )

            output[
                f"risk_driver_{rank + 1}_feature"
            ] = feature

            output[
                f"risk_driver_{rank + 1}_label"
            ] = friendly

            output[
                f"risk_driver_{rank + 1}_family"
            ] = feature_family(
                feature
            )

            output[
                f"risk_driver_{rank + 1}_value"
            ] = formatted_value

            output[
                f"risk_driver_{rank + 1}_shap"
            ] = shap_value

            output[
                f"risk_driver_{rank + 1}_text"
            ] = (
                f"{friendly} "
                f"({formatted_value}) "
                f"pushed the model toward "
                f"higher emerging risk."
            )


    # --------------------------------------------------------
    # Top 2 risk-reducing contributors
    # --------------------------------------------------------

    for rank in range(2):

        if rank < len(negative):

            feature, shap_value = (
                negative[rank]
            )

            raw_value = row[
                feature
            ]

            friendly = (
                FEATURE_LABELS.get(
                    feature,
                    feature,
                )
            )

            formatted_value = (
                format_feature_value(
                    feature,
                    raw_value,
                )
            )

            output[
                f"protective_{rank + 1}_feature"
            ] = feature

            output[
                f"protective_{rank + 1}_label"
            ] = friendly

            output[
                f"protective_{rank + 1}_family"
            ] = feature_family(
                feature
            )

            output[
                f"protective_{rank + 1}_value"
            ] = formatted_value

            output[
                f"protective_{rank + 1}_shap"
            ] = shap_value

            output[
                f"protective_{rank + 1}_text"
            ] = (
                f"{friendly} "
                f"({formatted_value}) "
                f"pushed the model toward "
                f"lower emerging risk."
            )


    local_rows.append(
        output
    )


local_df = pd.DataFrame(
    local_rows
)


local_df = local_df.sort_values(
    "risk_rank"
)


# ============================================================
# 17. Console summary
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "GLOBAL MODEL CONTRIBUTORS"
)

print("=" * 80)


print(

    global_df[
        [
            "friendly_name",
            "family",
            "importance_pct",
        ]
    ]

    .head(15)

    .to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "CONTRIBUTION BY INFORMATION FAMILY"
)

print("=" * 80)


print(
    family_df.to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "TOP CONTRIBUTORS AMONG HIGH-RISK CELLS"
)

print("=" * 80)


print(

    high_importance_df[
        [
            "friendly_name",
            "family",
            "mean_abs_shap_high_risk",
        ]
    ]

    .head(15)

    .to_string(
        index=False
    )
)


# ============================================================
# 18. Show example high-risk explanations
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "EXAMPLE HIGH-RISK CELLS"
)

print("=" * 80)


for _, row in (
    local_df
    .head(10)
    .iterrows()
):

    print(
        "\n"
        f"Rank {row['risk_rank']} | "
        f"{row['h3_index']} | "
        f"Band={row['risk_band']} | "
        f"Score={row['emerging_risk_score']:.4f} | "
        f"Actual next-quarter event="
        f"{row['actual_next_quarter_event']}"
    )

    for rank in range(
        1,
        4
    ):

        text_column = (
            f"risk_driver_{rank}_text"
        )

        if (
            text_column
            in row.index
            and pd.notna(
                row[
                    text_column
                ]
            )
        ):

            print(
                "  ↑ "
                + row[
                    text_column
                ]
            )


# ============================================================
# 19. Save interpretation outputs
# ============================================================

global_df.to_csv(

    OUTPUT_DIR
    / "10_shap_global_importance.csv",

    index=False,
)


family_df.to_csv(

    OUTPUT_DIR
    / "10_shap_family_importance.csv",

    index=False,
)


high_importance_df.to_csv(

    OUTPUT_DIR
    / "10_shap_high_risk_importance.csv",

    index=False,
)


local_df.to_parquet(

    OUTPUT_DIR
    / "10_final_test_local_explanations.parquet",

    index=False,
)


local_df[
    local_df[
        "risk_band"
    ] == "high"
].to_csv(

    OUTPUT_DIR
    / "10_high_risk_explanations.csv",

    index=False,
)


# ============================================================
# 20. Save dashboard explanation dictionary
# ============================================================

explanation_dictionary = {

    "model_version":
        "finals_quarterly_onset_v1.0.0",

    "method":
        "XGBoost SHAP contributions",

    "score_label":
        "Emerging-risk score",

    "score_warning":
        (
            "The model score is used for "
            "relative risk ranking and is "
            "not presented as a calibrated "
            "probability."
        ),

    "explanation_warning":
        (
            "Feature contributions explain "
            "which inputs influenced the "
            "model prediction. They do not "
            "establish the cause of future "
            "network degradation."
        ),

    "recommended_dashboard_heading":
        "Why is this area flagged?",

    "recommended_subheading":
        (
            "Factors contributing to the "
            "model's emerging-risk assessment"
        ),

    "feature_labels":
        FEATURE_LABELS,
}


with open(

    OUTPUT_DIR
    / "10_explanation_dictionary.json",

    "w",
    encoding="utf-8",

) as f:

    json.dump(
        explanation_dictionary,
        f,
        indent=2,
    )


# ============================================================
# 21. Finish
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CHECKPOINT 10 COMPLETE"
)

print("=" * 80)


print(
    "\nNo model parameters were changed."
)

print(
    "Final-test interpretation only."
)

print(
    "\nSaved interpretation outputs to:"
)

print(
    OUTPUT_DIR
)