from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from onset_feature_registry import (
    ANCHOR_COLUMN,
    CURRENT_STATE_COLUMN,
    FULL_CAUSAL_FEATURES,
    ID_COLUMN,
    SPATIAL_GROUP_COLUMN,
    TIMESTAMP_COLUMN,
    WARMUP_COLUMN,
    unique_preserving_order,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# SOURCE FILES
# ============================================================

HOURLY_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
    / "cell_time_dataset.parquet"
)

SELECTED_MODEL_SOURCE_PATH = (
    PROJECT_ROOT
    / "models"
    / "onset_xgboost"
    / "full_causal_xgboost.json"
)

ONSET_REGISTRY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
    / "onset_feature_registry.json"
)

VALIDATION_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
    / "onset_xgboost_validation_metrics.json"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
    / "onset_xgboost_selected_thresholds.json"
)

FINAL_TEST_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
    / "onset_final_test_metrics.json"
)

STATIC_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
    / "static_all_cell_predictions_with_reliability.parquet"
)

STATIC_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
    / "static_prediction_summary.json"
)

STATIC_RELIABILITY_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
    / "static_prediction_reliability_report.json"
)


# ============================================================
# COMPETITION PACKAGE
# ============================================================

PACKAGE_DIRECTORY = (
    PROJECT_ROOT
    / "competition_package"
)

PACKAGE_MODEL_DIRECTORY = (
    PACKAGE_DIRECTORY
    / "models"
)

PACKAGE_DASHBOARD_DIRECTORY = (
    PACKAGE_DIRECTORY
    / "dashboard"
)

PACKAGE_METADATA_DIRECTORY = (
    PACKAGE_DIRECTORY
    / "metadata"
)

FROZEN_MODEL_PATH = (
    PACKAGE_MODEL_DIRECTORY
    / "geoqoe_onset_full_causal_xgboost.json"
)

FROZEN_FEATURE_REGISTRY_PATH = (
    PACKAGE_METADATA_DIRECTORY
    / "geoqoe_onset_feature_registry.json"
)

DASHBOARD_PARQUET_PATH = (
    PACKAGE_DASHBOARD_DIRECTORY
    / "latest_hour_onset_risk.parquet"
)

DASHBOARD_CSV_PATH = (
    PACKAGE_DASHBOARD_DIRECTORY
    / "latest_hour_onset_risk.csv"
)

ELIGIBLE_RANKING_CSV_PATH = (
    PACKAGE_DASHBOARD_DIRECTORY
    / "latest_hour_ranked_eligible_cells.csv"
)

MODEL_CARD_PATH = (
    PACKAGE_METADATA_DIRECTORY
    / "geoqoe_model_card.json"
)

README_PATH = (
    PACKAGE_DIRECTORY
    / "README.txt"
)

MANIFEST_PATH = (
    PACKAGE_DIRECTORY
    / "package_manifest.json"
)


# ============================================================
# FROZEN MODEL SELECTION
# ============================================================

SELECTED_MODEL_NAME = "full_causal_xgboost"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 hash of one file."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """
    Convert NumPy and pandas values into JSON-compatible types.
    """

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
        ),
    ):
        return str(value)

    return value


def assign_operational_status(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Explain why each latest-hour cell was or was not scored.
    """

    anchored = df[
        ANCHOR_COLUMN
    ].astype(bool)

    currently_degraded = (
        pd.to_numeric(
            df[CURRENT_STATE_COLUMN],
            errors="coerce",
        )
        == 1
    )

    warmup = df[
        WARMUP_COLUMN
    ].astype(bool)

    eligible = df[
        "eligible_for_onset_scoring"
    ].astype(bool)

    return pd.Series(
        np.select(
            [
                ~anchored,
                currently_degraded,
                warmup,
                eligible,
            ],
            [
                "not_scored_unanchored",
                "currently_degraded",
                "not_scored_warmup",
                "onset_scored",
            ],
            default="not_scored_other",
        ),
        index=df.index,
        dtype="string",
    )


def assign_priority_tier(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Assign dashboard-oriented priority labels.

    These tiers are based on relative ranking and the frozen
    validation threshold. They are not probability bands.
    """

    status = df[
        "onset_operational_status"
    ].astype(str)

    top_1_percent = df[
        "top_1_percent_onset_risk"
    ].fillna(False)

    top_5_percent = df[
        "top_5_percent_onset_risk"
    ].fillna(False)

    threshold_alert = df[
        "alert_at_frozen_threshold"
    ].fillna(False)

    return pd.Series(
        np.select(
            [
                status == "currently_degraded",
                status == "not_scored_unanchored",
                top_1_percent,
                top_5_percent,
                threshold_alert,
                status == "onset_scored",
            ],
            [
                "current_degradation",
                "not_scored",
                "highest_priority",
                "high_priority",
                "elevated_watch",
                "routine_watch",
            ],
            default="not_scored",
        ),
        index=df.index,
        dtype="string",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Competition Model Finalisation")
    print("=" * 70)

    required_paths = [
        HOURLY_DATA_PATH,
        SELECTED_MODEL_SOURCE_PATH,
        ONSET_REGISTRY_PATH,
        VALIDATION_METRICS_PATH,
        THRESHOLDS_PATH,
        FINAL_TEST_METRICS_PATH,
        STATIC_PREDICTIONS_PATH,
        STATIC_SUMMARY_PATH,
        STATIC_RELIABILITY_REPORT_PATH,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{required_path}"
            )

    PACKAGE_MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PACKAGE_DASHBOARD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PACKAGE_METADATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_report = load_json(
        VALIDATION_METRICS_PATH
    )

    threshold_report = load_json(
        THRESHOLDS_PATH
    )

    final_test_report = load_json(
        FINAL_TEST_METRICS_PATH
    )

    static_summary = load_json(
        STATIC_SUMMARY_PATH
    )

    static_reliability_report = load_json(
        STATIC_RELIABILITY_REPORT_PATH
    )

    # ========================================================
    # 1. Verify that the frozen model matches validation choice.
    # ========================================================

    validation_winner = validation_report[
        "provisional_highest_pr_auc_model"
    ]

    final_test_selected_model = (
        final_test_report[
            "selection_protocol"
        ][
            "selected_model"
        ]
    )

    print("\n--- Frozen selection verification ---")

    print(
        "Expected selected model:",
        SELECTED_MODEL_NAME,
    )

    print(
        "Validation winner:",
        validation_winner,
    )

    print(
        "Final-test frozen model:",
        final_test_selected_model,
    )

    if validation_winner != SELECTED_MODEL_NAME:
        raise ValueError(
            "The selected model does not match the "
            "saved validation winner."
        )

    if (
        final_test_selected_model
        != SELECTED_MODEL_NAME
    ):
        raise ValueError(
            "The final-test report used a different model."
        )

    frozen_threshold = float(
        threshold_report[
            SELECTED_MODEL_NAME
        ][
            "threshold"
        ]
    )

    frozen_best_iteration = int(
        validation_report[
            "xgboost_results"
        ][
            SELECTED_MODEL_NAME
        ][
            "best_iteration"
        ]
    )

    print(
        "Frozen validation threshold:",
        f"{frozen_threshold:.8f}",
    )

    print(
        "Frozen best iteration:",
        frozen_best_iteration,
    )

    # ========================================================
    # 2. Copy the exact evaluated model and feature registry.
    # ========================================================

    shutil.copy2(
        SELECTED_MODEL_SOURCE_PATH,
        FROZEN_MODEL_PATH,
    )

    shutil.copy2(
        ONSET_REGISTRY_PATH,
        FROZEN_FEATURE_REGISTRY_PATH,
    )

    print("\n--- Frozen model files ---")

    print(
        "Model copied to:",
        FROZEN_MODEL_PATH,
    )

    print(
        "Feature registry copied to:",
        FROZEN_FEATURE_REGISTRY_PATH,
    )

    # ========================================================
    # 3. Load the latest available hourly state.
    # ========================================================

    required_hourly_columns = (
        unique_preserving_order(
            [
                ID_COLUMN,
                SPATIAL_GROUP_COLUMN,
                TIMESTAMP_COLUMN,
                WARMUP_COLUMN,
                ANCHOR_COLUMN,
                CURRENT_STATE_COLUMN,
                *FULL_CAUSAL_FEATURES,
            ]
        )
    )

    print(
        f"\nLoading latest hourly state from:\n"
        f"{HOURLY_DATA_PATH}"
    )

    hourly_df = pd.read_parquet(
        HOURLY_DATA_PATH,
        columns=required_hourly_columns,
    )

    hourly_df[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        hourly_df[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    latest_timestamp = hourly_df[
        TIMESTAMP_COLUMN
    ].max()

    latest_df = hourly_df[
        hourly_df[
            TIMESTAMP_COLUMN
        ] == latest_timestamp
    ].copy()

    latest_df = latest_df.reset_index(
        drop=True
    )

    print("\n--- Latest-hour dataset ---")

    print(
        "Timestamp:",
        latest_timestamp,
    )

    print(
        "Grid cells:",
        f"{len(latest_df):,}",
    )

    if latest_df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "The latest-hour dataset contains duplicate H3 cells."
        )

    # ========================================================
    # 4. Determine which cells are eligible for onset scoring.
    # ========================================================

    current_state = pd.to_numeric(
        latest_df[
            CURRENT_STATE_COLUMN
        ],
        errors="coerce",
    )

    anchored = latest_df[
        ANCHOR_COLUMN
    ].astype(bool)

    warmup = latest_df[
        WARMUP_COLUMN
    ].astype(bool)

    eligible_mask = (
        anchored
        & ~warmup
        & (current_state == 0)
    )

    latest_df[
        "eligible_for_onset_scoring"
    ] = eligible_mask

    latest_df[
        "next_hour_onset_risk_score"
    ] = np.nan

    eligible_count = int(
        eligible_mask.sum()
    )

    print(
        "Anchored currently normal eligible cells:",
        f"{eligible_count:,}",
    )

    print(
        "Currently degraded anchored cells:",
        f"{int((anchored & (current_state == 1)).sum()):,}",
    )

    print(
        "Unanchored cells:",
        f"{int((~anchored).sum()):,}",
    )

    if eligible_count == 0:
        raise ValueError(
            "No cells are eligible for onset scoring."
        )

    X_latest = latest_df.loc[
        eligible_mask,
        FULL_CAUSAL_FEATURES,
    ].astype("float32")

    if not np.isfinite(
        X_latest.to_numpy(copy=False)
    ).all():
        raise ValueError(
            "Eligible latest-hour features contain "
            "NaN or infinite values."
        )

    # ========================================================
    # 5. Score latest-hour eligible cells.
    # ========================================================

    model = xgb.XGBClassifier()

    model.load_model(
        str(FROZEN_MODEL_PATH)
    )

    risk_scores = model.predict_proba(
        X_latest,
        iteration_range=(
            0,
            frozen_best_iteration + 1,
        ),
    )[:, 1]

    latest_df.loc[
        eligible_mask,
        "next_hour_onset_risk_score",
    ] = risk_scores

    # Outputs are risk scores, not calibrated probabilities.
    latest_df[
        "alert_at_frozen_threshold"
    ] = False

    latest_df.loc[
        eligible_mask,
        "alert_at_frozen_threshold",
    ] = (
        risk_scores
        >= frozen_threshold
    )

    # ========================================================
    # 6. Rank eligible cells within the latest hour.
    # ========================================================

    eligible_scores = latest_df.loc[
        eligible_mask,
        "next_hour_onset_risk_score",
    ]

    latest_df[
        "onset_risk_rank"
    ] = pd.Series(
        pd.NA,
        index=latest_df.index,
        dtype="Int64",
    )

    latest_df[
        "onset_risk_percentile"
    ] = np.nan

    risk_rank = eligible_scores.rank(
        method="first",
        ascending=False,
    ).astype("int64")

    risk_percentile = eligible_scores.rank(
        method="average",
        pct=True,
        ascending=True,
    )

    latest_df.loc[
        eligible_mask,
        "onset_risk_rank",
    ] = risk_rank

    latest_df.loc[
        eligible_mask,
        "onset_risk_percentile",
    ] = risk_percentile

    latest_df[
        "top_20_onset_risk"
    ] = (
        latest_df[
            "onset_risk_rank"
        ].fillna(10**9)
        <= 20
    )

    latest_df[
        "top_50_onset_risk"
    ] = (
        latest_df[
            "onset_risk_rank"
        ].fillna(10**9)
        <= 50
    )

    latest_df[
        "top_100_onset_risk"
    ] = (
        latest_df[
            "onset_risk_rank"
        ].fillna(10**9)
        <= 100
    )

    latest_df[
        "top_1_percent_onset_risk"
    ] = (
        latest_df[
            "onset_risk_percentile"
        ].fillna(0.0)
        >= 0.99
    )

    latest_df[
        "top_5_percent_onset_risk"
    ] = (
        latest_df[
            "onset_risk_percentile"
        ].fillna(0.0)
        >= 0.95
    )

    latest_df[
        "onset_operational_status"
    ] = assign_operational_status(
        latest_df
    )

    latest_df[
        "onset_priority_tier"
    ] = assign_priority_tier(
        latest_df
    )

    latest_df[
        "onset_model_name"
    ] = SELECTED_MODEL_NAME

    latest_df[
        "onset_model_best_iteration"
    ] = frozen_best_iteration

    latest_df[
        "onset_frozen_threshold"
    ] = frozen_threshold

    latest_df[
        "onset_score_interpretation"
    ] = (
        "relative simulated next-hour onset risk score; "
        "not a calibrated probability"
    )

    latest_df[
        "hourly_data_status"
    ] = (
        "Ookla-anchored simulated hourly scenario"
    )

    # ========================================================
    # 7. Join static QoE and reliability information.
    # ========================================================

    static_df = pd.read_parquet(
        STATIC_PREDICTIONS_PATH
    )

    static_join_columns = [
        ID_COLUMN,
        "static_data_status",
        "display_dl_mbps",
        "ookla_dl_mbps",
        "ai_ensemble_estimate_mbps",
        "ensemble_std_mbps",
        "ensemble_disagreement_level",
        "feature_space_support_level",
        "relative_prediction_reliability",
    ]

    missing_static_columns = [
        column
        for column in static_join_columns
        if column not in static_df.columns
    ]

    if missing_static_columns:
        raise ValueError(
            "Static reliability output is missing columns: "
            f"{missing_static_columns}"
        )

    latest_df = latest_df.merge(
        static_df[
            static_join_columns
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if latest_df[
        "static_data_status"
    ].isna().any():
        raise ValueError(
            "Some latest-hour cells did not match the "
            "static reliability output."
        )

    # ========================================================
    # 8. Create dashboard-ready column order.
    # ========================================================

    dashboard_columns = [
        ID_COLUMN,
        SPATIAL_GROUP_COLUMN,
        TIMESTAMP_COLUMN,

        CURRENT_STATE_COLUMN,
        ANCHOR_COLUMN,

        "static_data_status",
        "display_dl_mbps",
        "ookla_dl_mbps",
        "ai_ensemble_estimate_mbps",
        "ensemble_std_mbps",
        "ensemble_disagreement_level",
        "feature_space_support_level",
        "relative_prediction_reliability",

        "eligible_for_onset_scoring",
        "onset_operational_status",
        "onset_priority_tier",

        "next_hour_onset_risk_score",
        "onset_risk_rank",
        "onset_risk_percentile",

        "alert_at_frozen_threshold",
        "top_20_onset_risk",
        "top_50_onset_risk",
        "top_100_onset_risk",
        "top_1_percent_onset_risk",
        "top_5_percent_onset_risk",

        "dl_median",
        "rtt_median",
        "jitter_median",
        "loss_median",
        "lag_1h_dl_median",
        "lag_1h_rtt_median",
        "rolling_mean_24h_dl_median",
        "rolling_mean_24h_rtt_median",
        "demand_pressure_base",

        "onset_model_name",
        "onset_model_best_iteration",
        "onset_frozen_threshold",
        "onset_score_interpretation",
        "hourly_data_status",
    ]

    missing_dashboard_columns = [
        column
        for column in dashboard_columns
        if column not in latest_df.columns
    ]

    if missing_dashboard_columns:
        raise ValueError(
            "Dashboard output is missing columns: "
            f"{missing_dashboard_columns}"
        )

    dashboard_df = latest_df[
        dashboard_columns
    ].copy()

    eligible_ranking_df = (
        dashboard_df[
            dashboard_df[
                "eligible_for_onset_scoring"
            ]
        ]
        .sort_values(
            "onset_risk_rank"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # 9. Print dashboard summary.
    # ========================================================

    print("\n--- Latest-hour scoring summary ---")

    print(
        "Eligible scored cells:",
        f"{len(eligible_ranking_df):,}",
    )

    print(
        "Frozen-threshold alerts:",
        f"{int(eligible_ranking_df['alert_at_frozen_threshold'].sum()):,}",
    )

    print("\nPriority-tier counts:")

    print(
        dashboard_df[
            "onset_priority_tier"
        ].value_counts()
    )

    print("\nTop 20 latest-hour onset risks:")

    print(
        eligible_ranking_df[
            [
                ID_COLUMN,
                "next_hour_onset_risk_score",
                "onset_risk_rank",
                "onset_priority_tier",
                "dl_median",
                "rtt_median",
                "demand_pressure_base",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 10. Save dashboard outputs.
    # ========================================================

    dashboard_df.to_parquet(
        DASHBOARD_PARQUET_PATH,
        index=False,
        compression="zstd",
    )

    dashboard_df.to_csv(
        DASHBOARD_CSV_PATH,
        index=False,
    )

    eligible_ranking_df.to_csv(
        ELIGIBLE_RANKING_CSV_PATH,
        index=False,
    )

    # ========================================================
    # 11. Create consolidated model card.
    # ========================================================

    selected_test_result = (
        final_test_report[
            "selected_model_result"
        ]
    )

    selected_top_k = (
        final_test_report[
            "selected_model_top_k"
        ]
    )

    model_card = {
        "project": "GeoQoE Pulse",
        "package_status": "competition prototype",
        "generated_from_frozen_models": True,

        "static_real_data_model": {
            "purpose": (
                "Estimate a static spatial QoE baseline "
                "for cells without direct Ookla labels"
            ),
            "dataset_status": (
                "Real quarterly Ookla targets with "
                "independent GIS predictors"
            ),
            "selected_model": (
                static_summary[
                    "selected_model"
                ]
            ),
            "observed_cells": (
                static_summary[
                    "observed_ookla_cells"
                ]
            ),
            "ai_estimated_cells": (
                static_summary[
                    "ai_estimated_cells"
                ]
            ),
            "honest_spatial_oof_metrics": (
                static_summary[
                    "honest_oof_metrics"
                ]
            ),
            "coverage_bias_diagnostic": (
                static_reliability_report[
                    "coverage_bias_diagnostic"
                ]
            ),
            "usage": (
                "Use as a rough spatial baseline and "
                "blind-spot estimate, not as a direct "
                "measurement or precise guaranteed Mbps value."
            ),
        },

        "hourly_onset_model": {
            "purpose": (
                "Rank currently normal anchored cells by "
                "next-hour degradation-onset risk"
            ),
            "dataset_status": (
                "Ookla-anchored simulated hourly scenario"
            ),
            "selected_model": (
                SELECTED_MODEL_NAME
            ),
            "feature_count": len(
                FULL_CAUSAL_FEATURES
            ),
            "features": (
                FULL_CAUSAL_FEATURES
            ),
            "frozen_best_iteration": (
                frozen_best_iteration
            ),
            "frozen_validation_threshold": (
                frozen_threshold
            ),
            "selection_basis": (
                "Highest validation PR-AUC before "
                "final-test evaluation"
            ),
            "validation_result": (
                validation_report[
                    "xgboost_results"
                ][
                    SELECTED_MODEL_NAME
                ]
            ),
            "final_test_result": (
                selected_test_result
            ),
            "final_test_top_k": (
                selected_top_k
            ),
            "score_interpretation": (
                "Relative onset-risk score, not a "
                "calibrated probability"
            ),
            "important_observation": (
                "Current and lagged QoE features dominate "
                "the model. GIS and demand context produced "
                "a modest validation improvement, but the "
                "QoE-history-only model achieved a higher "
                "PR-AUC on the final test period."
            ),
        },

        "dashboard_snapshot": {
            "timestamp": str(
                latest_timestamp
            ),
            "total_cells": len(
                dashboard_df
            ),
            "eligible_scored_cells": len(
                eligible_ranking_df
            ),
            "frozen_threshold_alerts": int(
                eligible_ranking_df[
                    "alert_at_frozen_threshold"
                ].sum()
            ),
        },

        "limitations": [
            (
                "Hourly QoE and onset labels are simulated. "
                "Forecast metrics do not validate performance "
                "on observed hourly telecom telemetry."
            ),
            (
                "The static model has low spatial out-of-fold "
                "R-squared and should be treated as a rough "
                "estimator."
            ),
            (
                "Ookla-labelled and unlabelled cells have "
                "strongly different GIS coverage patterns."
            ),
            (
                "Model disagreement and reliability categories "
                "are relative support indicators, not calibrated "
                "confidence probabilities."
            ),
            (
                "Test results were evaluated once and were not "
                "used to change the selected model or threshold."
            ),
        ],

        "recommended_pitch_wording": (
            "GeoQoE Pulse combines a real-data spatial QoE "
            "baseline with a transparently simulated next-hour "
            "onset-ranking workflow. The system highlights "
            "observed, AI-estimated and simulated outputs "
            "separately and exposes model-support indicators."
        ),
    }

    with open(
        MODEL_CARD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(model_card),
            file,
            indent=4,
        )

    # ========================================================
    # 12. Create package README.
    # ========================================================

    readme_text = f"""
GeoQoE Pulse — Competition AI Package
======================================

Generated latest-hour timestamp:
{latest_timestamp}

FILES
-----

models/geoqoe_onset_full_causal_xgboost.json
    Exact XGBoost model selected using validation PR-AUC and
    evaluated once on the final test period.

metadata/geoqoe_onset_feature_registry.json
    Frozen feature groups, exclusions and temporal-split rules.

metadata/geoqoe_model_card.json
    Consolidated model purpose, metrics, limitations and usage.

dashboard/latest_hour_onset_risk.parquet
dashboard/latest_hour_onset_risk.csv
    One row per H3 cell for the latest available hour.

dashboard/latest_hour_ranked_eligible_cells.csv
    Only anchored, currently normal cells that were eligible
    for onset scoring, sorted from highest to lowest risk.

IMPORTANT INTERPRETATION
------------------------

1. next_hour_onset_risk_score is a relative risk score.
   It is NOT a calibrated probability.

2. Hourly onset results are based on simulated hourly QoE,
   not observed hourly telecom telemetry.

3. static_data_status distinguishes observed Ookla cells from
   AI-estimated static cells.

4. onset_operational_status distinguishes:
   - onset_scored
   - currently_degraded
   - not_scored_unanchored

5. Use onset_risk_rank or the Top-K flags for dashboard
   prioritisation.

FROZEN MODEL
------------

Selected model:
{SELECTED_MODEL_NAME}

Best iteration:
{frozen_best_iteration}

Frozen validation threshold:
{frozen_threshold:.8f}

The model and threshold must not be changed based on the final
test results without creating a new development and evaluation
cycle.
""".strip()

    with open(
        README_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            readme_text
        )

    # ========================================================
    # 13. Create package manifest and hashes.
    # ========================================================

    package_files = sorted(
        [
            path
            for path in PACKAGE_DIRECTORY.rglob("*")
            if path.is_file()
            and path != MANIFEST_PATH
        ]
    )

    manifest_rows = []

    for path in package_files:
        relative_path = path.relative_to(
            PACKAGE_DIRECTORY
        )

        manifest_rows.append(
            {
                "path": str(
                    relative_path
                ).replace("\\", "/"),
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    manifest = {
        "project": "GeoQoE Pulse",
        "package_type": (
            "frozen competition AI package"
        ),
        "selected_onset_model": (
            SELECTED_MODEL_NAME
        ),
        "latest_dashboard_timestamp": str(
            latest_timestamp
        ),
        "files": manifest_rows,
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    # ========================================================
    # 14. Final reconciliation.
    # ========================================================

    expected_total_cells = 4918

    reconciliation_passed = (
        len(dashboard_df)
        == expected_total_cells
        and len(
            eligible_ranking_df
        )
        == eligible_count
        and FROZEN_MODEL_PATH.exists()
        and MODEL_CARD_PATH.exists()
        and MANIFEST_PATH.exists()
    )

    print("\n--- Package files ---")

    for path in sorted(
        PACKAGE_DIRECTORY.rglob("*")
    ):
        if path.is_file():
            print(
                "  -",
                path.relative_to(
                    PACKAGE_DIRECTORY
                ),
            )

    print("\n--- Final result ---")

    if reconciliation_passed:
        print(
            "Competition model finalisation PASSED."
        )

        print(
            "The static model, frozen onset model and "
            "dashboard-ready latest-hour outputs are complete."
        )

        print(
            f"\nCompetition package:\n"
            f"{PACKAGE_DIRECTORY}"
        )
    else:
        print(
            "Competition model finalisation FAILED."
        )

        print(
            "Dashboard rows:",
            len(dashboard_df),
        )

        print(
            "Expected dashboard rows:",
            expected_total_cells,
        )

        print(
            "Eligible ranking rows:",
            len(eligible_ranking_df),
        )

        print(
            "Expected eligible rows:",
            eligible_count,
        )


if __name__ == "__main__":
    main()