GeoQoE Pulse — Competition AI Package
======================================

Generated latest-hour timestamp:
2026-06-30 23:00:00

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
full_causal_xgboost

Best iteration:
18

Frozen validation threshold:
0.44769213

The model and threshold must not be changed based on the final
test results without creating a new development and evaluation
cycle.