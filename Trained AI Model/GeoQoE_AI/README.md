# GeoQoE Pulse — AI and Dashboard Handoff

GeoQoE Pulse is a hyperlocal mobile-network quality intelligence prototype for Klang Valley. It combines:

1. **A static spatial QoE model** trained against real quarterly Ookla download-speed labels.
2. **A simulated next-hour degradation-onset model** that ranks currently normal anchored cells by relative onset risk.

The two outputs must remain clearly separated. The static model is grounded in real quarterly labels but has limited accuracy. The hourly model demonstrates an operational forecasting workflow, but its hourly QoE and onset labels are simulated.

## 1. What the package can do

### Static spatial layer

- Show real Ookla download speed where a cell has a label.
- Estimate a rough download-speed baseline for unlabelled cells.
- Mark each value as `observed_ookla` or `ai_estimated`.
- Provide relative reliability, feature-space support and ensemble disagreement.

### Latest-hour onset layer

- Mark anchored cells that are already degraded.
- Score anchored cells that are currently normal.
- Rank eligible cells by relative next-hour onset risk.
- Provide Top-20, Top-50, Top-100, Top-1% and Top-5% flags.
- Provide dashboard priority tiers.

The onset score is **not a calibrated probability**. A score of `0.64` must not be displayed as “64% chance.”

## 2. Frozen model information

- Selected model: `full_causal_xgboost`
- Features: 41
- Frozen best iteration: `18`
- Frozen validation threshold: `0.44769213`
- Selection basis: highest validation PR-AUC before final-test evaluation
- Final-test PR-AUC: `0.160982`
- Final-test ROC-AUC: `0.995539`
- Final-test precision: `0.125457`
- Final-test recall: `0.990385`
- Final-test F1: `0.222703`

The model detected 103 of 104 simulated test onsets at the frozen threshold, but generated 718 false positives. The dashboard should emphasise **ranking and Top-K inspection**, not a claim of highly precise binary alerts.

## 3. Latest packaged snapshot

- Timestamp: `2026-06-30 23:00:00`
- Total H3 cells: `4,918`
- Eligible onset-scored cells: `2,843`
- Currently degraded anchored cells: `107`
- Unanchored/not scored cells: `1,968`
- Frozen-threshold alerts: `44`

## 4. Minimum files for dashboard development

Provide the entire `competition_package/` folder. The dashboard can be built from these files alone:

```text
competition_package/
├── README.txt
├── package_manifest.json
├── models/
│   └── geoqoe_onset_full_causal_xgboost.json
├── metadata/
│   ├── geoqoe_onset_feature_registry.json
│   └── geoqoe_model_card.json
└── dashboard/
    ├── latest_hour_onset_risk.parquet
    ├── latest_hour_onset_risk.csv
    └── latest_hour_ranked_eligible_cells.csv
```

### Recommended primary data file

Use:

```text
competition_package/dashboard/latest_hour_onset_risk.parquet
```

Use the CSV version when the development environment does not support Parquet.

### Ranked alert table

Use:

```text
competition_package/dashboard/latest_hour_ranked_eligible_cells.csv
```

This contains only eligible cells, ordered from highest to lowest onset risk.

## 5. Dashboard column reference

### Geographic and time keys

| Column | Meaning |
|---|---|
| `h3_index` | H3 resolution-8 cell identifier. Main map/join key. |
| `h3_parent_res6` | Larger H3 parent region used for spatial validation. |
| `timestamp` | Predictor timestamp represented by the row. |

### Static QoE fields

| Column | Meaning |
|---|---|
| `static_data_status` | `observed_ookla` or `ai_estimated`. |
| `display_dl_mbps` | Observed Ookla speed when available; otherwise AI static estimate. |
| `ookla_dl_mbps` | Real quarterly Ookla target where available. |
| `ai_ensemble_estimate_mbps` | Five-model static estimate. |
| `ensemble_std_mbps` | Disagreement among static fold models; not a confidence interval. |
| `ensemble_disagreement_level` | Relative `low`, `moderate` or `high` disagreement. |
| `feature_space_support_level` | `within_reference`, `limited_reference` or `outside_reference`. |
| `relative_prediction_reliability` | Relative `higher`, `moderate` or `lower` support. |

### Onset operational fields

| Column | Meaning |
|---|---|
| `target_is_degraded` | Current simulated degraded state. |
| `qoe_level_is_ookla_anchored` | Whether hourly QoE is anchored to a real Ookla-labelled cell. |
| `eligible_for_onset_scoring` | True only for anchored, currently normal, non-warm-up cells. |
| `onset_operational_status` | `onset_scored`, `currently_degraded`, `not_scored_unanchored`, etc. |
| `onset_priority_tier` | Dashboard category: current degradation, highest/high priority, elevated/routine watch, or not scored. |
| `next_hour_onset_risk_score` | Relative class-weighted risk score, not probability. |
| `onset_risk_rank` | Rank among eligible cells at the same hour; lower is higher risk. |
| `onset_risk_percentile` | Relative position among eligible latest-hour scores. |
| `alert_at_frozen_threshold` | Whether score exceeds the validation-frozen threshold. |
| `top_20_onset_risk` | Cell is in the Top 20 latest-hour eligible scores. |
| `top_50_onset_risk` | Cell is in the Top 50. |
| `top_100_onset_risk` | Cell is in the Top 100. |
| `top_1_percent_onset_risk` | Cell is in the top relative 1% band. |
| `top_5_percent_onset_risk` | Cell is in the top relative 5% band. |

### Supporting context

| Column | Meaning |
|---|---|
| `dl_median` | Current simulated median download speed. |
| `rtt_median` | Current simulated median latency. |
| `jitter_median` | Current simulated jitter. |
| `loss_median` | Current simulated packet loss. |
| `lag_1h_dl_median` | Download speed one hour earlier. |
| `lag_1h_rtt_median` | Latency one hour earlier. |
| `rolling_mean_24h_dl_median` | Trailing historical download average. |
| `rolling_mean_24h_rtt_median` | Trailing historical latency average. |
| `demand_pressure_base` | Static GIS-based demand pressure indicator. |

## 6. Recommended map logic

Use `onset_operational_status` first, then `onset_priority_tier`:

1. `current_degradation` — already degraded now.
2. `highest_priority` — top relative onset-risk band.
3. `high_priority` — next high-risk band.
4. `elevated_watch` — above frozen threshold but outside top bands.
5. `routine_watch` — eligible and scored, lower relative rank.
6. `not_scored` — outside onset-model scope.

Suggested user-facing labels:

| Internal value | UI label |
|---|---|
| `current_degradation` | Current degradation |
| `highest_priority` | Highest next-hour priority |
| `high_priority` | High next-hour priority |
| `elevated_watch` | Elevated watch |
| `routine_watch` | Routine watch |
| `not_scored` | Not scored |

Scores can tie because XGBoost tree leaves assign identical values to similar rows. Do not imply that tied ranks are meaningfully different. Prefer priority groups and Top-K flags.

## 7. Suggested tooltips and disclaimers

### Risk score tooltip

> Relative next-hour degradation-onset risk within the simulated hourly scenario. This class-weighted score is used for ranking and is not a calibrated probability.

### Static estimate tooltip

> AI-estimated quarterly spatial QoE baseline based on GIS patterns learned from real Ookla-labelled cells. This is not a direct measurement.

### Reliability tooltip

> Relative model support based on similarity to labelled cells and agreement among fold models. It is not percentage accuracy or a statistical confidence interval.

### Global dashboard disclaimer

> Observed, AI-estimated and simulated outputs are displayed separately. Hourly forecast results are based on a simulated scenario and require validation on real hourly telecom telemetry before operational use.

## 8. Recommended repository structure

```text
GeoQoE_AI/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── inspect_new_dataset.py
│   ├── static_inspect_columns.py
│   ├── static_feature_registry.py
│   ├── static_check_spatial_folds.py
│   ├── static_train_baselines.py
│   ├── static_train_xgboost.py
│   ├── static_generate_all_predictions.py
│   ├── static_check_prediction_reliability.py
│   ├── onset_check_readiness.py
│   ├── onset_feature_registry.py
│   ├── onset_train_logistic_baselines.py
│   ├── onset_train_xgboost.py
│   ├── onset_evaluate_final_test.py
│   └── finalize_competition_models.py
├── competition_package/
│   └── ... frozen deliverables ...
├── docs/
│   └── GeoQoE_AI_Model_Training_Explained.docx
├── outputs/
│   ├── static_model/
│   └── onset_model/
├── models/
│   ├── static_xgboost/
│   ├── onset_logistic/
│   └── onset_xgboost/
└── data/
    └── pipeline_v3_2/
```

Large Parquet files may exceed ordinary Git limits. Use Git LFS, a release asset, cloud storage, or provide a documented download path rather than committing oversized files directly.

## 9. Files required for reproducibility

For another AI developer to rerun the workflow, provide:

### Source data

```text
data/pipeline_v3_2/cell_static_features.parquet
data/pipeline_v3_2/cell_static_qoe_baseline.parquet
data/pipeline_v3_2/cell_time_dataset.parquet
data/pipeline_v3_2/feature_dictionary.xlsx
data/pipeline_v3_2/metadata.json
data/pipeline_v3_2/qoe_targets.parquet
```

### Code

Provide all 14 scripts listed in the suggested repository structure and `requirements.txt`.

### Frozen and comparison models

```text
models/static_xgboost/confidence_weighted_fold_1.json
models/static_xgboost/confidence_weighted_fold_2.json
models/static_xgboost/confidence_weighted_fold_3.json
models/static_xgboost/confidence_weighted_fold_4.json
models/static_xgboost/confidence_weighted_fold_5.json
models/onset_xgboost/time_only_xgboost.json
models/onset_xgboost/qoe_history_xgboost.json
models/onset_xgboost/full_causal_xgboost.json
models/onset_logistic/time_only_logistic.joblib
models/onset_logistic/qoe_history_logistic.joblib
```

### Key audit and evaluation outputs

Static:

```text
outputs/static_model/static_column_audit.csv
outputs/static_model/static_spatial_fold_assignments.csv
outputs/static_model/static_baseline_overall_metrics.json
outputs/static_model/static_xgboost_overall_metrics.json
outputs/static_model/static_xgboost_oof_predictions.parquet
outputs/static_model/static_xgboost_feature_importance.csv
outputs/static_model/static_all_cell_predictions_with_reliability.parquet
outputs/static_model/static_prediction_summary.json
outputs/static_model/static_prediction_reliability_report.json
```

Onset:

```text
outputs/onset_model/onset_readiness_report.json
outputs/onset_model/onset_temporal_split_summary.csv
outputs/onset_model/onset_feature_registry.json
outputs/onset_model/onset_logistic_validation_metrics.json
outputs/onset_model/onset_xgboost_validation_metrics.json
outputs/onset_model/onset_xgboost_feature_importance.csv
outputs/onset_model/onset_validation_top_k_per_timestamp.csv
outputs/onset_model/onset_final_test_metrics.json
outputs/onset_model/onset_final_test_metrics.csv
outputs/onset_model/onset_final_test_predictions.parquet
outputs/onset_model/onset_final_test_top_k_per_timestamp.csv
```

## 10. What not to commit

```text
.venv/
.idea/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/development/
temporary extracts or duplicate ZIP files
personal credentials, API keys or tokens
local absolute paths and personal notes
```

Suggested `.gitignore` entries:

```gitignore
.venv/
.idea/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.DS_Store
Thumbs.db
*.log
```

## 11. Important limitations

- Static model spatial OOF MAE: approximately `82.30 Mbps`.
- Static model R²: `0.0206`; use as a rough estimate only.
- Ookla-labelled and unlabelled cells have strong GIS coverage-selection differences.
- Hourly QoE and future onset labels are simulated.
- Onsets are extremely rare and highly concentrated at 07:00 and 16:00.
- Class weighting makes the onset score uncalibrated.
- The full model is dominated by current and lagged QoE features.
- QoE-history-only XGBoost achieved a higher final-test PR-AUC than the selected full model, but the full model remained frozen because it won on validation.
- Production use requires real hourly telecom telemetry, longer time coverage and external validation.

## 12. Integration checklist

1. Load `latest_hour_onset_risk.parquet`.
2. Convert `h3_index` to H3 polygons in the mapping library.
3. Use `display_dl_mbps` for the static speed layer.
4. Use `static_data_status` to visually separate observed and estimated cells.
5. Use `onset_operational_status` for current degradation versus scored/not-scored logic.
6. Use `onset_priority_tier` and Top-K flags for the primary risk layer.
7. Show `next_hour_onset_risk_score` as a relative score, not a percent.
8. Add the global simulation disclaimer.
9. Display timestamp and model version.
10. Keep the original frozen package unchanged; create a new version for future model changes.
