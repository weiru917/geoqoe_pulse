# Team Handoff File Checklist

This checklist separates the **minimum dashboard handoff**, the **recommended repository**, and the **full reproducibility archive**.

## A. Minimum package for dashboard teammates

Share the complete folder below without removing files:

- [ ] `competition_package/README.txt`
- [ ] `competition_package/package_manifest.json`
- [ ] `competition_package/models/geoqoe_onset_full_causal_xgboost.json`
- [ ] `competition_package/metadata/geoqoe_onset_feature_registry.json`
- [ ] `competition_package/metadata/geoqoe_model_card.json`
- [ ] `competition_package/dashboard/latest_hour_onset_risk.parquet`
- [ ] `competition_package/dashboard/latest_hour_onset_risk.csv`
- [ ] `competition_package/dashboard/latest_hour_ranked_eligible_cells.csv`

The dashboard developers mainly need the three files under `competition_package/dashboard/`. The remaining files explain how the data was generated and preserve the exact model version.

## B. Documentation to include in the repository

- [ ] `README.md`
- [ ] `docs/GeoQoE_AI_Model_Training_Explained.docx`
- [ ] `docs/TEAM_HANDOFF_FILE_CHECKLIST.md`

## C. AI source code for reproducibility

- [ ] `code/inspect_new_dataset.py`
- [ ] `code/static_inspect_columns.py`
- [ ] `code/static_feature_registry.py`
- [ ] `code/static_check_spatial_folds.py`
- [ ] `code/static_train_baselines.py`
- [ ] `code/static_train_xgboost.py`
- [ ] `code/static_generate_all_predictions.py`
- [ ] `code/static_check_prediction_reliability.py`
- [ ] `code/onset_check_readiness.py`
- [ ] `code/onset_feature_registry.py`
- [ ] `code/onset_train_logistic_baselines.py`
- [ ] `code/onset_train_xgboost.py`
- [ ] `code/onset_evaluate_final_test.py`
- [ ] `code/finalize_competition_models.py`
- [ ] `requirements.txt`

## D. Source data needed to rerun everything

- [ ] `data/pipeline_v3_2/cell_static_features.parquet`
- [ ] `data/pipeline_v3_2/cell_static_qoe_baseline.parquet`
- [ ] `data/pipeline_v3_2/cell_time_dataset.parquet`
- [ ] `data/pipeline_v3_2/feature_dictionary.xlsx`
- [ ] `data/pipeline_v3_2/metadata.json`
- [ ] `data/pipeline_v3_2/qoe_targets.parquet`

Use Git LFS or external storage for large Parquet files.

## E. Trained model files

### Static model ensemble

- [ ] `models/static_xgboost/confidence_weighted_fold_1.json`
- [ ] `models/static_xgboost/confidence_weighted_fold_2.json`
- [ ] `models/static_xgboost/confidence_weighted_fold_3.json`
- [ ] `models/static_xgboost/confidence_weighted_fold_4.json`
- [ ] `models/static_xgboost/confidence_weighted_fold_5.json`

### Onset models and baselines

- [ ] `models/onset_xgboost/time_only_xgboost.json`
- [ ] `models/onset_xgboost/qoe_history_xgboost.json`
- [ ] `models/onset_xgboost/full_causal_xgboost.json`
- [ ] `models/onset_logistic/time_only_logistic.joblib`
- [ ] `models/onset_logistic/qoe_history_logistic.joblib`

## F. Key evidence and audit outputs

### Static model

- [ ] `outputs/static_model/static_column_audit.csv`
- [ ] `outputs/static_model/static_spatial_fold_assignments.csv`
- [ ] `outputs/static_model/static_baseline_fold_metrics.csv`
- [ ] `outputs/static_model/static_baseline_overall_metrics.json`
- [ ] `outputs/static_model/static_baseline_oof_predictions.parquet`
- [ ] `outputs/static_model/static_ridge_coefficients.csv`
- [ ] `outputs/static_model/static_xgboost_fold_metrics.csv`
- [ ] `outputs/static_model/static_xgboost_overall_metrics.json`
- [ ] `outputs/static_model/static_xgboost_oof_predictions.parquet`
- [ ] `outputs/static_model/static_xgboost_feature_importance.csv`
- [ ] `outputs/static_model/static_all_cell_predictions_with_reliability.parquet`
- [ ] `outputs/static_model/static_all_cell_predictions_with_reliability.csv`
- [ ] `outputs/static_model/static_prediction_summary.json`
- [ ] `outputs/static_model/static_prediction_reliability_report.json`
- [ ] `outputs/static_model/static_coverage_bias_fold_metrics.csv`
- [ ] `outputs/static_model/static_coverage_bias_feature_coefficients.csv`

### Hourly onset model

- [ ] `outputs/onset_model/onset_dictionary_feature_candidates.csv`
- [ ] `outputs/onset_model/onset_temporal_split_summary.csv`
- [ ] `outputs/onset_model/onset_readiness_report.json`
- [ ] `outputs/onset_model/onset_feature_audit.csv`
- [ ] `outputs/onset_model/onset_feature_registry.json`
- [ ] `outputs/onset_model/onset_logistic_validation_metrics.json`
- [ ] `outputs/onset_model/onset_logistic_validation_metrics.csv`
- [ ] `outputs/onset_model/onset_logistic_validation_predictions.parquet`
- [ ] `outputs/onset_model/onset_logistic_top_k_metrics.csv`
- [ ] `outputs/onset_model/onset_logistic_coefficients.csv`
- [ ] `outputs/onset_model/onset_xgboost_validation_metrics.json`
- [ ] `outputs/onset_model/onset_xgboost_validation_metrics.csv`
- [ ] `outputs/onset_model/onset_xgboost_validation_predictions.parquet`
- [ ] `outputs/onset_model/onset_xgboost_feature_importance.csv`
- [ ] `outputs/onset_model/onset_validation_top_k_per_timestamp.csv`
- [ ] `outputs/onset_model/onset_xgboost_selected_thresholds.json`
- [ ] `outputs/onset_model/onset_final_test_metrics.json`
- [ ] `outputs/onset_model/onset_final_test_metrics.csv`
- [ ] `outputs/onset_model/onset_final_test_predictions.parquet`
- [ ] `outputs/onset_model/onset_final_test_predictions.csv`
- [ ] `outputs/onset_model/onset_final_test_top_k_per_timestamp.csv`

## G. Files normally excluded from Git

- [ ] Do not commit `.venv/`
- [ ] Do not commit `.idea/`
- [ ] Do not commit `__pycache__/`
- [ ] Do not commit temporary ZIP extractions or duplicate outputs
- [ ] Do not commit personal credentials or API keys
- [ ] Do not commit local absolute-path configuration
- [ ] Do not commit `data/development/` unless it is deliberately required

## H. Final dashboard validation checks

- [ ] All 4,918 H3 cells load.
- [ ] `h3_index` converts to valid map polygons.
- [ ] Observed and AI-estimated static values use different legends.
- [ ] Unanchored cells show “Not scored” for onset forecasting.
- [ ] Current degradation is visually separate from predicted onset risk.
- [ ] The risk score is not formatted as a percentage probability.
- [ ] Top-K filters work by latest timestamp.
- [ ] Tied scores are not presented as meaningfully different fine-grained ranks.
- [ ] Timestamp and model version are visible.
- [ ] Simulation and reliability disclaimers are visible.
