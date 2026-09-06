# GeoQoE Pulse — Finals AI Handover

This folder contains the final quarterly AI forecasting model, validated frontend assets, model documentation and audit outputs for the GeoQoE Pulse finals prototype.

---

## 1. What the AI does

The AI is an early-warning model for emerging mobile-network underperformance.

For every quarter, each measured H3 cell is ranked according to its Ookla download performance relative to other measured Klang Valley cells in the same quarter.

A cell is defined as **underperforming** when its download-speed percentile is below the 25th percentile, meaning it is in the bottom 25% of measured cells for that quarter.

The finals forecasting task is:

> Among cells that are currently NOT underperforming, which cells are most likely to fall into the bottom 25% of Klang Valley download performance in the next quarter?

This means:

- Current underperformance is an observed condition from real Ookla measurements.
- Emerging underperformance is predicted by the AI.
- Cells already underperforming are not treated as an AI forecast problem.

The model does NOT claim that demand caused degradation, identify a verified network root cause, or prescribe a specific engineering intervention.

---

## 2. Final model

Model family:

`XGBoost Classifier`

Final model version:

`finals_quarterly_onset_v1.0.0`

Final model file:

`../../models/finals_quarterly/final_xgb_full_v1.0.0.json`

Number of model features:

`44`

The model combines:

1. QoE / network-performance history
2. Geographic and infrastructure context
3. Activity and population context

The final selected model uses the full feature set.

---

## 3. Final frontend package

The recommended frontend file is:

`dashboard_assets/2026Q1_to_2026Q2_cells.geojson`

This file contains:

- H3 polygon geometry
- risk rank
- risk band
- current network measurements
- measurement-support information
- model explanation fields
- actual 2026Q2 outcome
- Time Machine correctness category

The same data is also available as:

`dashboard_assets/2026Q1_to_2026Q2_cells.parquet`

and

`dashboard_assets/2026Q1_to_2026Q2_cells.csv`

The GeoJSON contains 2,186 H3 cells that were:

- measured in 2026Q1,
- currently outside the bottom 25% in 2026Q1,
- and had a valid real next-quarter observation in 2026Q2.

Important:

The GeoJSON contains the forecast-eligible cells only.

Cells that were already underperforming in 2026Q1 should be displayed separately as observed current issues using the quarterly dataset.

---

## 4. Final unseen-test results

Final training period:

`2023Q1 to 2025Q4`

Final predictor quarter:

`2026Q1`

Final unseen outcome quarter:

`2026Q2`

Final test cells:

`2,186`

Actual emerging underperformance events:

`205`

Actual event rate:

`9.38%`

Final model performance:

- PR-AUC: `0.449`
- ROC-AUC: `0.857`

Operational ranking performance:

- Top 50: 32 / 50 later became underperforming = `64%`
- Top 100: 57 / 100 later became underperforming = `57%`
- Top 200: 92 / 200 later became underperforming = `46%`

The Top-200 captured:

`92 / 205 = 44.9%`

of all emerging underperformance events in the final test.

The no-skill PR-AUC reference was approximately the event rate:

`0.094`

PR-AUC = 0.449 does NOT mean 44.9% accuracy.

See `MODEL_CARD.md` for explanation.

---

## 5. Dashboard risk bands

The finals prototype uses ranking-based risk bands:

- High emerging risk: Top 50 cells
- Elevated emerging risk: ranks 51–200
- Lower emerging risk: ranks 201 and below

Final-test observed outcomes:

- High: 32 / 50 = `64%`
- Elevated: 60 / 150 = `40%`
- Lower: 113 / 1,986 = `5.7%`

These are ranking bands, not calibrated probabilities.

The raw XGBoost model score must NOT be presented to users as a percentage probability.

Prefer:

> High emerging risk — Rank 8 of 2,186

Avoid:

> 32.8% probability of degradation

---

## 6. Model explanations

The dashboard package contains SHAP-based explanation fields.

SHAP identifies which model inputs pushed an individual prediction toward higher or lower emerging risk.

Recommended dashboard heading:

> Why is this area flagged?

Recommended subheading:

> Factors contributing to the model's emerging-risk assessment

SHAP explanations do NOT establish causality.

Do not label SHAP results as:

- root cause,
- cause of degradation,
- proof of congestion,
- proof of demand pressure.

Some cells have fewer than three positive SHAP contributors. The frontend should display only the contributors that are available and hide empty rows.

---

## 7. Measurement support

Ookla data is crowdsourced and sample counts vary between cells.

The dashboard therefore includes a separate measurement-support indicator:

- Limited: fewer than 20 Ookla tests
- Typical: 20–149 Ookla tests
- Strong: 150 or more Ookla tests

Measurement support is NOT prediction confidence.

It only describes how much measurement evidence contributed to the current cell-quarter observation.

---

## 8. Time Machine demo

Recommended demonstration:

2026Q1 observed data
→ AI predicts emerging risk for 2026Q2
→ user clicks "Reveal Actual"
→ dashboard compares prediction with real 2026Q2 measurement

Possible Time Machine result values:

- `correctly_anticipated`
- `false_alert`
- `missed`
- `correctly_lower_risk`

Final test counts:

- Correctly anticipated: 92
- False alerts: 108
- Missed: 113
- Correctly lower-risk: 1,873

See `FRONTEND_INTEGRATION.md` for exact frontend fields and display rules.

---

## 9. Important files

Final model:

`../../models/finals_quarterly/final_xgb_full_v1.0.0.json`

Frontend integration guide:

`FRONTEND_INTEGRATION.md`

Model methodology and limitations:

`MODEL_CARD.md`

Frontend GeoJSON:

`dashboard_assets/2026Q1_to_2026Q2_cells.geojson`

Model freeze record:

`audit/08_model_freeze.json`

Final test metrics:

`audit/09_final_test_metrics.csv`

Final Top-K results:

`audit/09_final_test_topk.csv`

SHAP information-family importance:

`audit/10_shap_family_importance.csv`

---

## 10. Reproducing the modelling workflow

The full modelling and validation workflow is stored under:

`src/modelling/finals/`

Scripts are numbered in execution order from:

`01_audit_quarterly_dataset.py`

through:

`12_validate_dashboard_package.py`

The final test was opened only after model selection and risk-band policy were frozen.

Do not retune the model using the 2026Q2 final-test results and continue reporting those results as an independent unseen test.

---

## 11. Main communication rule

GeoQoE Pulse should be presented as:

> An independent GeoAI early-warning and prioritisation system that reveals current underperformance, anticipates where new underperformance may emerge, and helps prioritise where investigation is needed first.

It should NOT be presented as an automated root-cause diagnosis or network-planning engine.