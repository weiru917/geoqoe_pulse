# GeoQoE Pulse Finals AI — Model Card

Model version: `finals_quarterly_onset_v1.0.0`

Model family: `XGBoost Classifier`

Status: Final model frozen and evaluated on unseen 2026Q1 → 2026Q2 data

---

## 1. Purpose

The GeoQoE Pulse finals AI model provides an early-warning ranking of currently-normal geographic cells that may become relatively underperforming in the following quarter.

The intended use is decision support for regulators or mobile-network stakeholders.

The model helps answer:

> Which currently-normal locations deserve investigation first because their network performance is at higher risk of falling behind the rest of Klang Valley next quarter?

The model is designed for:

- emerging-risk forecasting,
- spatial prioritisation,
- transparent model interpretation,
- dashboard decision support.

It is not designed for automatic root-cause diagnosis or engineering intervention selection.

---

## 2. Geographic unit

The modelling unit is an Uber H3 geographic cell.

The frontend uses `h3_index` as the unique spatial join key.

Each model row represents one H3 cell during one quarter.

---

## 3. Definition of underperformance

For every quarter, measured cells are ranked by Ookla download performance.

The dataset contains:

`dl_percentile_in_quarter`

which represents the cell's download-speed percentile among measured cells in the same quarter.

The current-quarter underperformance label is:

`is_below_area_norm`

Definition:

`is_below_area_norm = 1` when `dl_percentile_in_quarter < 25`

In plain language:

> The cell is in the bottom 25% of measured Klang Valley download performance for that quarter.

This is a relative citywide benchmark.

It is not a fixed minimum-speed threshold.

---

## 4. Prediction target

The model target is:

`target_is_below_area_norm_next_q`

This represents whether the same H3 cell falls into the bottom 25% in the following quarter.

The finals emerging-risk task further restricts modelling to cells where:

`is_below_area_norm == 0`

and:

`target_next_q_valid == 1`

Therefore, the actual modelling question is:

> Among cells that are currently outside the bottom 25%, which will newly enter the bottom 25% next quarter?

This excludes currently-underperforming cells because those are already observable current issues rather than emerging issues.

---

## 5. Dataset

Primary modelling dataset:

`data/outputs/cell_quarter_dataset.parquet`

Quarterly coverage:

`2023Q1 through 2026Q2`

The quarterly panel is based on real quarterly Ookla measurements.

The older hourly panel should not be used for finals forecasting-performance claims.

---

## 6. Input features

The final model contains 44 predictors.

They are grouped into three broad information families.

### QoE / performance history

Examples include:

- current download speed,
- current upload speed,
- current latency,
- current download percentile,
- previous-quarter download speed,
- one-year lagged download speed,
- rolling download statistics,
- previous-quarter latency,
- rolling latency statistics,
- historical underperformance recurrence,
- historical underperformance rate.

### Geographic / infrastructure context

Examples include:

- tower density,
- distance to nearest tower,
- building density,
- built-up intensity,
- built-up land coverage,
- road density,
- intersection density,
- transit accessibility,
- POI counts,
- land-cover characteristics,
- terrain characteristics.

### Activity / population context

Examples include:

- population density,
- local population,
- district population,
- district population growth,
- nighttime-light activity,
- year-on-year nighttime-light change.

VIIRS nighttime light is treated as an activity proxy.

It is not treated as direct measurement of mobile-network demand.

---

## 7. Feature leakage controls

The model does not use future target variables as predictors.

Excluded future information includes:

- `target_is_degraded_next_q`
- `target_is_below_area_norm_next_q`
- `target_dl_mbps_next_q`
- `target_next_q_valid`

H3 IDs are used as keys, not model predictors.

Sunway case-study variables are not used as citywide predictors.

Ookla test count is not used as a training weight in the final model.

---

## 8. Training and model selection

The development process used a chronological split.

Initial training period:

`2023Q1 through 2025Q2`

Validation period:

`2025Q3 and 2025Q4`

Final sealed test:

Predictor quarter:

`2026Q1`

Outcome quarter:

`2026Q2`

The final test was not used for:

- feature selection,
- hyperparameter selection,
- model-family selection,
- risk-band selection.

After model selection was frozen, training plus validation data through 2025Q4 were used to fit the final model.

---

## 9. Final XGBoost configuration

Selected configuration:

`less_regularized`

Final hyperparameters:

- `n_estimators = 23`
- `max_depth = 3`
- `learning_rate = 0.03`
- `min_child_weight = 3`
- `gamma = 0.0`
- `subsample = 0.90`
- `colsample_bytree = 0.90`
- `reg_alpha = 0.2`
- `reg_lambda = 3.0`
- `objective = binary:logistic`
- `eval_metric = aucpr`
- `tree_method = hist`
- `random_state = 42`

Final training weighting:

`none`

Ookla sample counts remain available separately for measurement-support display.

---

## 10. Why PR-AUC is the primary metric

Only a minority of currently-normal cells become underperforming next quarter.

In the final test:

205 of 2,186 cells became underperforming.

Event rate:

`9.38%`

For an uninformative ranking, PR-AUC is approximately equal to the positive event rate.

Therefore the no-skill reference is approximately:

`0.094`

The final model achieved:

`PR-AUC = 0.449`

PR-AUC evaluates how effectively the model ranks genuine future problems above non-problems while considering the trade-off between precision and recall across many possible operating thresholds.

Important:

`PR-AUC = 0.449` does NOT mean:

- 44.9% accuracy,
- 44.9% probability,
- 44.9% of all cells correctly predicted.

It is a ranking-quality metric.

---

## 11. Final unseen-test results

Final test:

Predictor quarter:

`2026Q1`

Outcome quarter:

`2026Q2`

Eligible currently-normal cells:

`2,186`

Actual emerging underperformance events:

`205`

Event rate:

`9.38%`

Final full XGBoost:

- PR-AUC: `0.449`
- ROC-AUC: `0.857`

Comparison models:

- XGBoost QoE-history only PR-AUC: `0.445`
- Historical recurrence baseline PR-AUC: `0.386`
- Current percentile baseline PR-AUC: `0.171`
- Constant/no-skill PR-AUC: `0.094`

The full model therefore achieved the best overall final-test ranking.

---

## 12. Top-K interpretation

Top-K evaluates the model from an operational prioritisation perspective.

The model ranks every eligible cell from highest to lowest emerging risk.

Top-50 means:

> the 50 highest-ranked cells.

Top-100 means:

> the 100 highest-ranked cells.

Top-200 means:

> the 200 highest-ranked cells.

Final full-model results:

### Top 50

32 of the 50 highest-ranked cells became underperforming.

Precision:

`32 / 50 = 64%`

### Top 100

57 of the top 100 became underperforming.

Precision:

`57 / 100 = 57%`

### Top 200

92 of the top 200 became underperforming.

Precision:

`92 / 200 = 46%`

The Top-200 captured:

`92 / 205 = 44.9%`

of all emerging events in the final test.

The 44.9% value is recall within the Top-200.

It is not model accuracy.

---

## 13. Frozen dashboard risk bands

The dashboard converts risk ranking into three operational groups.

### High emerging risk

Ranks:

`1–50`

Final-test result:

32 / 50 became underperforming.

Observed group event rate:

`64%`

### Elevated emerging risk

Ranks:

`51–200`

Final-test result:

60 / 150 became underperforming.

Observed group event rate:

`40%`

### Lower emerging risk

Ranks:

`201+`

Final-test result:

113 / 1,986 became underperforming.

Observed group event rate:

`5.7%`

These percentages describe what happened to the groups in the final test.

They are NOT probabilities assigned to individual cells.

---

## 14. Raw XGBoost score

The model produces a continuous output between 0 and 1.

This output is used for relative ranking.

The score was not separately calibrated as a probability.

Therefore the dashboard should not display:

> 34% probability of degradation

Prefer:

> High emerging risk

and:

> Rank 8 of 2,186

The raw score may remain internally available for sorting.

---

## 15. SHAP explanation

XGBoost SHAP contributions are calculated for each final-test prediction.

A positive SHAP contribution means that the corresponding feature pushed the model toward a higher-risk prediction.

A negative SHAP contribution means it pushed the model toward lower risk.

SHAP explains model behaviour.

It does NOT establish causality.

Do not describe SHAP outputs as:

- root causes,
- proven congestion causes,
- proof that demand caused degradation.

Recommended wording:

> Factors contributing to the model's emerging-risk assessment

---

## 16. Global SHAP findings

Approximate share of absolute SHAP influence:

- QoE / performance history: `80.6%`
- Geographic / infrastructure context: `14.1%`
- Activity / population context: `5.3%`

These percentages describe model influence.

They do NOT mean that:

> QoE history contributes 80.6% of model accuracy.

The most influential individual features included:

- historical underperformance rate,
- current upload speed,
- built-up land coverage,
- current citywide download percentile,
- recent average download speed,
- previous-quarter download speed,
- current download speed,
- estimated local population,
- population density.

---

## 17. Measurement support

Ookla is crowdsourced and the number of measurements varies between cells.

The final dashboard package includes a measurement-support category based on `ookla_tests`.

Thresholds:

### Limited

`< 20 tests`

### Typical

`20–149 tests`

### Strong

`>= 150 tests`

This represents measurement support only.

It does not mean:

- prediction accuracy,
- model confidence,
- probability of correctness.

---

## 18. Final model limitations

The model has several important limitations.

1. The forecast target is relative download-performance underperformance, not complete network QoE.

2. The model does not prove that congestion occurred.

3. Geographic and activity features may be associated with emerging risk but do not establish causal relationships.

4. The model cannot identify the exact engineering root cause of degradation.

5. It does not determine whether an operator should build a tower, add spectrum, rebalance load, adjust antennas or perform another intervention.

6. Ookla measurements are crowdsourced and spatial sampling density is uneven.

7. The model score is not a calibrated probability.

8. The final evaluation covers one fully sealed future quarter, 2026Q1 → 2026Q2.

---

## 19. Claims supported by the model

Supported:

> The model ranks currently-normal locations by emerging next-quarter underperformance risk.

Supported:

> The final model achieved PR-AUC 0.449 on an unseen 2026Q1 → 2026Q2 test.

Supported:

> 32 of the Top-50 ranked cells became underperforming.

Supported:

> The Top-200 captured 92 of 205 emerging events.

Supported:

> Historical performance provides most of the model's predictive influence, while geography and activity provide complementary information.

Not supported:

> The model proves demand caused congestion.

Not supported:

> The AI identifies the root cause.

Not supported:

> A specific cell requires a new tower.

Not supported:

> The raw model score is the probability that degradation will occur.

---

## 20. Final intended role

GeoQoE Pulse should be positioned as:

> An independent GeoAI early-warning and prioritisation system that helps stakeholders reveal current underperformance, anticipate where new underperformance may emerge, and prioritise where investigation is needed most.