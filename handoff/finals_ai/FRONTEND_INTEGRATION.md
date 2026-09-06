# GeoQoE Pulse Finals AI — Frontend Integration Guide

This document describes how the finals AI outputs should be translated into the GeoQoE Pulse dashboard.

---

## 1. Primary frontend file

Recommended file:

`dashboard_assets/2026Q1_to_2026Q2_cells.geojson`

The GeoJSON contains both:

- H3 geometry
- final AI/dashboard attributes

for 2,186 forecast-eligible cells.

Equivalent tabular versions are available as:

`dashboard_assets/2026Q1_to_2026Q2_cells.parquet`

and:

`dashboard_assets/2026Q1_to_2026Q2_cells.csv`

---

## 2. Join key

Primary spatial key:

`h3_index`

Every GeoJSON feature represents one H3 cell.

The final package passed structural QA with:

- 2,186 table rows
- 2,186 GeoJSON cells
- 0 duplicate H3 IDs
- 0 missing geometry IDs
- 0 extra geometry IDs

---

## 3. Forecast scope

The package represents:

Predictor quarter:

`2026Q1`

Forecast / outcome quarter:

`2026Q2`

Important:

These 2,186 cells were all currently outside the bottom 25% in 2026Q1.

Therefore they are eligible for emerging-risk prediction.

Cells already underperforming in 2026Q1 are not included in this forecast package.

Those should be shown separately using current observed data.

Recommended dashboard logic:

Current underperforming
→ display as observed current issue

Currently normal
→ display AI emerging-risk forecast

---

## 4. Core risk fields

### `risk_rank`

Integer rank within the eligible population.

Example:

`1`

means highest emerging risk among all eligible cells.

Example:

`200`

means 200th highest risk.

Lower number = higher priority.

---

### `risk_band`

Machine-readable category.

Possible values:

`high`

`elevated`

`lower`

---

### `risk_band_label`

Human-readable label.

Recommended display values:

`High emerging risk`

`Elevated emerging risk`

`Lower emerging risk`

Use this field for normal UI display.

---

### `risk_percentile`

Relative ranking position within the eligible cell population.

This is NOT a probability.

It may be used as secondary ranking information if needed.

---

### `xgb_full`

Raw final XGBoost score.

Use internally for ordering if necessary.

Do NOT display this as:

> Probability of degradation

The score has not been calibrated as a probability.

Recommended user-facing representation:

`risk_rank`

plus:

`risk_band_label`

Example:

> High emerging risk  
> Rank 8 of 2,186

---

## 5. Frozen dashboard risk policy

### High emerging risk

Definition:

`risk_rank <= 50`

Final unseen-test event rate:

`64%`

Meaning:

32 of the 50 High-risk cells became underperforming.

---

### Elevated emerging risk

Definition:

`51 <= risk_rank <= 200`

Final unseen-test event rate:

`40%`

Meaning:

60 of 150 Elevated-risk cells became underperforming.

---

### Lower emerging risk

Definition:

`risk_rank >= 201`

Final unseen-test event rate:

`5.7%`

Meaning:

113 of 1,986 Lower-risk cells became underperforming.

Important:

These percentages are observed group outcomes from the final test.

Do not show them as the probability for an individual cell.

---

## 6. Recommended forecast map

Before revealing the future, style the map using:

`risk_band`

Recommended visual hierarchy:

High
→ strongest warning styling

Elevated
→ secondary warning styling

Lower
→ subdued styling

Current already-underperforming cells should use a visually different observed-status style because they are not forecasts.

The user should clearly be able to distinguish:

Observed current problem

from:

AI-predicted emerging risk

Recommended labels:

> Current underperformance

and:

> Emerging risk

---

## 7. Time Machine

The finals Time Machine demonstrates the forecast using historical real data.

Workflow:

1. Display information available in 2026Q1.
2. Display the model's risk ranking for 2026Q2.
3. User clicks "Reveal Actual".
4. Display the real 2026Q2 outcome.
5. Re-style cells by prediction result.

Field:

`time_machine_result`

Possible values:

### `correctly_anticipated`

The cell was in the Top-200 forecast and actually became underperforming.

Final-test count:

`92`

Suggested display:

> Correctly anticipated

---

### `false_alert`

The cell was in the Top-200 but did not become underperforming.

Final-test count:

`108`

Suggested display:

> Flagged, but remained outside the underperforming group

or shorter:

> False alert

---

### `missed`

The cell was outside the Top-200 but later became underperforming.

Final-test count:

`113`

Suggested display:

> Missed emerging issue

---

### `correctly_lower_risk`

The cell was outside the Top-200 and did not become underperforming.

Final-test count:

`1,873`

Suggested display:

> Correctly lower-risk

---

## 8. Current network measurement fields

Useful current-quarter fields include:

### `ookla_dl_mbps`

Current measured download performance.

Recommended label:

> Download

Unit:

`Mbps`

---

### `ookla_ul_mbps`

Current measured upload performance.

Recommended label:

> Upload

Unit:

`Mbps`

---

### `ookla_rtt_ms`

Current measured round-trip latency.

Recommended label:

> Latency

Unit:

`ms`

---

### `ookla_tests`

Number of Ookla tests contributing to the cell-quarter observation.

Recommended label:

> Ookla tests

Do not describe this as model confidence.

---

### `dl_percentile_in_quarter`

Current download-performance percentile among measured cells in the quarter.

The model target threshold is:

`< 25`

---

## 9. Measurement support

Field:

`measurement_support`

Possible values:

`limited`

`typical`

`strong`

Thresholds:

Limited:

`ookla_tests < 20`

Typical:

`20 <= ookla_tests < 150`

Strong:

`ookla_tests >= 150`

Recommended UI:

> Measurement support: Strong

Optional tooltip:

> Based on the number of Ookla measurements contributing to this cell-quarter observation.

Do NOT label this:

> AI confidence

or:

> Prediction confidence

because it only represents measurement evidence.

---

## 10. SHAP explanation fields

The dashboard package contains up to three positive risk contributors.

Fields:

`risk_driver_1_label`

`risk_driver_1_value`

`risk_driver_1_family`

`risk_driver_2_label`

`risk_driver_2_value`

`risk_driver_2_family`

`risk_driver_3_label`

`risk_driver_3_value`

`risk_driver_3_family`

Recommended panel heading:

> Why is this area flagged?

Recommended subheading:

> Factors contributing to the model's emerging-risk assessment

Example:

> Historical underperformance rate — 0.67

> Current upload speed — 8.4 Mbps

> Recent average download speed — 78.2 Mbps

Important:

Some cells have fewer than three positive risk contributors.

Frontend logic should therefore:

- show contributor 1 if available,
- show contributor 2 if available,
- show contributor 3 if available,
- hide missing rows.

Do not display blank placeholders.

---

## 11. Risk-reducing SHAP fields

Optional fields:

`protective_1_label`

`protective_1_value`

`protective_1_family`

`protective_2_label`

`protective_2_value`

`protective_2_family`

These represent model inputs that pushed the prediction toward lower risk.

They are optional for the finals UI.

If included, suggested heading:

> Factors reducing the model's risk assessment

---

## 12. SHAP interpretation warning

SHAP describes model contribution, not causality.

Allowed:

> High built-up coverage contributed to the model's emerging-risk assessment.

Avoid:

> High built-up coverage caused the degradation.

Allowed:

> Historical underperformance was an important model signal.

Avoid:

> Historical underperformance proves congestion.

Do not label the explanation panel:

> Root cause

Preferred:

> Why is this area flagged?

or:

> Investigation evidence

---

## 13. Context fields

Useful contextual variables include:

### Population

`population_density_per_km2`

`population_day`

`district_population`

Recommended display:

> Population exposure

or:

> Population context

---

### Activity

`viirs_rad_quarter`

`viirs_rad_yoy_log_ratio`

Recommended wording:

> Nighttime activity proxy

Do not call VIIRS:

> mobile demand

It is a contextual activity proxy.

---

### Built environment

`building_density_per_km2`

`built_up_ratio`

`landcover_built_up_pct`

Recommended labels:

> Building density

> Built-up intensity

> Built-up land coverage

---

### Infrastructure

`tower_density_per_km2`

`dist_to_nearest_tower_m`

`road_density_km_per_km2`

`intersection_density_per_km2`

`dist_to_nearest_transit_m`

These are contextual indicators.

Do not interpret them automatically as engineering root causes.

---

## 14. Recommended selected-cell panel

Suggested structure:

### Emerging Risk

`High emerging risk`

`Rank 8 of 2,186`

---

### Current Network Performance

Download: `63.4 Mbps`

Upload: `12.7 Mbps`

Latency: `24 ms`

---

### Why is this area flagged?

Historical underperformance rate: `...`

Current upload performance: `...`

Recent average download performance: `...`

Only display available SHAP contributors.

---

### Area Context

Population density: `...`

Built-up coverage: `...`

Nighttime activity trend: `...`

Nearby tower density: `...`

---

### Measurement Evidence

Ookla tests: `...`

Measurement support: `Strong`

---

### Interpretation note

> These factors influenced the model's emerging-risk ranking. They do not establish the cause of future network degradation.

---

## 15. Recommended main workflow

The finals dashboard should follow:

### Reveal

What is happening now?

Display real current-quarter underperformance.

---

### Anticipate

What may happen next?

Display AI emerging-risk ranking for currently-normal cells.

---

### Explain

Why did the model flag this location?

Display SHAP contributors plus geographic/activity context.

---

### Prioritise

Which locations deserve investigation first?

Use:

- risk rank,
- risk band,
- population/context,
- measurement support,
- relevant facilities if required.

---

## 16. Wording to use

Recommended:

> Current underperformance

> High emerging risk

> Elevated emerging risk

> Lower emerging risk

> Rank 8 of 2,186

> Factors contributing to the model's emerging-risk assessment

> Measurement support: Strong

> Prioritise for investigation

> This degradation coincides with...

> This pattern is consistent with...

---

## 17. Wording to avoid

Do not use:

> 34% probability of congestion

Do not use:

> AI detected congestion

Do not use:

> Demand caused this degradation

Do not use:

> Root cause identified

Do not use:

> Build a tower here

Do not use:

> Add capacity here

unless those statements are clearly presented as separate engineering investigation scenarios rather than model predictions.

---

## 18. Important distinction: prediction vs prioritisation

The model predicts:

> emerging relative underperformance risk

The dashboard may then use other contextual information to prioritise investigation.

Do not mix these concepts into one unexplained score.

Recommended architecture:

AI emerging-risk forecast

+

population / activity / infrastructure context

+

measurement support

→

prioritised investigation

---

## 19. Main finals map payload

For the historical validation / Time Machine demo, start with:

`dashboard_assets/2026Q1_to_2026Q2_cells.geojson`

The package has already passed QA for:

- complete H3 geometry matching,
- unique cell IDs,
- complete risk ranking,
- frozen risk-band logic,
- Time Machine result consistency.

No additional model computation is required by the frontend for this demo.