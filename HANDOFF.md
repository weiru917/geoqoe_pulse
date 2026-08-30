# GeoQoE Pulse — Data & GIS Handoff

**Status:** final. Every figure below was computed from the shipped files, not from the code that produced them.
**Pipeline version:** 3.3.0 · **Regenerate:** `python run_pipeline.py` (~27 min, single entry point)
**Last run:** 0 errors · 492 columns documented across 5 tables · 0 undescribed · 21/21 independent tests passing

This is the data contract. If it disagrees with the data, the data is right and this is a bug — tell the Data/GIS owner.

**Scope:** `data/` and `src/data_preprocessing/` only. The dashboard and the trained-model package are owned by their teams and are untouched — where they need changing, this says *what* to change rather than changing it.

---

## 1. Read this first if you trained on the old data

The files you trained on (`Trained AI Model/…/pipeline_v3_2/`) are **superseded**. Three changes matter:

| | Then | Now |
|---|---|---|
| `cell_static_features` | 4,918 × **31** | 4,918 × **96** |
| `cell_static_qoe_baseline` | 4,918 × **35** | 4,918 × **113** |
| `cell_quarter_dataset` | **did not exist** | **41,448 × 146** ⭐ |

**No columns were removed.** Your existing feature list still resolves — it is purely additive there.

**But the hourly labels changed.** A timezone bug had the M-Lab diurnal curve applied 8 hours out of phase:

```
target_is_degraded changed on 6,743 rows (0.8%)
positive rate      2.234%  →  1.653%
peak degradation hour   10  →  22
```

Your onset model learned a daily pattern that was **inverted**. Its metrics do not describe the corrected data.

**`is_peak_period` also changed meaning, not just values.** It was a hardcoded commuter-hours assumption (08–10, 17–20); it is now derived from the realised curve (19:00–00:00). The old definition still ships as `is_commuter_period`. Peak hours now average 95 Mbps against 150 off-peak — before, "peak" hours were *faster*, so the feature pointed the wrong way.

**Both models need retraining.** The onset model should arguably be *replaced* — see §5.

---

## 2. Which file for which job

| Your job | Use | Why |
|---|---|---|
| **Forecast degradation** | `cell_quarter_dataset.parquet` | Only dataset with a real target measured on both sides |
| **Predict QoE where it is unmeasured** | `cell_static_qoe_baseline.parquet` | Real Ookla labels, real GIS features, zero synthetic QoE |
| **Dashboard map layers** | `data/processed/grid_geometry.geojson` + `cell_static_features.parquet` | Real geometry + all real static features, incl. `district_name` for roll-up |
| **Dashboard real time-series** | `cell_quarter_dataset.parquet` | 14 real quarters. Point the build at this |
| **Dashboard hourly demo** | `cell_time_dataset.parquet` | Labelled fixture — read §5 before showing any number from it |

**The one rule:** never report a forecast metric from `cell_time_dataset.parquet`. Use `cell_quarter_dataset.parquet`.

---

## 3. Start here — copy-paste

### Forecasting (recommended)

```python
import pandas as pd

df = pd.read_parquet("data/outputs/cell_quarter_dataset.parquet")
d  = pd.read_excel("data/outputs/feature_dictionary.xlsx")

features = d[(d.file == "cell_quarter_dataset")
             & (d.is_feature_for_next_hour_target)]["column"].tolist()   # 118 features

valid = df.target_next_q_valid == 1          # 36,193 rows with a REAL one-step-ahead label
train = valid & (df.quarter_index <  2026 * 4)   # 2023Q1–2025Q4, 33,364 rows
test  = valid & (df.quarter_index >= 2026 * 4)   # 2026Q1,        2,829 rows

y = "target_is_below_area_norm_next_q"       # ~22% positive
# alternative: target_is_degraded_next_q     # ~2% positive, near-degenerate
```

**The benchmark you must beat** (same split — see `quarterly_baseline_check.json`):

```
prior only          PR-AUC 0.213
persistence         PR-AUC 0.480
cell's own history  PR-AUC 0.689   ← the one to beat
XGBoost (100 feat)  PR-AUC 0.761   P@200 = 90.5%
```

Do not claim "0.76 PR-AUC". Say: *"a cell's own history already predicts this well; the model adds ~10% on top, and the top-200 list is 90% precise."* A knowledgeable judge will ask what your baseline was.

Weight or filter by `ookla_tests` — a cell-quarter can rest on very few real tests. Group on `h3_parent_res6` if you also want a spatial holdout.

### Spatial prediction

```python
df = pd.read_parquet("data/outputs/cell_static_qoe_baseline.parquet")
features = d[(d.file == "cell_static_qoe_baseline")
             & (d.is_feature_for_current_target)]["column"].tolist()   # 86 features (was 24)

labelled = df[df.has_real_ookla_label]
model.fit(labelled[features], labelled.ookla_dl_mbps,
          sample_weight=1.0 / labelled.coverage_propensity.clip(0.05, 1.0))   # ← §4
```

**Always read the feature list from the dictionary.** It is asserted complete on every run, so it picks up new columns automatically and never lets a target through.

---

## 4. Coverage bias — read before trusting any spatial prediction

Ookla covers ~60% of the grid, and **not randomly**. GIS features predict which cells it covers at **ROC-AUC 0.978**.

```
unlabelled cells                 1,968
…outside common support          1,681  (85.4%)
```

So "train on labelled, predict the rest" is **extrapolation for 85% of the cells you would predict**. This is the best explanation for the old static model's near-zero out-of-fold R² — a data property, not a model choice.

| Column | Use |
|---|---|
| `coverage_propensity` | Inverse-propensity weight when training |
| `in_common_support` | **Flag predictions where `False`** — do not show them at equal confidence |
| `ookla_label_confidence` | Weight by label reliability (median cell ≈ 34 tests, p25 ≈ 8) |

All three are excluded from the feature lists — weights and diagnostics, never predictors.

**Dashboard:** grey out cells where `in_common_support == False` rather than colouring them like the rest.

---

## 5. The hourly panel is a demo fixture

For Ookla-anchored cells the hourly value is exactly `ookla_static(cell) × mlab_curve(hour)` — one shared citywide curve, no noise term. Verified: dl/rtt reconstruct with **max absolute error 0.0**, and the reconstructed label matches the stored one on **100%** of anchored rows.

- **The forecast target is a lookup.** Given cell identity and hour-of-week it is fully determined. Any accuracy figure measures memorisation.
- **No cell-specific temporal dynamics exist.** No model can learn that one area degrades earlier than another — the information is not there.
- **Hotspot recurrence is noise.** Gi\* is invariant to a global scale factor, so categories move only via the Gaussian noise on unanchored cells.

Genuinely useful for the dashboard and for exercising the pipeline. Not a metric source. Full detail in `metadata.json → panel_structure_disclosure`.

---

## 6. What is new, and what it unlocks

### Real temporal axis — `cell_quarter_dataset.parquet` ⭐
41,448 rows · 3,566 cells × up to 14 quarters (2023Q1–2026Q2) · **3,959,197 real speed tests**.
Every value an independent Ookla measurement. 49.2% of cells change state over time (vs 0.8% on the hourly panel).

### Demand that varies over time
| Feature | Resolution | Varies over time |
|---|---|---|
| `viirs_rad_quarter`, `viirs_rad_yoy_log_ratio` | **cell** | 3,368 / 3,566 |
| `district_population`, `..._yoy_growth_pct` | 8 districts | 3,298 / 3,566 |
| `population_night`, POIs, built-up | cell | none |

Use `viirs_rad_yoy_log_ratio` for modelling — the percentage version is heavy-tailed on dark cells.

### New real static layers
Copernicus DEM (`elevation_mean_m`, `terrain_ruggedness_m`, 100% coverage), ESA WorldCover (all 11 classes), VIIRS stability, OSM building heights, the Sunway inset, and `district_name` for dashboard roll-up.

**Cite the Sunway data, never present it as our own fieldwork.** It is a published third-party dataset: Kabeer, Nordin & Behjati (2025), *An Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments*, Mendeley Data v1, [DOI 10.17632/dx5xyyfz2y.1](https://doi.org/10.17632/dx5xyyfz2y.1). Collected with GNetTrack Pro on Samsung S21 Ultra across 3 anonymised operators, 4G and 5G. Every row carries `sunway_source = "REAL (Mendeley dx5xyyfz2y)"`.

### Application profiles — `app_*`
Per-application pass/fail for voice, browsing, video calls, streaming and gaming, plus `app_user_impact_score` and `app_worst_failing`.

```
Worst failing:  Web browsing 766 · Video calls 597 · Voice 203
```

**The binding constraint is latency and upload, not download.** Thresholds come from engineering guidance (ITU-T G.114, codec requirements); weights come from MCMC's BQoES 2024 survey. The two are deliberately not conflated — a survey does not set technical thresholds.

**Dashboard:** `app_worst_failing` answers "what is actually broken here" in user language, which reads far better than a millisecond figure.

### Findings you can cite
- `demand_vs_degradation.json` — fastest activity-growth quartile is **1.55× more likely** to fall below the city norm next quarter (14.98% vs 9.69%, p = 3e-16), confounders ruled out
- `quarterly_baseline_check.json` — the model beats every trivial baseline on a real held-out quarter
- `separability_test.json` — the hourly panel's shared-shape assumption **fails** when tested
- `source_manifest.json` — every source used, with re-fetch URLs

### Two findings to quote verbatim

**On capacity — when asked "why didn't you use network capacity data?":**

> *"We went to the regulator's own register. It publishes no site coordinates — only internal site codes — so nothing joins to a spatial grid. And every sector we retrieved was 700 or 800 MHz coverage spectrum from U Mobile, YTL and DNB; Maxis and CelcomDigi don't appear. The bands that actually determine throughput are held under national assignments with no site-level geography, so even with coordinates the register couldn't explain a speed difference."*

Backing evidence: 30 sector rows / 10 sites retrieved 2026-08-30 — **100% sub-1 GHz, 0 capacity sectors.** U Mobile and DNB on Band 28 (TX 788/768, RX 733/713), YTL on Band 20 (TX 800.5, RX 841.5 — reverse duplex). Raw rows in `data/raw/mcmc_register/`, decoding in `scripts/analysis/check_mcmc_register_bands.py`, finding in `data/outputs/mcmc_register_findings.json`.

**Lead with the spectrum argument, not the geography one** — it holds even if coordinates were published. **State it as a property of the data, never as "capacity doesn't matter."** Capacity absolutely drives speed; the register simply doesn't contain it.

*Supersedes an earlier "Spearman −0.065 (p = 0.69) across 40 matched cells" claim. That was not wrong, but it rested on district-name matching and its per-cell counts were never persisted — a correlation invites "your sample was too small", whereas "the data contains none of the relevant spectrum" does not.*

**On why the QoE weights are what they are:**

> *"Our index weights latency heavily because MCMC's national survey shows online voice calls and web browsing are the highest-importance activities for Malaysian mobile users — 4.31 and 4.24 out of 5 — and both carry the largest satisfaction gaps, −0.34 and −0.31."*

Source: MCMC Broadband Quality of Experience Survey 2024, mobile broadband results, 2,304 respondents. See §7 for how it is used and what it cannot do.

---

## 7. MCMC BQoES 2024 — cited, never a pipeline input

The survey is **not** in the pipeline and cannot be: it is a consumer opinion survey with no coordinates, no timestamps and no measurements, and its finest geography is urban vs rural. It cannot join the H3 grid.

It is used for exactly two things, and both belong in slides rather than features:

**1. Justifying the QoE index weights.** Mobile-broadband results, national:

| Activity | Importance (1–5) | Satisfaction gap | Technically driven by |
|---|---|---|---|
| Online voice calls | **4.31** | **−0.34** | latency, jitter, loss |
| Web browsing | 4.24 | −0.31 | latency, then throughput |
| Online video calls | 4.09 | −0.19 | throughput + jitter, both directions |
| Video streaming | 3.88 | +0.04 | sustained download |
| Online gaming | 3.47 | +0.33 | latency |

The two activities Malaysians value most are both latency-driven and both are where operators fall furthest short. That replaces "we chose these weights" with regulator evidence.

**2. Translating metrics into user language.** The survey's complaint breakdown maps onto metrics you already have — video stalls 52.4% → throughput; choppy voice 49.0% → jitter and loss; dropped calls 24.0% → coverage; waiting for pages 24.0% → latency. This is what `app_worst_failing` is built on.

**Keep the two sources separate.** Thresholds in `application_profiles.py` come from engineering guidance (ITU-T G.114, codec requirements). BQoES supplies *weights only*. Presenting a satisfaction survey as if it set technical requirements would be a category error.

One caveat to state yourself before anyone asks: 29.9% of Malaysia's mobile users are rural, and this study area is entirely urban Klang Valley.

---

## 8. What you can and cannot claim

**Defensible:**
- "Twelve real sources, zero synthetic fallbacks fired across all 4,918 cells" — every `*_is_synthetic` flag is `False`, checksums and vintages in `metadata.json`
- "Degradation recurrence measured across 14 real quarters"
- "Next-quarter forecast validated on a real held-out quarter, beating persistence and history baselines"
- "Degradation is **associated with** rising local activity — 1.55× relative risk, mean reversion ruled out"
- "Independent real ground truth exists" — 30,925 field measurements, 3 operators, 10 cells

**Not defensible:**
- Any hourly forecast accuracy (§5)
- "Confirmed congestion" or "demand **caused** degradation" — no capacity data exists (§9)
- Comparing `sunway_test_event_dl_mbps` to Ookla — different quantities (§9)
- Claiming the GIS features drive the forecast — QoE history dominates; the GIS layer's value is the *spatial* model

---

## 9. Known limits — bounded, documented, not defects

1. **No capacity data, and this was verified against the register itself.** Malaysia's public apparatus register cannot supply a per-cell capacity feature, for two independent reasons — either alone is sufficient:

   **(a) It publishes no site coordinates.** Only operator-internal site codes, e.g. `LGAT(BT14 HULU LANGAT)`. Nothing joins to a hexagon; the finest achievable match is district name — 8 districts across 4,918 cells. Name matching also crosses states: a search for the study area's `SEPANG` returns `TLKSEPANGGAR` — Telok Sepanggar, Kota Kinabalu, ~1,600 km away.

   **(b) It contains none of the spectrum that carries throughput.** Every sector retrieved was **Band 28 (700 MHz)** or **Band 20 (800 MHz)** — sub-1 GHz *coverage* spectrum, which determines reach and building penetration, not speed. Holders were U Mobile, YTL and DNB; **Maxis and CelcomDigi do not appear at all.** Their capacity spectrum (1800/2100/2600 MHz, 5G mid-band) sits under national Spectrum Assignments with no site-level geography.

   Reason (b) is the one to lead with: it holds **even if MCMC published exact coordinates tomorrow**.

   **Both tiers of the regulator's records were checked, and both fail.** Above the apparatus register sits the national **Spectrum Assignment** register — where Maxis and CelcomDigi do appear (900 MHz, 5–10 MHz paired each, 2017–2032). It has **no geography at all**, not even a site code: assignments are nationwide, so every one of 4,918 cells would receive an identical value. Per-operator bandwidth is real quantity, but Ookla aggregates across operators, and Sunway's operators are anonymised by its authors — so there is nothing to join it to from either direction. Finding the 2600 or 3500 MHz tables would not help: they share the same aspatial structure. In Malaysia spectrum is licensed nationally, and geography exists only at apparatus level, where the capacity bands are absent.

   Evidence in `data/outputs/mcmc_register_findings.json` and `data/raw/mcmc_register/`; reproduce with `python scripts/analysis/check_mcmc_register_bands.py`. The public data cannot support a capacity claim, so we make none.

2. **Demand resolution.** VIIRS gives per-cell quarterly activity; DOSM gives 8 districts annually. VIIRS measures *light* — a proxy for activity, not mobile traffic. Say "associated with rising activity".

3. **Ookla is consumer-initiated.** People run speed tests when the network feels slow. Low scores partly mean "people here were annoyed enough to check". Weight by `ookla_tests`.

4. **Sunway cannot calibrate Ookla.** Its download test averages ~0.5 Mbps where Ookla measures ~150 in the same cells — verified two ways; a short constrained transfer versus a saturating speed test. Compare Sunway cells only with each other.

5. **Building heights are sparse and biased** — 6.5% of buildings tagged, 21% of cells, clustered where OSM volunteers map. `building_height_coverage_is_biased = True`. Never impute.

6. **DEM is a surface model** — includes buildings and canopy, so `elevation_mean_m` is not ground height and correlates with the building layers by construction.

7. **Voice and video-call profiles are partial** — Ookla has no jitter or packet loss, so a cell can pass every measurable criterion and still sound broken. Check `app_n_evaluable`.

8. **Mixed vintages.** Population 2020–2025, land cover 2021, OSM 2026, Ookla 2023–2026. The hourly panel's dates are a synthetic calendar week. Per-source `vintage` in `metadata.json`.

---

## 10. Guarantees

- **Keys unique** on every table; target table keys match the panel exactly
- **CRS discipline** — WGS84 for storage, EPSG:3375 for every distance, area and length
- **No leakage** — lags use `shift()` before any window; scalers fit on the training window only (parameters in `metadata.json → fitted_scalers`); quarterly lags validated against real quarter gaps so uneven coverage cannot masquerade as one-step-ahead
- **Build-time guards that fail the run:** dictionary completeness · diurnal timezone alignment · derived-layer passthrough (every extracted column must reach the outputs) · derived-layer freshness (no layer older than its own inputs)
- **Deterministic** — single seed (42), set once, recorded every run

Last run: **0 errors, 32 warnings** — all warnings are disclosures, not faults. Read them in `data_quality_report.json`.

### Independent verification — `python -m pytest tests/ -q` (21 tests, ~7 s)

The build-time guards above check internal consistency, which catches a
*broken* build but not a *wrong but self-consistent* one. `tests/` checks the
outputs against the outside world instead, and is worth running after any
change to the grid, the joins, or a source.

**Agreement between sources that never coordinated** — if a spatial join were
misplacing data, these would collapse:

| | Spearman ρ |
|---|---|
| WorldPop vs NASA VIIRS | **+0.93** |
| ESA WorldCover vs OSM buildings | **+0.96** |
| WorldPop vs OSM roads | +0.85 |
| Tree cover vs built-up | −0.69 |

**Landmarks match reality** — KLCC bright and dense, Hulu Langat forest at 96%
tree cover and 182 m, KLIA 94% built-up with 523 people/km² (a large structure
where nobody sleeps — the sharpest test that cells are not offset).

**No leakage** — highest feature-to-target correlation is **0.24**; nothing is
a disguised copy of the label.

**The data is genuinely predictive.** On a forward-in-time split (train
2023Q1–2025Q2, test 2025Q3–2026Q1) against `target_is_below_area_norm_next_q`:

```
prevalence            0.220
model PR-AUC          0.720
persistence baseline  0.494   <- the number to beat, not chance
ROC-AUC               0.895
```

Persistence — "assume next quarter looks like this one" — is the honest
baseline. Beating chance is easy; beating persistence is what shows the
features carry information.

---
