# GeoQoE Pulse

**A mobile network quality decision-support prototype for the Klang Valley (Kuala Lumpur & surrounding urban Selangor).**

GeoQoE Pulse takes real, open geospatial and network data, combines it into
one map, and helps a telecom operator or regulator answer four connected
questions:

1. **REVEAL:** Which measured cells are already underperforming in 2026Q1?
2. **ANTICIPATE:** Which currently-normal cells should be watched for 2026Q2?
3. **EXPLAIN:** Which model contributors and contextual evidence support review?
4. **PRIORITISE:** Which locations should an operator or regulator investigate first?

This document is the single entry point for judging — what's inside the
submission, where every dataset and model actually lives on disk, how the
pieces fit together, and the fastest way to open the working prototype.

> **Current finals note (2026Q1 → 2026Q2):** The active dashboard uses the
> frozen quarterly finals package and a separate real 2026Q1 observed-status
> layer. The older next-hour AI files are retained for provenance/legacy
> comparison only and are not the validated default workflow.

---

## Table of contents

1. [30-second orientation](#1-30-second-orientation)
2. [How it works, in plain terms](#2-how-it-works-in-plain-terms)
3. [How to open it (for judging)](#3-how-to-open-it-for-judging)
4. [The datasets, in detail](#4-the-datasets-in-detail)
5. [The AI models, in detail](#5-the-ai-models-in-detail)
6. [The dashboard, in detail](#6-the-dashboard-in-detail)
7. [What's real, and what's a demo stand-in](#7-whats-real-and-whats-a-demo-stand-in-read-before-quoting-a-number)
8. [Full repository map](#8-full-repository-map)
9. [Reproducing the pipeline / retraining the models](#9-reproducing-the-pipeline--retraining-the-models)
10. [Known limitations](#10-known-limitations)

---

## 1. 30-second orientation

This submission has three layers that build on each other:

| Layer | What it does | Where it lives |
|---|---|---|
| **① Data pipeline** | Pulls real open data (cell towers, buildings, roads, population, transit, and real Ookla speed-test data) for **4,918** hexagonal map cells across the Klang Valley, and turns it into one clean, analysis-ready dataset. | `src/`, `data/`, `run_pipeline.py`, `config.py` |
| **② AI models** | The finals model ranks currently-normal cells by emerging underperformance risk in the next quarter. Earlier static and next-hour models remain available as legacy/demo artifacts, not the validated default forecast. | `models/finals_quarterly/`, `handoff/finals_ai/`, `Trained AI Model/GeoQoE_AI/` |
| **③ Interactive dashboard** | A map-based decision-support app combining real quarterly observed status, quarterly AI ranking, model contributors, Time Machine evaluation, and a clearly separated hourly Demo / Scenario lab. | `dashboard/build/` |

**The dashboard (layer ③) is the prototype itself — that's what to open
first (see §3).** Layers ① and ② are the data/AI engine behind it, included
in full — raw-to-model code, trained model files, and every evaluation
report — so judges can inspect the entire pipeline, not just the front end.

---

## 2. How it works, in plain terms

1. **Build the map grid.** The Klang Valley is divided into **4,918**
   hexagonal cells (H3 grid, resolution 8, ~0.86 km² each, ~461 m edge —
   roughly a few city blocks).
2. **Attach real-world context to every cell.** For each cell: number of
   cell towers, building density, road/POI/transit density, resident and
   working population, and — where available — a **real** quarterly
   average internet speed from Ookla's open speed-test dataset (**2,950 of
   4,918 cells, ~60%**, carry a real Ookla label).
3. **Score network quality.** A custom 0–100 Quality-of-Experience (QoE)
   index is calculated from download/upload speed, latency, jitter, and
   packet loss. Cells are then flagged against two targets: the
   **primary relative target** (`is_below_area_norm` — bottom 25% of
   citywide download performance that quarter, prevalence stable at ~22%
   across all 14 quarters), which is what the finals model forecasts, and
   a **legacy absolute flag** (download < 10 Mbps or latency > 100 ms),
   retained for regulatory familiarity. The absolute threshold was
   superseded because Malaysian speeds rose from ~141 to ~192 Mbps over
   the study window, shrinking it to ~3% prevalence and falling — see
   `DATA_METHODOLOGY.md` §4.2.
4. **Find hotspots.** Real spatial statistics (Getis-Ord Gi\*, Local
   Moran's I) identify statistically significant clusters of poor quality,
   not just individually bad cells.
5. **Rank next-quarter emerging risk.** The finals XGBoost model uses the
   2026Q1 quarterly state to rank 2,186 currently-normal, evaluable cells
   for possible underperformance in 2026Q2. High means ranks 1–50,
   Elevated means 51–200, and Lower means 201 onward. The score is a
   relative ranking, not a calibrated probability.
6. **Explain without claiming causality.** SHAP values describe which
   features contributed to the model's ranking. Context and scenario
   signals guide investigation but are not diagnosed root causes.
7. **Show it in one decision workflow.** The dashboard follows
   REVEAL → ANTICIPATE → EXPLAIN → PRIORITISE, while retaining hourly
   playback, hotspots, comparison and scenario tools under explicit
   DEMO / SCENARIO labels.
8. **Add public context after the fact, never before.** A prioritised cell
   can be checked against publicly scheduled events and holidays in the
   forecast quarter. This runs *after* the ranking and cannot influence it;
   an event is shown only if its source URL came from a page the search
   actually returned. See §6.

---

## 3. How to open it (for judging)

The dashboard is a self-contained web app — **no Python environment or
installation required.** All the data it needs has already been generated
and sits in `dashboard/build/data/` and `dashboard/build/ai_model/`.

```bash
cd dashboard/build
node server.js 3000
```

Then open **http://localhost:3000** in a browser. Node ships a zero-dependency
server in this repo — nothing to install.

> **Use Node, not `python3 -m http.server`.** The Python static server cannot
> answer the `POST /api/context-brief` route that the **Find local context**
> button needs, so that button silently reports "no corroborated upcoming
> context found" for every cell — indistinguishable from a genuine empty
> result. Everything else on the dashboard works fine under Python.

Other options:

```bash
# Docker
docker build -t geoqoe-pulse .
docker run --rm -p 3000:3000 --env-file .env geoqoe-pulse

# Python - static only, see the warning above
python3 -m http.server 3000
```

> Opening `dashboard/build/index.html` directly by double-clicking it will
> **not** work — browsers block the page from loading its data files that
> way, hence the small local server above.

### 3.1 Recommended dashboard workflow

Use the dashboard in this order:

1. **REVEAL** — start with the real quarterly observed-status map and the
   executive overview. Identify cells already marked degraded or severe.
2. **ANTICIPATE** — open **AI Forecast** to rank currently-normal cells for
   emerging underperformance in the next quarter. Ranks are priorities, not
   probabilities.
3. **EXPLAIN** — use **Investigation Evidence** to inspect independent
   citywide-relative signals, SHAP contributors, geographic/activity context,
   and measurement support. A signal is evidence for investigation, not a
   diagnosed cause.
4. **PRIORITISE** — filter the investigation queue by High/Medium/Low,
   workflow status, or evidence category, then record the next investigation
   step.

The map's **Quarterly** lens is the validated default. **QoE**, **Demand**,
**Hotspots**, Time-Series Analytics, and hourly playback are explicitly
**DEMO / SCENARIO** views. Never combine an hourly scenario number with a
quarterly AI rank without reading both labels.

For a complete plain-language explanation of every tab, button, filter, map
control, and selected-cell field, see [`dashboard/build/USER_GUIDE.md`](dashboard/build/USER_GUIDE.md).

### Alternative: the Streamlit version

A second, Python-based view of the same underlying pipeline is available
at the repo root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This is a simpler, more analytical view (filters, tables, charts) built
directly from the pipeline output — useful for judges who want to see the
raw pipeline data outside of the map-first dashboard.

### What to click on first

- Open **Reveal** and read the three primary KPIs: observed
  underperformance, High emerging risk and Elevated emerging risk.
- Select the Top Priority cell or click any quarterly-map hexagon to open
  **Cell Intelligence**.
- In **Anticipate**, filter the ranked queue by High, Elevated or Lower,
  then use **Reveal Actual** in Time Machine to compare forecasts with the
  held-out next-quarter result.
- Use **Explain** to review SHAP contributors and scenario evidence without
  treating them as diagnosed causes.
- Use **Prioritise** for the actionable shortlist and workflow status.
- Open **Scenario Lab** only when demonstrating the separate 168-hour
  hourly scenario. The amber badge is the reminder that it is not the
  quarterly validated forecast.
- Check **Data & Trust** for model metadata, provenance, definitions and
  limitations before quoting results.

---

## 4. The datasets, in detail

Everything below is generated by the pipeline in `src/` and
`run_pipeline.py`, orchestrated per the settings in `config.py`, and
written to `data/outputs/`. The identical set is duplicated at
`Trained AI Model/GeoQoE_AI/data/pipeline_v3_2/` as the frozen snapshot the
AI models were actually trained against.

> **Datasets download.** The raw and processed datasets are mirrored here:
> <https://drive.google.com/drive/folders/1EG-etCo-w5oxZ3QlJQFvC-xa1H-_myGX?usp=drive_link>
> Raw source files are not committed to this repository (see `.gitignore`);
> every source is re-fetchable from its original publisher via the URLs
> recorded in `source_manifest.json`.

### 4.1 Raw open-data sources ingested by the pipeline

| Source | What it provides | Real / not |
|---|---|---|
| **OpenCellID** (`502.csv.gz`) | Cell tower locations | ✅ Real |
| **Google Open Buildings** (`31d_buildings.csv.gz`) | Building footprints / density | ✅ Real |
| **OpenStreetMap** (`malaysia-singapore-brunei-*.osm.pbf`) | Roads, POIs, transit stops | ✅ Real |
| **WorldPop** (`mys_ppp_2020_UNadj.tif`) | Gridded population raster | ✅ Real |
| **OpenDOSM** (`population_malaysia.csv`) | Official Malaysian population statistics | ✅ Real |
| **Ookla Open Data** (`2024-10-01_performance_mobile_tiles.parquet`) | Real quarterly average mobile download/upload speed, latency | ✅ Real |
| Transit ridership (`ridership_headline.csv`) | Headline ridership figures used as a transit-demand proxy | ✅ Real |

Every source file's presence, size, and content hash is recorded in
`data/outputs/metadata.json` for provenance — nothing is silently
substituted. The pipeline's synthetic-fallback tracker confirms **0%** of
tower, building, OSM, or population features fall back to synthetic data;
the only intentionally simulated component is the hourly QoE shape (see
§7).

### 4.2 Processed / output tables (`data/outputs/`)

| File | Rows | What it is |
|---|---|---|
| `cell_static_features.parquet` | 4,918 | One row per H3 cell: tower count, building density, POI/road/transit density, population, demand-pressure features. |
| `cell_static_qoe_baseline.parquet` | 4,918 (2,950 real-labelled) | Static per-cell QoE baseline used to train/evaluate the **static AI model** — real Ookla download/upload/RTT where available. |
| `cell_time_dataset.parquet` | 826,224 (4,918 cells × 168 hours) | The full hourly panel driving the dashboard's time slider — dl/ul/rtt/jitter/loss per cell per hour (hourly *shape* is simulated; see §7). |
| `cell_quarter_dataset.parquet` ⭐ | 41,448 (3,566 cells × 14 quarters) | **The real temporal dataset** — 2023Q1–2026Q2, every value an independent Ookla measurement, 3,959,197 real speed tests. Carries the **primary modelling target `target_is_below_area_norm_next_q`**. Use this for any forecasting claim. |
| `qoe_targets.parquet` | 826,224 | Hourly degradation labels derived from the QoE thresholds in `config.py`. `target_is_degraded_next_1h` is a **demonstration label only** — the hourly panel is rank-one, so it is reconstructable from (cell, hour-of-week) and any accuracy on it measures memorisation. See `config.PANEL_STRUCTURE_DISCLOSURE`. |
| `feature_dictionary.xlsx` | — | Human-readable column-by-column data dictionary for every feature above, including which columns are flagged as leakage-risk (see below). |
| `metadata.json` | — | Pipeline version, run timestamp, git commit, per-source-file provenance hashes, row counts, and the synthetic-fallback summary. |

### 4.3 Grid & boundary (`data/boundary/`, `data/processed/`)

- `study_boundary.geojson` — real GADM level-2 Selangor districts +
  Putrajaya, merged with a real WP Kuala Lumpur polygon (Klang Valley core:
  Gombak, Hulu Langat, Klang, Kuala Langat, Petaling, Sepang, Putrajaya,
  Kuala Lumpur). Deliberately excludes rural/exurban Selangor.
- `grid_geometry.parquet` / `.geojson` — the 4,918 H3 resolution-8 hexagon
  polygons over that boundary.

### 4.4 Data-quality safeguards worth knowing about

- **Leakage audit built into `config.py`.** Columns that are mechanically
  circular with the simulated QoE generator (e.g. `demand_pressure_index`,
  `qoe_level_is_ookla_anchored`, `qoe_risk_label`) are explicitly flagged in
  `HIGH_LEAKAGE_RISK_FEATURES` and hard-excluded from the default feature
  list used to train both AI models — not just documented, actually removed.
- **Single source of truth for thresholds.** All degradation labels
  (`target_is_degraded`, `target_degradation_class`) are derived from the
  same two QoE-threshold tiers, so they can never disagree with each other.
- **Spatial holdout key.** Every output row carries `h3_parent_res6`, a
  coarse parent-cell grouping key used to hold out whole geographic regions
  (not just individual cells) during model validation — this is what makes
  the **legacy static spatial estimator's** reported accuracy a genuine
  out-of-region estimate rather than a same-neighbourhood leak (§5.1).
  **The finals quarterly model does not use this key.** Its validation is
  strictly temporal (train 2023Q1–2025Q4, sealed test 2026Q1 → 2026Q2), so
  its reported figures are a forward-in-time estimate, not a
  region-transfer estimate. Adding a spatial holdout is planned work, not a
  claim this project already makes.
- **Reproducibility.** One global random seed (`42`), applied once at
  pipeline start.

---

## 5. The AI models, in detail

Both models live in `Trained AI Model/GeoQoE_AI/`, with the deployable,
judge-facing snapshot at
`Trained AI Model/GeoQoE_AI/competition_package/` (also mirrored into
`dashboard/build/ai_model/` for the live dashboard). Full training code for
both is in `Trained AI Model/GeoQoE_AI/code/`.

### 5.0 Active dashboard model versus legacy package

The dashboard's validated AI view is the **frozen quarterly finals model**:

- predictor quarter: **2026Q1**;
- forecast quarter: **2026Q2**;
- input/output: `dashboard/build/data/finals_quarterly.geojson`;
- metadata and risk policy: `dashboard/build/data/finals_metadata.json`;
- observed current-quarter layer: `dashboard/build/data/quarterly_observed_2026Q1.geojson`.

It ranks currently-normal cells for emerging next-quarter underperformance and
keeps observed underperformance separate from forecast eligibility. The
dashboard loads model version, metrics, thresholds, interpretation notes, and
the `score_is_probability` flag from metadata rather than hard-coding them in
JavaScript.

The files under `dashboard/build/ai_model/dashboard/` and the old
`data/onset_model.json`/`data/model_card.json` path are **legacy next-hour
artifacts**. They remain useful for provenance and comparison, but they are
not the source of the current quarterly forecast screen.

### 5.1 Static spatial QoE estimator

- **Task:** estimate a quarterly download-speed baseline for the **1,968**
  cells that have no real Ookla label, trained against the **2,950** cells
  that do.
- **Model:** 5-fold confidence-weighted XGBoost ensemble
  (`models/static_xgboost/confidence_weighted_fold_1..5.json`), validated
  with the spatial holdout key from §4.4 so no fold trains and tests on
  neighbouring cells.
- **Validated performance (spatial out-of-fold):** MAE ≈ **82.30 Mbps**,
  RMSE ≈ **116.18 Mbps**, R² ≈ **0.021**. This is honestly reported as a
  *rough spatial baseline*, not a precise measurement — Ookla-labelled and
  unlabelled cells have systematically different GIS coverage patterns,
  which caps how far this can go.
- **Every prediction carries its own reliability tag:** `observed_ookla`
  vs. `ai_estimated`, ensemble disagreement (`ensemble_std_mbps`), and
  feature-space support (`within_reference` / `limited_reference` /
  `outside_reference`) — so the dashboard never presents an estimate as if
  it were a measurement.

### 5.2 Next-hour degradation-onset ranker

- **Task:** among currently-normal, Ookla-anchored cells, rank which ones
  are at highest relative risk of degrading in the next hour.
- **Model:** `full_causal_xgboost` (41 features, frozen best iteration 18,
  frozen validation decision threshold 0.4477), selected by highest
  validation PR-AUC before a single final-test evaluation.
- > ⚠️ **Read the metrics below as memorisation, not skill.** They were
  > computed on the hourly panel, which `metadata.json →
  > panel_structure_disclosure` proves is rank-one: its target is
  > analytically reconstructable from (cell, hour-of-week) with a maximum
  > absolute error of 0.0. A ROC-AUC of 0.996 on a lookup table measures
  > recall of that table, not predictive ability. `HANDOFF.md` §8 lists
  > "any hourly forecast accuracy" under **Not defensible**. These numbers
  > are retained only to document what was superseded — **the project's
  > forecast claim is the quarterly model in §5.0.**
- **Final-test performance (held-out week):** PR-AUC **0.161**, ROC-AUC
  **0.996**, precision **0.125**, recall **0.990**, F1 **0.223** — the
  model caught 103 of 104 simulated test onsets, at the cost of 718 false
  positives. This precision/recall trade-off is surfaced verbatim in the
  dashboard's Model Info panel rather than hidden behind a single accuracy
  number.
- **Output is an uncalibrated, class-weighted relative score** — the
  dashboard explicitly labels it "NOT a %" and uses it for ranking
  (Top-20/50/100/1%/5%) and priority tiers, never as a stated probability.
- **Baselines it beat:** two simpler comparison models
  (`models/onset_xgboost/time_only_xgboost.json`,
  `qoe_history_xgboost.json`) and two logistic baselines
  (`models/onset_logistic/`) were trained alongside it for comparison —
  all included for judges who want to see the ablation.

### 5.3 Where the frozen deliverables live

```text
Trained AI Model/GeoQoE_AI/competition_package/
├── README.txt
├── package_manifest.json
├── models/geoqoe_onset_full_causal_xgboost.json
├── metadata/
│   ├── geoqoe_model_card.json        ← full metrics + limitations, machine-readable
│   └── geoqoe_onset_feature_registry.json
└── dashboard/
    ├── latest_hour_onset_risk.parquet   ← primary per-cell AI output (all 4,918 cells)
    ├── latest_hour_onset_risk.csv       ← same, CSV
    └── latest_hour_ranked_eligible_cells.csv  ← eligible cells only, ranked by risk
```

Latest packaged snapshot: timestamp `2026-06-30 23:00:00`, 4,918 total
cells, 2,843 eligible onset-scored cells, 107 currently-degraded anchored
cells, 44 alerts at the frozen threshold.

Full column-by-column reference (what every field in
`latest_hour_onset_risk.parquet` means, and the recommended map-coloring
logic built from it) is documented in
`Trained AI Model/GeoQoE_AI/README.md`, and the full narrative writeup of
how both models were built, validated, and audited for leakage is in
`Trained AI Model/GeoQoE_AI/docs/GeoQoE_AI_Model_Training_Explained.docx`.

---

## 6. The dashboard, in detail

`dashboard/build/` is an HTML/CSS/JS app (Leaflet for the map, Plotly for
charts, Bootstrap for layout) that reads pre-built static files — no database,
and no backend for anything the forecast depends on.

The one server-side piece is **local context retrieval** (below): `server.js`
exposes two small JSON routes so API credentials stay off the browser. With no
credentials configured the dashboard still runs; it serves whatever context is
already cached on disk and reports nothing for the rest.

| File | What it is | Size |
|---|---|---|
| `data/h3_grid.geojson` | Real hex polygons + static attributes | ~5 MB |
| `data/panel.json` | Full hourly panel (4,918 cells × 168 hours), columnar | ~64 MB |
| `data/city_timeseries.json` | Pre-aggregated city-wide stats per hour | ~24 KB |
| `data/summary_stats.json` | Provenance, KPI constants, disclaimer text | ~4 KB |
| `data/finals_quarterly.geojson` | Finals per-cell 2026Q1 inputs, 2026Q2 risk bands/ranks, quarterly measurements and explanation fields | ~5.6 MB |
| `data/quarterly_observed_2026Q1.geojson` | Separate real 2026Q1 observed-status layer | ~3.7 MB |
| `data/finals_metadata.json` | Model version, evaluation metrics, risk thresholds, Time Machine definitions and interpretation notes | ~3 KB |
| `data/onset_model.json`, `data/model_card.json` | Legacy next-hour/static model integration retained for provenance and Demo / Scenario compatibility | ~1.3 MB / ~5 KB |

These are generated from the raw pipeline/AI outputs by
`build_dashboard_data.py` and `build_onset_data.py` (both already run —
re-run only if you regenerate the pipeline yourself, see §9). A
from-scratch pure-Python Parquet reader
(`dashboard/build/tools/pyparquet_min.py`) was written so this conversion
needed no extra dependencies; it's validated against known pipeline facts
in `test_data_pipeline.py` (e.g. it decodes `ookla_dl_mbps` non-null rate
to exactly 59.98%, matching `feature_dictionary.xlsx`).

**What the dashboard adds on top of raw data:**

- **Reveal:** an executive summary of current 2026Q1 observed issues,
  future 2026Q2 AI risk, coverage metadata and the top-priority location.
- **Compare and Hotspots:** compact hourly scenario comparisons and spatial
  statistics, explicitly separated from quarterly model outputs.
- **AI Forecast:** data-driven High/Elevated/Lower filters, a ranked
  emerging-risk queue, methodology disclosure and Time Machine evaluation.
- **Explain:** selected-cell quarterly measurements and SHAP contributors,
  plus independent top-quartile scenario investigation signals. Scenario
  measurement support is shown separately from possible-driver signals.
- **Prioritise:** a future-risk shortlist by default, with observed issues
  available as a secondary view. A transparent prototype triage layer recommends
  Investigate now, Verify first, or Monitor without changing AI ranks. Users can
  save a browser-local investigation case, record human verification outcomes,
  and hand off a copy, JSON file, or print/PDF evidence brief.
- **Scenario Lab and Time-Series Analytics:** the 168-hour demonstration
  timeline, profile/lens controls and detailed hourly trend exploration.
- **Data & Trust:** provenance, REAL / AI MODEL / DEMO / SCENARIO / HEURISTIC
  labels, limitations, model metadata and collapsed technical methodology.
- **Cell Intelligence:** selecting a hexagon synchronises the map and side
  panel, showing observed status, emerging-risk band/rank, real quarterly
  measurements, contributors, measurement support and investigation priority.

The Explain-to-Reveal handoff preserves the selected H3 cell: **Review quarterly
measurements** opens that cell's Reveal detail, not the general executive overview.
Saved investigation cases and outcomes are local operational records; they are not
written into the datasets and do not automatically retrain the quarterly model.

Full technical detail on every dashboard feature (including a "what-if"
capacity simulator and the AI-vs-traditional hotspot comparison
methodology) is in `dashboard/build/README.md`.

---

## 7. What's real, and what's a demo stand-in (read before quoting a number)

We'd rather over-explain this than have it come up as a surprise question.

| Data | Status |
|---|---|
| Cell tower locations, buildings, roads/POIs/transit, population | ✅ **Real.** Sourced from OpenCellID, Google Open Buildings, OpenStreetMap, WorldPop, and OpenDOSM, joined with real GIS operations. |
| Quarterly average download/upload speed & latency (2,950 of 4,918 cells, ~60%) | ✅ **Real.** From Ookla's Open Data program. |
| **14 quarters of real speed data** (`cell_quarter_dataset.parquet`, 2023Q1–2026Q2) | ✅ **Real on both sides.** 3,959,197 independent Ookla speed tests across 3,566 cells. 36,193 rows carry a real one-step-ahead target measured a quarter later — nothing simulated, downscaled or shared-shape. **This is the dataset for any forecasting claim.** |
| **Hour-by-hour** network quality for the 168-hour (1-week) panel shown in the dashboard's time slider | ⚠️ **Simulated.** No public source of real per-hour QoE exists for this area, so hourly values are generated by a documented diurnal-demand formula. Where a real Ookla number exists for a cell, the simulation is re-centered on it — but the hour-to-hour *shape* is always simulated. This is clearly labelled in the dashboard, never presented as measured. |
| Finals quarterly emerging-risk model | ✅ **Trained and evaluated on real quarterly measurements.** It ranks currently-normal cells using 2026Q1 data for 2026Q2 review. Its output is a relative priority ranking, not a probability or guaranteed failure prediction. |
| Legacy static and next-hour model artifacts | ⚠️ **Retained for provenance/demo compatibility.** They are not the dashboard's validated default quarterly forecast. |
| **Local context** (events/holidays under a prioritised cell) | 🔎 **Really retrieved, machine-extracted, not yet human-verified.** Each item is shown only if its source URL came from a page the web search actually returned; unsourced items are discarded and counted. That guarantees the *link is real*, not that the page says what the summary claims — open the source before quoting it. Retrieved after the forecast; it plays no part in the ranking. |
| SHAP contributors and investigation signals | 🔧 **Explanatory / heuristic evidence.** SHAP explains the model's ranking behaviour; percentile-based scenario signals support investigation. Neither establishes engineering root cause. |

**Bottom line for judges asking "is this real data?":** the quarterly
observations and finals next-quarter modelling dataset use real Ookla
measurements. Towers, buildings, population and geographic context also
come from real open sources. The animated hour-by-hour view is a separate,
honestly-labelled simulation used to demonstrate how operational monitoring
could work with a live feed.

---

## 8. Full repository map

```
geoqoe_pulse/
├── README.md                  ← you are here
├── app.py                     ← Streamlit dashboard (alternative view)
├── config.py                  ← pipeline settings (CRS, thresholds, seed, leakage audit)
├── run_pipeline.py            ← runs the full data pipeline, 10 stages
├── data_loader.py             ← loads pipeline output for app.py
├── requirements.txt
│
├── data/                      ← pipeline inputs & outputs (§4)
│   ├── boundary/                study area definition (real GADM + KL polygon)
│   ├── processed/                H3 grid geometry
│   └── outputs/                  final datasets used by the dashboard & models
│
├── src/                       ← pipeline source code
│   ├── data_preprocessing/       geometry, joins, feature engineering, QoE targets
│   └── modelling/                 baseline training scripts, reports, plots
│
├── scripts/
│   └── trim_boundary.py          trims raw Selangor GADM extract to the study area
│
├── Trained AI Model/          ← the two trained AI models + training code (§5)
│   └── GeoQoE_AI/
│       ├── README.md              full AI/dashboard handoff doc — column reference, tooltips
│       ├── check_setup.py
│       ├── requirements.txt
│       ├── code/                  14 training/evaluation scripts (static + onset)
│       ├── models/                saved model files (static ensemble, onset xgboost/logistic)
│       ├── outputs/                metrics, audits, feature importance, OOF predictions
│       ├── competition_package/   ★ frozen model card + latest predictions ★
│       ├── data/pipeline_v3_2/     frozen copy of the dataset the models were trained on
│       └── docs/                  training write-up (.docx) + team handoff checklist
│
└── dashboard/build/           ← ★ the prototype — open this ★ (§6)
    ├── server.js              zero-dependency server + 2 JSON API routes
    ├── context_brief.js       local context retrieval (search + extract + ground)
    ├── regulator_actions.json closed catalogue of ITU-referenced actions
    ├── prewarm_context.js     cache the top N cells before a demo
    ├── .env.example           provider/API-key configuration template
    └── data/context/          cached briefs, one JSON per H3 cell
    ├── index.html               entry point
    ├── dashboard.js / .css      app logic & styling
    ├── server.js, Dockerfile    ways to serve it locally
    ├── data/                     pre-built JSON/GeoJSON the dashboard reads
    ├── ai_model/                 frozen AI model outputs used by the dashboard
    ├── tools/                    pure-Python parquet reader + Klang Valley gazetteer
    ├── test_data_pipeline.py    validates the data conversion against known pipeline facts
    ├── build_dashboard_data.py  regenerates data/ from a fresh pipeline run
    └── build_onset_data.py      regenerates the AI layer from a fresh model run
```

---

## 9. Reproducing the pipeline / retraining the models

Not required for judging (everything is pre-built), but included for
completeness / technical due diligence:

```bash
# 1. Regenerate the full data pipeline from raw sources
pip install -r requirements.txt
python3 run_pipeline.py          # 10 stages: boundary → grid → joins →
                                  # features → QoE targets → validation → export

# 2. Retrain the AI models against the pipeline output
cd "Trained AI Model/GeoQoE_AI"
pip install -r requirements.txt
python3 code/static_train_xgboost.py
python3 code/onset_train_xgboost.py
python3 code/finalize_competition_models.py   # freezes competition_package/

# 3. Rebuild the dashboard's static data files
cd ../../dashboard
python3 build_dashboard_data.py
python3 build_onset_data.py
```

`Trained AI Model/GeoQoE_AI/docs/TEAM_HANDOFF_FILE_CHECKLIST.md` lists the
exact minimum file set for each of these three stages (dashboard handoff,
documentation, and full reproducibility archive) if you want to verify
nothing is missing.

---

## 10. Known limitations

- No real per-hour network measurements exist for this area — the hourly
  panel is a labelled simulation, not observed data (see §7).
- No real network capacity/traffic data exists — "demand pressure" is an
  activity proxy, not confirmed congestion.
- The custom 0–100 QoE index is our own composite, not an ITU-T
  G.107/P.1203 standard.
- The static AI model's spatial out-of-fold R² is low (≈0.02) — treat it
  as a rough estimator/blind-spot filler, not a precise Mbps prediction.
- The finals model output is a relative ranking score, not a calibrated
  probability. High/Elevated bands are prioritisation policies, not
  engineering failure thresholds or guarantees of future degradation.
- Time Machine results are retrospective held-out evaluation outcomes;
  they demonstrate validation behaviour but do not guarantee future results.
- Investigation hypotheses and scenario signals are decision-support cues,
  not diagnosed root causes or engineering failure thresholds.
- `dashboard/build/data/panel.json` is ~64 MB and loads once on page
  load — fine for a local judged demo, a few seconds on a slow connection.
- The Docker path is written to standard conventions but was untested in
  the build environment (no Docker/network access there) — worth a quick
  local test before relying on it during judging.
- Retrieved local context is machine-extracted and **not human-verified**.
  The code guarantees every displayed item cites a page the search actually
  returned; it cannot guarantee that page says what the summary claims. Open
  the source links before quoting an event. Near-duplicate events reported by
  two different sources are not merged.
- The ITU references attached to suggested actions are transcribed into
  `dashboard/build/regulator_actions.json` for convenience and are **not**
  validated by the code. Confirm them against the ITU-T catalogue before
  presenting them as authority.
- Local context retrieval needs an API key and a network at retrieval time.
  Pre-warmed cells are served from committed JSON and work offline with no
  key; uncached cells return "no corroborated context" instead of an error.

Full technical detail on every pipeline stage, feature, and design decision
lives in `src/modelling/` reports, `Trained AI Model/GeoQoE_AI/README.md`,
and `dashboard/build/README.md` for anyone who wants to go deeper than
this overview.

## Finals Quarterly AI Forecast

The finals GeoQoE Pulse prototype uses a real quarterly XGBoost early-warning model to rank currently-normal H3 cells by their risk of falling into the bottom 25% of Klang Valley download performance in the following quarter.

Final model:

`models/finals_quarterly/final_xgb_full_v1.0.0.json`

AI handover:

`handoff/finals_ai/README.md`

Technical model card:

`handoff/finals_ai/MODEL_CARD.md`

Frontend integration guide:

`handoff/finals_ai/FRONTEND_INTEGRATION.md`
