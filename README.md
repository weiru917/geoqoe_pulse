# GeoQoE Pulse

**A mobile network quality decision-support prototype for the Klang Valley (Kuala Lumpur & surrounding urban Selangor).**

GeoQoE Pulse takes real, open geospatial and network data, combines it into
one map, and helps a telecom operator or regulator answer two questions:

1. **Where is mobile internet quality currently bad, and why?**
2. **Where is it likely to get bad next, and what should we do about it?**

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
| **② AI models** | Two trained machine-learning models sit on top of that data: one **estimates** network quality in cells with no direct measurement, and one **ranks** cells by risk of degrading in the next hour. | `Trained AI Model/GeoQoE_AI/` |
| **③ Interactive dashboard** | A map-based web app where a judge can explore quality, hotspots, AI predictions, and recommended interventions, hour by hour. | `dashboard/build/` |

**The dashboard (layer ③) is the prototype itself — that's what to open
first (see §3).** Layers ① and ② are the data/AI engine behind it, included
in full — raw-to-model code, trained model files, and every evaluation
report — so judges can inspect the entire pipeline, not just the front end.

---

## 2. How it works, in plain terms

1. **Build the map grid.** The Klang Valley is divided into **4,918**
   hexagonal cells (H3 grid, resolution 8, ~0.74 km² each, ~461 m edge —
   roughly a few city blocks).
2. **Attach real-world context to every cell.** For each cell: number of
   cell towers, building density, road/POI/transit density, resident and
   working population, and — where available — a **real** quarterly
   average internet speed from Ookla's open speed-test dataset (**2,950 of
   4,918 cells, ~60%**, carry a real Ookla label).
3. **Score network quality.** A custom 0–100 Quality-of-Experience (QoE)
   index is calculated from download/upload speed, latency, jitter, and
   packet loss, and cells are flagged as "degraded" against fixed
   thresholds (download < 10 Mbps or latency > 100 ms).
4. **Find hotspots.** Real spatial statistics (Getis-Ord Gi\*, Local
   Moran's I) identify statistically significant clusters of poor quality,
   not just individually bad cells.
5. **Predict risk.** Two trained XGBoost/logistic models — one estimating
   quality for cells with no direct measurement, one ranking cells by
   next-hour degradation risk — sit on top of the real data (full detail
   in §5).
6. **Show it all on one dashboard.** A judge can scrub through time,
   change what the map is colored by, compare cells, simulate "what if we
   added X% more capacity," and see a ranked, prioritized list of
   recommended interventions.

---

## 3. How to open it (for judging)

The dashboard is a self-contained web app — **no Python environment or
installation required.** All the data it needs has already been generated
and sits in `dashboard/build/data/` and `dashboard/build/ai_model/`.

```bash
cd dashboard/build
python3 -m http.server 3000
```

Then open **http://localhost:3000** in a browser.

No Python on hand? Two alternatives that need nothing else installed:

```bash
# Node.js (a zero-dependency server is already included)
node server.js 3000

# Docker
docker build -t geoqoe-pulse .
docker run --rm -p 3000:3000 geoqoe-pulse
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

- Start with the **"How to use this dashboard"** panel at the top — it's
  written for a first-time viewer.
- Use the **time slider** to scrub through a full simulated week (168
  hours).
- Try the **🤖 Trained AI Model** panel for the model's live risk rankings
  and its Model Info modal (validated metrics, plain-language limitations).
- Try **Color by → 🤖 AI Onset Risk** and **🤖 AI Static DL Estimate** on
  the map, and open a cell's popup — it shows the AI section with a
  "why is this cell flagged?" explanation.
- Try the **Overview | Full Dashboard** toggle above the KPI row for a
  presentation-ready six-card summary.
- Check the **Data Quality & Trust** panel — every panel is badged
  REAL / SIMULATED / HEURISTIC so it's always clear what you're looking at.

---

## 4. The datasets, in detail

Everything below is generated by the pipeline in `src/` and
`run_pipeline.py`, orchestrated per the settings in `config.py`, and
written to `data/outputs/`. The identical set is duplicated at
`Trained AI Model/GeoQoE_AI/data/pipeline_v3_2/` as the frozen snapshot the
AI models were actually trained against.

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
  the AI models' reported accuracy a genuine out-of-region estimate rather
  than a same-neighbourhood leak.
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

`dashboard/build/` is a standalone HTML/CSS/JS app (Leaflet for the map,
Plotly for charts, Bootstrap for layout) that reads only pre-built static
files — no backend, no database.

| File | What it is | Size |
|---|---|---|
| `data/h3_grid.geojson` | Real hex polygons + static attributes | ~5 MB |
| `data/panel.json` | Full hourly panel (4,918 cells × 168 hours), columnar | ~64 MB |
| `data/city_timeseries.json` | Pre-aggregated city-wide stats per hour | ~24 KB |
| `data/summary_stats.json` | Provenance, KPI constants, disclaimer text | ~4 KB |
| `data/onset_model.json` (built from `ai_model/dashboard/`) | Per-cell AI static estimate + onset risk, one fixed snapshot | ~1.3 MB |
| `data/model_card.json` | Frozen AI model's validated metrics + limitations | ~8 KB |

These are generated from the raw pipeline/AI outputs by
`build_dashboard_data.py` and `build_onset_data.py` (both already run —
re-run only if you regenerate the pipeline yourself, see §9). A
from-scratch pure-Python Parquet reader
(`dashboard/build/tools/pyparquet_min.py`) was written so this conversion
needed no extra dependencies; it's validated against known pipeline facts
in `test_data_pipeline.py` (e.g. it decodes `ookla_dl_mbps` non-null rate
to exactly 59.98%, matching `feature_dictionary.xlsx`).

**What the dashboard adds on top of raw data:**

- Time slider over the full 168-hour simulated week, with multiple
  "Color by" map modes (current QoE, degradation, AI onset risk, AI static
  estimate, AI-vs-traditional hotspot comparison).
- A ranked, prioritized **Interventions** panel built from the same
  candidate list used by the Executive Overview cards, so the two views
  can never disagree with each other.
- Per-cell popups showing observed vs. AI-estimated speed (with
  reliability tag), onset status/tier/score, and a rule-based "likely
  trigger" explanation built from real z-scored RTT/jitter/loss/demand at
  the AI snapshot hour.
- An honest time-sync banner: since the onset model was only ever scored
  for one real hour, the AI panel tells you whether the time slider is
  currently on that scored hour or not, rather than silently pretending
  every hour has a live forecast.
- A **Data Quality & Trust** panel that badges every number on screen as
  REAL / SIMULATED / HEURISTIC.

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
| The two AI models (static quality estimator + next-hour onset risk ranker) | ✅ **Really trained, on real held-out data.** The static model is trained/validated against the real Ookla labels. The onset-risk model is trained on the simulated hourly panel (see above) and evaluated on a real held-out split of it — its own validated precision/recall numbers are shown in the dashboard, including the honest trade-off (high recall, low precision, stated plainly rather than hidden). |
| Root-cause labels, cost estimates, operator contacts on intervention cards | 🔧 **Heuristic / illustrative placeholders**, computed transparently from real features — not pipeline-measured facts. Clearly labelled as such in the UI. |

**Bottom line for judges asking "is this real data?":** the static spatial
layer (towers, buildings, population, real Ookla speeds) is 100% real and
is what the AI static-quality model is trained and validated on. The
animated hour-by-hour view is an honestly-labelled simulation used to
demonstrate what the dashboard and forecasting pipeline would look like
running on live data.

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
- The onset model's risk score is uncalibrated and class-weighted by
  design — high recall, low precision at the frozen threshold (stated
  plainly in the dashboard rather than hidden).
- Root cause, cost, and operator-contact fields on intervention cards are
  editable illustrative placeholders, not measured pipeline outputs.
- `dashboard/build/data/panel.json` is ~64 MB and loads once on page
  load — fine for a local judged demo, a few seconds on a slow connection.
- The Docker path is written to standard conventions but was untested in
  the build environment (no Docker/network access there) — worth a quick
  local test before relying on it during judging.

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
