# GeoQoE Pulse — Dashboard

> **Active AI integration:** the dashboard now uses the frozen quarterly finals
> package (`2026Q1 → 2026Q2`) plus a separate real 2026Q1 observed-status layer.
> The older next-hour files under `ai_model/`, `data/onset_model.json`,
> `data/model_card.json`, and `build_onset_data.py` are retained for provenance
> only and are not loaded by the current dashboard.

The validated default workflow is **REVEAL → ANTICIPATE → EXPLAIN →
PRIORITISE**: reveal observed 2026Q1 underperformance, anticipate emerging
2026Q2 risk among currently-normal cells, explain the assessment with SHAP
contributors plus context and measurement support, then prioritise where to
investigate. Contributors and dashboard heuristics are investigation evidence,
not diagnosed root causes. VIIRS is a human-activity proxy, not real network
demand. Hourly M-Lab / `cell_time_dataset` views are separate and labelled
**Demo / Scenario**.

For presentations, the quarterly map can be switched between **Observed Q1**,
**Forecast Q2**, and **Combined**. These are display filters over the same
validated finals package; they do not recompute model scores or change H3 data.
The guided walkthrough selects these views in workflow order. The separate
hourly export is shown only inside Scenario Lab and is labelled **Export
scenario report**.

A standalone HTML/CSS/JS dashboard (Leaflet + Plotly, Bootstrap for layout)
built directly on the **real** geoqoe_pulse pipeline output — real Ookla,
OpenCellID, Google Open Buildings, OpenStreetMap, WorldPop, OpenDOSM, and
transit ridership data over the Klang Valley, joined at H3 resolution 8 —
now combined with the **trained AI model package** (`ai_model/`): a real,
already-trained static spatial QoE estimator and a next-hour degradation-
onset XGBoost model, both evaluated on real held-out data. See section 1b.

## 0. Placement

This folder sits at `dashboard/build/` inside the
`geoqoe_pulse` repo; its build script resolves inputs from the repository root — it's
already been generated once against your real pipeline output, so:

- **Just want to view it?** `data/*.json` here are already built from your
  real data (including the AI model outputs — see section 1b). Skip
  straight to section 2 ("Run it") — no repo needed.
- **Want to regenerate after re-running the pipeline?** Keep it at
  `geoqoe_pulse/dashboard/build/`, then follow section 1.

## 1. Generate the data (only needed if you've re-run the pipeline)

```bash
cd geoqoe_pulse/dashboard/build
python3 build_dashboard_data.py
```

This reads `data/outputs/cell_static_features.parquet`,
`data/outputs/cell_time_dataset.parquet`, `data/processed/grid_geometry.geojson`
and `data/outputs/metadata.json` from the repo root, and writes the active files into
`dashboard/build/data/`:

| File | What it is | Size |
|---|---|---|
| `h3_grid.geojson` | Real hex polygons + static attributes (towers, population, POIs, etc.) | ~5 MB |
| `panel.json` | Full hourly panel (4,918 cells x 168 hours), columnar | ~54 MB |
| `city_timeseries.json` | Pre-aggregated city-wide stats per hour, for the trend charts | ~20 KB |
| `summary_stats.json` | Provenance, KPI constants, disclaimer text | <1 KB |
| `quarterly_observed_2026Q1.geojson` | Real 2026Q1 observed status and network measurements | ~4 MB |
| `finals_quarterly.geojson` | Frozen finals AI forecast and held-out outcome fields | ~6 MB |
| `finals_metadata.json` | Model version, risk policy, metrics and explanation metadata | <10 KB |

**Why a conversion script at all, instead of reading parquet directly in the
browser:** this sandbox had no network access and neither `pyarrow` nor
`fastparquet` installed, so `../tools/pyparquet_min.py` — a from-scratch
pure-Python Parquet reader (Thrift compact-protocol metadata parsing,
Snappy/GZIP/ZSTD page decompression, RLE/bit-packed dictionary decode) — was
written to read your actual files rather than falling back to synthetic data.
It's validated in `test_data_pipeline.py` (see section 5) against facts the
pipeline already documents about itself: e.g. `ookla_dl_mbps` in
`cell_static_qoe_baseline.parquet` decodes to exactly 59.98% non-null, matching
`feature_dictionary.xlsx` to two decimal places, and zero un-anchored cells
ever show a degraded target — exactly the correlation `config.py` warns about.
If you have `pyarrow` available in your own environment, feel free to swap
`pyparquet_min.read_parquet` for `pandas.read_parquet` in
`build_dashboard_data.py` — the rest of the pipeline is unaffected.

## 1b. Legacy next-hour AI layer (retained, inactive)

`ai_model/` is the unzipped **frozen competition package** from the
`GeoQoE_AI` project — a real, already-trained static spatial QoE ensemble
and a real, already-trained next-hour degradation-onset XGBoost model, both
evaluated once on real held-out data. Nothing in this layer is invented by
the dashboard: every number comes from `ai_model/metadata/geoqoe_model_card.json`
or `ai_model/dashboard/latest_hour_onset_risk.csv`.

```bash
cd geoqoe_pulse/dashboard
python3 build_onset_data.py
```

This reads `ai_model/dashboard/latest_hour_onset_risk.csv` and
`ai_model/metadata/geoqoe_model_card.json`, reindexes every row to the exact
same 4,918-cell order as `panel.json` (so the AI layer can be looked up by
the same cell index with zero extra work in `dashboard.js`), and writes:

| File | What it is | Size |
|---|---|---|
| `data/onset_model.json` | Per-cell static AI estimate + next-hour onset risk, one fixed snapshot | ~1.3 MB |
| `data/model_card.json` | Frozen model's validated metrics (PR-AUC, ROC-AUC, precision/recall) and stated limitations | ~2 KB |

**This is a single fixed snapshot, not another animated hour of the
168-hour panel.** The onset model was only run once, against the pipeline's
own latest hour (`2026-06-30 23:00:00`, which happens to be the panel's
final timestamp) — see `dashboard_snapshot` in the model card. The
dashboard therefore treats the AI layer as independent of the time slider,
the same way the tower layer is: always visible when selected, regardless
of which hour the slider is on.

**What it adds to the dashboard:**

- Two new **Color by** map modes: 🤖 *AI Onset Risk* (colors cells by
  `onset_priority_tier` — current degradation → highest/high priority →
  elevated/routine watch → not scored) and 🤖 *AI Static DL Estimate*
  (colors cells by `display_dl_mbps`, with a dashed purple outline flagging
  AI-estimated cells vs. a solid outline for real observed Ookla cells).
- A **🤖 Trained AI Model** panel: ranked watchlists (currently-degraded,
  top-10 next-hour onset risk) plus a **Model Info** modal with the frozen
  model's validated validation/final-test PR-AUC, ROC-AUC, precision,
  recall, F1, and its stated limitations verbatim from the model card.
- Every cell's map popup gets an AI section: observed vs. AI-estimated DL
  speed (with reliability tag), and the onset status/tier/score for that
  cell — always labelled as a **relative ranking, not a calibrated
  probability**, per the model card's own caveat.
- If `data/onset_model.json` is missing or doesn't line up with the current
  `panel.json` (different pipeline run), the dashboard degrades gracefully:
  the AI panel hides itself and the two AI color-by options disable, rather
  than breaking the page.

## 1c. Operational trust upgrades (new)

Four changes made after an internal review, all sourcing only real,
already-computed numbers — nothing here is a new model or a new data file:

- **Honest AI time-sync, not a fake forecast.** The onset model has only
  ever been scored for one hour (`onset.snapshot_timestamp`). Rather than
  interpolate or repeat that score across the other 167 hours to *look*
  temporal, the AI panel now shows a banner comparing the time slider's
  hour to the model's actual scored hour: a green confirmation when they
  match, an amber warning with a "jump to that hour" link when they don't.
  The panel is relabelled **"AI Forecast — Latest Available Model Run"**
  everywhere, so nobody mistakes the frozen snapshot for a live per-hour
  forecast.
- **Per-cell "why is this cell flagged?" explanation.** For cells at
  elevated risk or currently degraded, the map popup now shows the
  city-wide z-score of that cell's real RTT/jitter/loss/Scenario Pressure at
  the AI snapshot hour (🔴 very high / 🟠 elevated / 🟢 normal), plus the
  hour-over-hour QoE change, plus a one-line possible-driver statement
  built from the same rule-based categorization used in
  Investigation Evidence / Investigation Priorities. This is a computed scenario cue,
  not a fabricated SHAP value.
- **Risk score vs. probability, made hard to miss.** Every onset score now
  carries an inline "— NOT a %" tag, and a new **Model behavior** badge on
  the AI panel states the frozen model's actual precision/recall trade-off
  in plain language (e.g. "High recall / high alert volume — final-test
  precision 13%, recall 99%"), computed directly from `model_card.json`.
- **Executive Overview mode.** An `Overview | Full Dashboard` toggle above
  the KPI row. Overview shows six presentation-ready cards — Network
  Health, Degraded Cells, AI Forecast, Top Problem Area, Primary Driver,
  Recommended Action — computed from the exact same filtered candidate
  list the Investigation Priorities panel uses, so the two views can't disagree.
  "Full Dashboard" (default) shows everything as before.

## 1d. Medium-priority upgrades (new)

Three more changes from the same review, all built from real numbers
already sitting in `ai_model/` or the pipeline output — nothing fabricated:

- **AI vs. traditional hotspot comparison.** A new map mode
  (`🔍 AI vs Traditional Hotspot`) plus a **Detection Comparison** panel
  compare the AI onset model against the existing Gi* spatial hotspot
  analysis (`hotspot_code` in {hot_90, hot_95, hot_99}), both evaluated at
  the AI's own snapshot hour so it's a fair comparison. It reports real
  counts — e.g. cells the AI flags that traditional analysis misses, and
  vice versa — with a couple of named examples, so you can see whether the
  onset model is finding genuinely different things or just agreeing with
  spatial statistics.
- **Forecast Validation view.** A new modal (button on the AI panel) shows
  the onset model's real confusion matrix — true/false positives and
  negatives — for both the validation and held-out final-test sets,
  straight from `geoqoe_model_card.json`, plus the model's own real
  alert-volume/precision/recall sweep at different cutoffs (top-20/50/100
  per hour, top-1%/5%). Note: there's no ground-truth outcome for the
  dashboard's *own* snapshot hour to grade against (it's the last hour of
  the 168-hour panel, so "what actually happened next" was never
  observed) — this view intentionally uses the model's real held-out
  evaluation instead of fabricating a snapshot-level matrix.
- **Richer investigation cards.** Each card now has an explicit **Evidence**
  line (QoE, Scenario Pressure, RTT, persistent-hotspot frequency, and — when
  the AI package is loaded — the cell's latest AI onset risk tier), a
  P1/P2/P3 priority label alongside the existing urgency pill, and a third
  tracking status ("Investigating") between Planned and Resolved.

## 2. Run it

Any of these three work — pick whichever is already on your machine:

```bash
# Option A - Python (simplest, no install)
python3 -m http.server 3000

# Option B - Node (included, zero dependencies)
node server.js 3000

# Option C - Docker (see Dockerfile; build once data/ exists)
docker build -t geoqoe-pulse .
docker run --rm -p 3000:3000 geoqoe-pulse
```

Then open **http://localhost:3000**. Opening `index.html` directly via
`file://` will NOT work — browsers block `fetch()` of local files under that
scheme, hence the static server.

## 3. What's real vs. simulated — read this before quoting a number

`target_qoe_index` (and everything derived from it — degradation class,
hotspot classes, Scenario Pressure) comes from **`cell_time_dataset.parquet`**,
which the pipeline's own `config.SIMULATION_DISCLAIMER` describes as *"a
simulated diurnal-demand scenario, not observed network measurements ... a
software-integration and dashboard-demo fixture."* 60% of cells have that
simulated hourly shape re-centred on a real quarterly Ookla average
(`qoe_level_is_ookla_anchored`); the other 40% are simulated-only. Every
degraded cell-hour in the panel comes from an anchored cell — zero exceptions,
confirmed by `test_data_pipeline.py`.

What IS 100% real and independent of that simulation: tower locations/counts
(OpenCellID), building footprints and built-up ratio (Google Open Buildings),
roads/POIs/transit stops (OpenStreetMap), population (WorldPop + OpenDOSM),
the H3 grid geometry, and the Gi*/Local Moran's I spatial statistics (computed
on the real geometry, even though the input `rtt_median` values they're
computed over are simulated). The dashboard's **Data Quality & Trust** panel
states this plainly, and relevant panel headers carry a
REAL / SIMULATED / HEURISTIC badge so it's never ambiguous which you're
looking at.

The **Investigation Evidence** breakdown and **Investigation Priorities** ranking's possible-driver
label are a Demo / Scenario heuristic computed client-side from static features (tower
distance, built-up ratio, latency versus Scenario Pressure, sample confidence) — the
pipeline itself exports no diagnosed-cause column. The **cost estimates** and
**operator contact** lines on investigation cards are explicitly labelled
editable placeholders (`ILLUSTRATIVE_COST_RANGES` / `OPERATOR_CONTACTS` at the
top of `dashboard.js`) — this dataset has no real per-cell operator
attribution or verified deployment cost data, so nothing here fabricates
numbers and presents them as measured.

## 4. Feature tour

- **Time slider + Play button** — scrub or animate through all 168 hours; the
  Demo / Scenario map, KPIs, hotspot lists, investigation evidence, and investigation queues all update live.
- **Trend range** (dual slider under the charts) — select a multi-hour window
  for the range summary (avg QoE, total degraded cell-hours, worst hour)
  independent of the single "current hour" used by the map. Hourly/Daily
  toggle re-aggregates the trend charts client-side.
- **Color by** — QoE, Scenario Pressure, Gi* hotspot class, Local Moran's I
  cluster, or historical next-hour risk.
- **Application-specific QoE profiles** (Balanced / Video Calling / Gaming /
  Streaming) — reweights the composite QoE formula for a specific use case.
  "Balanced" uses the pipeline's real `QOE_INDEX_WEIGHTS` from `config.py`
  and reproduces `target_qoe_index` exactly (verified in
  `test_data_pipeline.py` to within 0.5pt over a sampled sweep of the real
  panel). The other three profiles are **our own authored reweighting** of
  the same methodology — clearly labelled as such in the UI, not a validated
  per-application standard (the pipeline's own docstring says even the
  balanced formula isn't a validated ITU-T G.107 E-model). A cell can be
  green under Streaming and red under Gaming — that's the point, and it's
  computed from real dl/ul/rtt/jitter/loss values, not invented per profile.
- **🪄 What-if capacity simulator** — a slider that recomputes dl/rtt (and
  derived QoE/degradation) as if X% more or less capacity existed. Uses the
  *real* demand-elasticity coefficients from `data_downloader.py`'s
  `build_dynamic_panel()` (0.3/0.15 for anchored cells, 0.8/0.9 for
  unanchored), applied to `demand_pressure_index` as a proxy for the
  generator's per-cell `load_factor` — which isn't itself exported, so this
  is a **documented approximation**, not an exact inverse of the stochastic
  generator. Says so directly in the UI. Shows a real before/after: degraded
  cell count and avg QoE, recomputed against the pipeline's actual
  degradation thresholds.
- **📄 Export scenario report** — opens a print-ready hourly Demo / Scenario report (KPIs, top-5
  worst cells with real coordinates, QoE trend and root-cause chart images)
  in a new tab; use your browser's native Print → Save as PDF. Deliberately
  does *not* try to screenshot the Leaflet map (cross-origin tile canvases
  are frequently blocked from capture, which would silently produce a blank
  image) — a real data table is more reliable in print anyway.
- **Live rain radar** — real current rainfall via RainViewer's free public
  API (no key required), toggleable on the map. Explicitly **not** presented
  as an explanation for the historical/simulated QoE panel: that panel
  covers one fixed simulated week, this shows whatever is raining right now,
  and the two time windows don't line up — the UI disclaimer says this
  outright when the layer is turned on. Untested in this sandbox (no
  network); degrades to a visible alert if the fetch fails at demo time.
- **Basemap** — OSM standard (roads/labels) or Carto Light.
- **Dark mode** — CSS-variable based, persisted per-browser.
- **Confidence filter** — hides low-sample-count cells (`qoe_confidence_score`)
  from the ranked lists and investigation queues; the map always shows every cell.
- **Compare cells** — click the button, then click hexes on the map (up to 5)
  to line them up in a side-by-side table + multi-line QoE history chart.
- **(?) spatial-stats explainer** next to "Spatial-Temporal Hotspots" — a
  plain-language modal on what Gi* and Local Moran's I actually mean, tied
  to the exact class labels the dashboard uses (hot_99, HH, HL, etc.).
- **Mark planned / done** on investigation cards, persisted in the browser's
  `localStorage` (this is a standalone page you run locally, not a hosted
  multi-user app, so this is per-machine, not shared).
- **How to use this dashboard** — collapsible walkthrough at the top, written
  for a judge seeing this for the first time.

## 5. Testing

```bash
python3 test_data_pipeline.py
```

Seven checks: parquet row counts against `metadata.json`, the
dictionary-encoded string decode, the anchored/degraded correlation, the
built `panel.json`'s shape being a complete regular `cells x timestamps` grid,
and a regression test for a units bug caught during development (the
`time_range` field was briefly double-converted back to microseconds — now
covered so it can't silently reappear). All seven currently pass against this
repo's real data.

There's no coverage of `dashboard.js` itself (no headless browser available in
the environment this was built in) — the DOM logic is exercised manually. If
you have Playwright/Puppeteer available, that'd be the natural next step.

## 6. Known limitations, stated plainly

- `panel.json` is ~54 MB. Loaded once on page load via `fetch()`; on a slow
  connection or older machine this can take a few seconds. It's a flat file,
  not chunked per-timestamp — fine for a local judged demo, not what you'd
  ship to production users over the public internet as-is.
- The Dockerfile is written to standard conventions but **untested** in this
  environment (no Docker available here, and no network to pull the base
  image). Test it once locally before relying on it for the actual judging
  session.
- Root cause / cost / operator-contact are heuristics and editable
  placeholders, not pipeline outputs — see section 3.
- No trained ML model is invoked live; "Next-Hour Risk" on the map is the
  real historical `target_is_degraded_next_1h` outcome, not a live prediction
  from `src/modelling/`.
