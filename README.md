# GeoQoE Pulse — Dynamic Mobile QoE & Demand Pressure Intelligence System

GeoQoE Pulse is an explainable spatial-temporal decision-support prototype
for mobile network operators and regulators. It integrates eleven real
geospatial and network-measurement sources — OpenCellID, Google Open
Buildings V3, OpenStreetMap, WorldPop, OpenDOSM, Ookla Open Data, public
transport ridership, M-Lab NDT, VIIRS nighttime lights, ESA WorldCover,
and a published multi-operator field-measurement dataset (Mendeley
`dx5xyyfz2y`) — to estimate local
demand pressure and, where real QoE measurements exist, to relate that
pressure to service quality.

The pipeline uses real GeoPandas geometry throughout: a study-boundary
polygon drives H3 grid generation via `h3.polygon_to_cells`, CRS-correct
area and distance calculations are performed in EPSG:3375 (Peninsular
Malaysia RSO), and all spatial joins use proper GeoPandas operations
(`sjoin`, `sjoin_nearest`, `overlay`, zonal statistics). Hotspot detection
uses real `esda` implementations (Getis-Ord Gi\* and Local Moran's I, both
with genuine permutation-based p-values). Temporal features are
leakage-free — warm-up rows carry an honest `NaN` plus a `*_valid` flag
instead of back-filled values. The pipeline is deterministic: a single
global seed, set once before any randomness, is recorded in every
`metadata.json`.

> **Handing this to the modelling or dashboard team?** Start at
> **[HANDOFF.md](HANDOFF.md)** — it is the data contract: which file to use
> for which job, copy-paste starting code, what you can and cannot claim,
> and every known limit stated up front.

---

## ⚠️ What's real, what's simulated — read this first

| Layer | Real? | Use for |
|---|---|---|
| **Static GIS features** — towers (deduplicated), buildings, OSM POIs/roads/transit, WorldPop population, VIIRS nighttime lights, ESA WorldCover land cover | ✅ Real. Every `*_is_synthetic` flag is `False` across all 4,918 cells. Joins use real `sjoin`/`sjoin_nearest`/`overlay`/zonal-statistics, and extraction bounds are derived from the actual study grid. VIIRS and WorldCover have 100% cell coverage. | Any modelling or dashboard use. |
| **Sunway field inset** — `sunway_*` columns | ✅ Real, but covers **10 of 4,918 cells**. 30,925 real GNetTrack measurements across 3 operators and 2 technologies (Mendeley `dx5xyyfz2y`). `NaN` elsewhere is honest sparsity, never imputed. ⚠️ `sunway_test_event_dl_raw` has an **unverified unit** — see Known Limitations. | Independent ground-truth evidence; radio-quality (RSRP/SNR) context for those cells. |
| **`cell_quarter_dataset.parquet`** — real quarterly panel (41,448 rows = 3,566 cells × up to 14 quarters, 2023 Q1 – 2026 Q2) | ✅ **Real on both sides.** Every value is an independent Ookla measurement; nothing is downscaled, shared-shape or simulated. 36,193 rows carry a **real** one-step-ahead target. Cells genuinely differ in trajectory — 56% improving, 44% degrading — and 49.2% change relative-degradation state over the window. | **The dataset for any forecasting claim.** Real temporal holdout: train 2023 Q1 – 2025 Q4 (33,364 rows), test 2026 Q1 (2,829 rows). |
| **`cell_static_qoe_baseline.parquet`** — real Ookla labels on real GIS features | ✅ Real. `ookla_dl_mbps`/`ookla_rtt_ms`/`ookla_tests` are the real Ookla quarterly averages for cells with tile coverage (~60% of cells); `NaN` (never imputed) for the rest, flagged via `has_real_ookla_label`. Every feature is from the static GIS layer — zero QoE-derived circularity. | **The recommended dataset for any modelling claim you want to defend.** Train/validate on labelled cells; predict on the rest. |
| **`cell_time_dataset.parquet`** — hourly panel (826,224 rows = 4,918 cells × 168 hours) | ⚠️ **Hierarchical, not measured.** The hourly *shape* is **real** — the M-Lab NDT citywide diurnal curve over 2.05 M real tests (`qoe_hourly_shape_source`), keyed in UTC in the source CSV and **converted to Asia/Kuala_Lumpur at load**. The *level* is the cell's real Ookla quarterly average where one exists (`qoe_level_is_ookla_anchored`), formula-derived otherwise. Jitter, loss and upload remain simulated (`qoe_jitter_loss_is_simulated`). **One curve is shared by every cell**, so for anchored cells the panel is exactly `level(cell) × curve(hour)` — deterministic and **rank-one**. | Dashboard demos and building/testing a forecasting *framework*. **Not** for reporting forecast accuracy: the target is analytically reconstructable from two static columns plus the hour (verified: 100% match). See `metadata.json` → `panel_structure_disclosure`. |

If your team needs to defend a technical claim to judges, lead with
`cell_static_qoe_baseline.parquet`. If you want to demo what the dashboard
would look like with real hourly data, `cell_time_dataset.parquet` is
exactly that demo — labelled as one, not presented as measured.

**QoE index — what it actually is:** `target_qoe_index` is a **custom**
0–100 composite (logarithmic download/upload utility minus exponential
latency/loss penalty). It is *not* an ITU-T G.107, P.1203, or P.1204
implementation. Degradation labels are *not* derived from a QoE index
cutoff — see the "QoE Target Design" section below.

**Study area:** Klang Valley core — Kuala Lumpur federal territory,
Putrajaya federal territory, and six urbanised Selangor districts (Gombak,
Hulu Langat, Klang, Kuala Langat, Petaling, Sepang). Rural/exurban
Selangor (Hulu Selangor, Sabak Bernam, etc.) is deliberately excluded.

---

## 📁 Repository Structure

```text
geoqoe_pulse/
├── run_pipeline.py                 # THE entry point — 10 stages, ~27 min
├── config.py                       # CRS, H3 res, boundary, seed, source registry
│                                   # + vintages, thresholds, disclosures
├── run_full_data_pipeline.py       # deprecated shim -> run_pipeline.main()
├── README.md · HANDOFF.md          # HANDOFF.md is the data contract — start there
│
├── scripts/
│   ├── extract/                    # raw source -> per-H3 layer (run before the pipeline)
│   │   ├── viirs.py                #   VNP46A3 .h5      -> nighttime lights, monthly + long
│   │   ├── worldcover.py           #   ESA tiles        -> 11 land-cover classes
│   │   ├── dem.py                  #   Copernicus GLO-30-> elevation + ruggedness
│   │   ├── building_heights.py     #   OSM PBF          -> building heights
│   │   └── merge_layers.py         #   VIIRS + WorldCover -> one derived-layer file
│   ├── analysis/                   # one-off studies, each writes a JSON/CSV report
│   │   ├── analyse_demand_vs_degradation.py    # the proposal's central question
│   │   ├── check_quarterly_baselines.py        # is the dataset learnable?
│   │   ├── test_sunway_separability.py         # does the panel's assumption hold?
│   │   └── check_mcmc_register_bands.py         # why the register can't give capacity
│   └── trim_boundary.py            # reproducible study-boundary filter
│
├── src/data_preprocessing/         # pipeline modules, imported by run_pipeline
│   ├── geometry_utils.py           # H3 grid from the real boundary
│   ├── data_downloader.py          # spatial joins for the wide-area sources
│   ├── ookla_quarterly.py          # 14 real quarters -> the forecasting dataset
│   ├── viirs_quarterly.py          # per-cell per-quarter activity growth
│   ├── district_population.py      # DOSM annual series -> time-varying demand
│   ├── mlab_diurnal_shape.py       # real hourly curve, UTC -> local
│   ├── ingest_sunway_real.py       # real field measurements -> per-H3 inset
│   ├── coverage_bias.py            # propensity, common support, label confidence
│   ├── application_profiles.py     # per-application QoE (voice/browse/video/...)
│   ├── spatial_processor.py        # libpysal weights, Gi*, Local Moran's I
│   ├── feature_engineer.py         # causal lags, demand pressure, peak window
│   ├── qoe_target.py               # absolute + relative degradation targets
│   ├── data_validator.py           # structural checks + regression guards
│   └── metadata_utils.py           # provenance, vintages, disclosures
│
├── data/
│   ├── boundary/                   # study_boundary.geojson
│   ├── raw/                        # 11 real sources (git-ignored)
│   ├── processed/                  # per-H3 intermediates from scripts/extract/
│   └── outputs/                    # ML-ready deliverables + reports
│
├── app.py · data_loader.py         # Streamlit dashboard (dashboard team)
└── dashboard/                      # static dashboard build (dashboard team)
```

> `src/modelling/` is **obsolete and unused** — the current trained model lives
> under `Trained AI Model/`. Ignore it.

---

## 🔧 Pipeline Stages

`run_pipeline.py` runs ten stages in sequence:

| Stage | Description |
| :---: | :--- |
| **1** | **H3 Study Grid** — Loads the study boundary and generates 4,918 H3 res-8 cells via `h3.polygon_to_cells`. Stamps each cell with an `h3_parent_res6` grouping key for spatial holdout splits. Exports grid geometry to `data/processed/`. |
| **2** | **Spatial Joins & Baselines** — Real joins for all raw sources (`sjoin`, `sjoin_nearest`, `overlay`, zonal stats), with OpenCellID records deduplicated to approximate physical sites. **2a:** merges the real VIIRS nighttime-lights, ESA WorldCover land-cover and Sunway field-measurement layers **before** the baseline is built, so they reach every downstream table (previously this ran after export, and those columns reached no consumer). **2b/2c:** computes the honest static Ookla baseline — cells with real tile coverage get a real measurement, the rest retain `NaN`. **2d:** assembles the hourly panel, applying the real M-Lab curve converted from UTC to local time. |
| **3** | **Temporal Features** — Leakage-free lag/rolling stats (`shift()` only, no backfill), with `*_valid` flags. |
| **4** | **Demand Pressure Index** — Real static-feature base (`demand_pressure_base`), plus a historical-QoE component explicitly flagged as circular with the simulated panel. |
| **5** | **Spatial Features** — Real `libpysal` 1-ring adjacency; spatial lags, variance, rolling mean. |
| **6** | **Hotspot Detection** — Real `esda.G_Local` / `esda.Moran_Local`, 99 permutations, genuine p-values (bottom out at exactly 0.01 = 1/(99+1)), confidence-band classifications, HH/LH/LL/HL quadrant labels. Ordinal integer encodings of categorical columns added for ML compatibility. |
| **7** | **QoE Targets & Forecasting** — Computes the custom composite QoE index (0–100), a 1–5 quality rescaling, binary same-hour degradation, 3-class severity, and a genuine 1-hour-ahead forecast target via `groupby.shift(-1)`. Degradation is threshold-based on raw dl/rtt (see QoE Target Design below). Pre-computes priority score components for the dashboard. |
| **8** | **Validation** — Structural checks (duplicate keys, invalid geometry, CRS consistency) plus regression guards: target consistency (hard error if `target_is_degraded` and `target_degradation_class` ever disagree), QoE index score-range check, anchor-status/target-leakage audit, and a **diurnal timezone alignment guard** (hard error if the panel's worst hour doesn't fall in the evening — the signature of the UTC/local M-Lab bug). |
| **9** | **Feature Dictionary** — `feature_dictionary.xlsx`: every column across the 4 output tables, `is_feature_for_current_target` / `is_feature_for_next_hour_target` flags, non-null percentages, dtypes, and `leakage_risk_note`. **Completeness is asserted** — the pipeline fails if any exported column is missing from the dictionary, or vice versa. |
| **10** | **Export** — Compressed ZSTD `.parquet` files, `feature_dictionary.xlsx`, `data_quality_report.json` (validator output, previously discarded), and `metadata.json` (source fingerprints + vintages, `simulation_disclosure` with per-component provenance, `panel_structure_disclosure`, `high_leakage_risk_features`, `known_limitations`) to `data/outputs/`. |

---

## 💾 Raw Datasets & Google Drive Access

All raw datasets are hosted on Google Drive:

👉 **[Google Drive Raw Datasets Folder](https://drive.google.com/drive/folders/1EG-etCo-w5oxZ3QlJQFvC-xa1H-_myGX?usp=sharing)**

Download all files and place them into `data/raw/`. The directory is
git-ignored; raw files are not checked in.

---

## 📊 Dataset Inventory & Sources

| Dataset | Description | Source | File in `data/raw/` |
| :--- | :--- | :--- | :--- |
| **Ookla Open Data Mobile** | Mobile download/upload/latency tiles — **14 quarters, 2023 Q1 – 2026 Q2** | [AWS S3 `ookla-open-data`](https://github.com/teamookla/ookla-open-data) | fetched per quarter, extracted, raw deleted |
| **Google Open Buildings V3** | Building footprints & built-up area | [Google Open Buildings](https://sites.research.google/open-buildings/) | `31d_buildings.csv.gz` |
| **OpenCellID Cell Towers** | Cellular base station locations (MCC 502) | [OpenCellID](https://opencellid.org/) | `502.csv.gz` |
| **OpenStreetMap** | Roads, transit stops, POIs | [Geofabrik](https://overpass-api.de/) | `malaysia-singapore-brunei-260727.osm.pbf` |
| **WorldPop Malaysia 2020** | Gridded 100 m population counts | [WorldPop Hub](https://hub.worldpop.org/) | `mys_ppp_2020_UNadj.tif` |
| **OpenDOSM Census Data** | Official Malaysian population statistics | [OpenDOSM](https://data.gov.my/) | `population_malaysia.csv` |
| **Public Transit Ridership** | Daily LRT/MRT/Monorail ridership | [data.gov.my](https://data.gov.my/) | `ridership_headline.csv` |
| **DOSM district population** | Official population by district, **annual 2020–2025** — the only time-varying demand source | [OpenDOSM](https://data.gov.my/) | `population_district.parquet` |
| **M-Lab NDT hourly curve** | Real hour × day-of-week download/RTT curve (2.05 M tests, **UTC-keyed**) | [Measurement Lab BigQuery](https://www.measurementlab.net/data/) | `mlab.csv` |
| **Sunway multi-operator QoE** | 30,925 real GNetTrack field measurements, 3 operators, Dec 2024 – Apr 2025 | [Mendeley `10.17632/dx5xyyfz2y.1`](https://doi.org/10.17632/dx5xyyfz2y.1) | `sunway_qoe.csv` |
| **VIIRS VNP46A3** | Monthly nighttime-lights radiance, 6 composites | [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/) | `VNP46A3_2-.../*.h5` |
| **ESA WorldCover 2021 v200** | 10 m global land cover, tiles N00E099 + N03E099 | [ESA WorldCover](https://esa-worldcover.org/) | `worldcover/*.tif` |

All eleven are fingerprinted (size, mtime, partial SHA-256 or file
count/bytes) with an acquisition **vintage** in every `metadata.json`.

**No per-CELL per-hour QoE measurement source exists in this list or
anywhere in the project.** M-Lab gives a real hourly *shape* for the city
as a whole; Ookla gives a real *level* per cell. Combining them is a
downscaling method under a stated separability assumption — not a
measurement. `cell_static_qoe_baseline.parquet` remains the recommended
modelling dataset.

---

## 🚀 Quick Setup & Installation

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd geoqoe_pulse
pip install -r requirements.txt
# Download raw datasets from the Google Drive folder above into data/raw/
python run_pipeline.py
```

The pipeline scans `data/raw/`, ingests all 7 datasets, runs the full
10-stage workflow, and writes outputs to `data/outputs/`.

---

## 📐 Output Datasets

1. **`cell_static_features.parquet`** (4,918 cells × 31 columns) — Real static GIS features only (towers, buildings, POIs, roads, population, demand pressure base). No QoE columns.

2. **`cell_static_qoe_baseline.parquet`** (4,918 cells × 35 columns) — Static features plus real Ookla-anchored QoE labels. Cells with real Ookla tile coverage (~60%) have real `ookla_dl_mbps`/`ookla_rtt_ms`/`ookla_tests`; the rest are `NaN`, flagged via `has_real_ookla_label`.

3. **`cell_time_dataset.parquet`** (826,224 rows × 108 columns = 4,918 cells × 168 hours) — The simulated hourly panel: temporal lags, rolling stats, spatial lags/variance, Gi\*/Local Moran's, demand pressure index, historical hotspot frequency, plus same-hour and next-hour targets.

4. **`qoe_targets.parquet`** (826,224 rows × 14 columns) — A dedicated target-only table mirroring the target columns in `cell_time_dataset.parquet`, for lightweight joins when only labels are needed.

6. **`feature_dictionary.xlsx`** (188 rows × 8 columns) — `file`, `column`, `dtype`, `pct_non_null`, `is_feature_for_current_target`, `is_feature_for_next_hour_target`, `leakage_risk_note`, `description`.

7. **`metadata.json`** — Pipeline version, timestamp, CRS, seed (`42`), git commit, raw-file checksums, `synthetic_fallback_summary`, `simulation_disclosure`, `high_leakage_risk_features`, and `known_limitations`.

---

## 🛠 Model Training Instructions

### Option A: the static spatial baseline

```python
import pandas as pd

df = pd.read_parquet("data/outputs/cell_static_qoe_baseline.parquet")
labelled = df[df["has_real_ookla_label"]]          # ~2,950 cells with real Ookla targets
unlabelled = df[~df["has_real_ookla_label"]]        # real features, prediction targets

# Don't hand-maintain an exclusion list — read the approved feature list
# straight from the dictionary. It now covers every exported column
# (completeness is asserted at build time), so this picks up the VIIRS,
# WorldCover and Sunway features automatically.
df_dict = pd.read_excel("data/outputs/feature_dictionary.xlsx")
feature_cols = df_dict[
    (df_dict["file"] == "cell_static_qoe_baseline")
    & (df_dict["is_feature_for_current_target"])
]["column"].tolist()

X, y = labelled[feature_cols], labelled["ookla_dl_mbps"]  # or classify qoe_risk_label
# Train/validate on `labelled` using spatial k-fold (group by h3_parent_res6,
# hold out whole groups — don't randomly sample individual cells, spatially
# autocorrelated neighbours would leak across a random split).
# Then predict on `unlabelled`.
```

Every number in `X` and `y` here is real. This is the dataset to point a
hackathon judge at if asked "is this real data."

### Option B (recommended for forecasting): the real quarterly panel

Every value here is measured, on both sides of the prediction. This is the
only forecasting claim in the project that survives scrutiny.

```python
import pandas as pd
from xgboost import XGBClassifier

df = pd.read_parquet("data/outputs/cell_quarter_dataset.parquet")
df_dict = pd.read_excel("data/outputs/feature_dictionary.xlsx")

features = df_dict[
    (df_dict["file"] == "cell_quarter_dataset")
    & (df_dict["is_feature_for_next_hour_target"])   # = next-QUARTER mode
]["column"].tolist()

valid = df["target_next_q_valid"] == 1               # real one-step-ahead label only
train = valid & (df["quarter_index"] <  2026 * 4)    # 2023Q1-2025Q4
test  = valid & (df["quarter_index"] >= 2026 * 4)    # 2026Q1

y = "target_is_below_area_norm_next_q"               # ~22% positive; the absolute
                                                     # label is ~2% and near-degenerate
model = XGBClassifier(n_estimators=300, max_depth=5, random_state=42)
model.fit(df.loc[train, features], df.loc[train, y])
print("Test PR-AUC on a REAL held-out quarter:", ...)
```

Group by `h3_parent_res6` if you also want a spatial holdout. `ookla_tests`
travels with every row — weight or filter by it, since a cell-quarter can
rest on very few real tests.

### Option C: the hourly panel (pipeline/dashboard demo only)

Query the feature dictionary rather than maintaining a manual exclusion
list — this also self-documents which target mode you are in:

```python
import pandas as pd
from xgboost import XGBClassifier

df_panel = pd.read_parquet("data/outputs/cell_time_dataset.parquet")
df_targets = pd.read_parquet("data/outputs/qoe_targets.parquet")
df_dict = pd.read_excel("data/outputs/feature_dictionary.xlsx")

# Next-hour forecast target — the recommended primary target for the panel.
# Current-hour QoE values (dl_median, rtt_median, etc.) are legitimate
# features here because they are genuinely observed before the hour being
# predicted.
next_hour_features = df_dict[
    (df_dict["file"] == "cell_time_dataset") &
    (df_dict["is_feature_for_next_hour_target"] == True)
]["column"].tolist()

df_panel = df_panel.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)

# Drop warm-up rows AND end-of-series rows with no future observation.
valid = (
    (df_panel["in_warmup_window"] == 0) &
    (df_targets["target_next_1h_valid"] == 1)
)
X = df_panel.loc[valid, next_hour_features]
y = df_targets.loc[valid, "target_is_degraded_next_1h"]

split_point = int(len(X) * 0.80)                    # temporal holdout
X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)
print("Test Accuracy:", model.score(X_test, y_test))
print("LABEL CLEARLY: synthetic-panel pipeline validation only — "
      "not evidence of real-world predictive performance.")
```

If you need the same-hour target (`target_is_degraded`) instead, filter
`df_dict["is_feature_for_current_target"] == True` — this additionally
excludes every raw QoE column and their same-hour spatial lags, which
would otherwise leak the answer.

---

## QoE Target Design

`target_qoe_index` (0–100) and `target_service_quality_1to5` (1–5) are a
**custom composite index** — logarithmic download/upload utility minus an
exponential latency/loss penalty, with positive weights summing to 1.0 so
the index can genuinely reach 100. They are **not** ITU-T G.107 E-model
MOS and **not** P.1203/P.1204 implementations.

Degradation labels are **not** derived from a QoE-index cutoff. They come
from `config.QOE_THRESHOLDS`, applied directly to `dl_median`/`rtt_median`:

- `target_degradation_class = 1` ("degraded") if `dl < 10 Mbps` **or** `rtt > 100 ms`
- `target_degradation_class = 2` ("severe") if `dl < 5 Mbps` **or** `rtt > 150 ms`
- `target_is_degraded = (target_degradation_class >= 1)`

Both labels are derived from the same threshold source in one function
call so they cannot disagree — enforced by construction.

### The relative target — why an absolute cutoff isn't enough

`dl < 10 Mbps` fires on only **2.3%** of labelled cells, against a real
Ookla median of **175 Mbps** and a p25 of **108 Mbps**. That makes
"degraded" a rare tail event defined by a constant nobody calibrated
against this city, and it cannot express the proposal's actual question —
*localised* degradation that aggregate statistics hide. A cell at 40 Mbps
surrounded by neighbours at 200 is invisible to a 10 Mbps cutoff, and is
exactly what an operator should investigate.

So the pipeline ships both. `compute_relative_targets()` ranks every cell
against **all other cells at the same timestamp**:

- `target_qoe_percentile_in_area` — 0–100; 50 means "exactly typical for
  the study area at this hour"
- `target_is_below_area_norm` — 1 when that percentile is below
  `config.RELATIVE_DEGRADATION_PERCENTILE` (default 25)

plus `_next_1h` versions of both.

**Caveat, measured every run:** while one shared diurnal curve drives the
panel, a common positive factor cannot reorder cells, so these columns are
**time-invariant by construction** — they carry spatial information only.
`validate_relative_targets()` reports the actual time-variation each run,
so this is checked rather than assumed. Once real per-district M-Lab
curves exist the ranking genuinely moves through the week, and that
warning stops firing. The columns ship now so the target contract is
stable and the modelling team can build against the definition that stays
correct as the data improves.

---

## 🔑 Key Design Decisions

- **CRS discipline** — WGS84 (`EPSG:4326`) for storage/export; all distance/area/length calculations in Peninsular Malaysia RSO (`EPSG:3375`). Raw degree-based math is never used.

- **Two-signal QoE provenance** — `qoe_level_is_ookla_anchored` (was a real anchor used to re-centre the formula?) and `qoe_hourly_shape_is_simulated` (always `True`) replace a single ambiguous flag that could imply measured data exists where it doesn't.

- **Temporal integrity** — `shift()`-only lags, `*_valid` flags, no backfill. Warm-up null counts are exact multiples of the cell count (4,918 / 9,836 / 118,032 for 1h/2h/24h lags), confirming genuinely causal construction.

- **Target integrity** — Same-hour and next-hour degradation labels share one threshold source and cannot disagree. A genuine 1-hour-ahead forecast target exists alongside the same-hour classification target.

- **Target-mode-aware leakage discipline** — `feature_dictionary.xlsx` provides `is_feature_for_current_target` / `is_feature_for_next_hour_target` columns because what counts as leakage depends on whether you are predicting the current hour or the next. Anchor-status columns are hard-excluded from both modes (not just flagged) because nearly all positive labels on the simulated panel come from anchored cells.

- **Spatial holdout grouping** — Every output table carries `h3_parent_res6` (H3 resolution-6 parent, ~130 groups) so a real spatial train/test split (hold out whole groups) doesn't require deriving `h3.cell_to_parent()` by hand. A random split on individual `h3_index` rows leaks across spatially autocorrelated neighbours.

- **Reproducibility** — Single global seed (`config.GLOBAL_RANDOM_SEED = 42`), set once before any randomness. Every run's `metadata.json` records the seed, git commit, and raw-file checksums.

- **Honesty about what's real** — `cell_static_qoe_baseline.parquet` exists specifically so the project has a modelling-ready deliverable that requires zero caveats. `cell_time_dataset.parquet` remains useful, clearly labelled as what it is.

---

## Known Limitations

These are also recorded in every `metadata.json`:

1. **No per-cell per-hour QoE data exists** — `cell_time_dataset.parquet` combines a real citywide hourly shape (M-Lab) with a real per-cell level (Ookla). It is a downscaled panel, not a measurement.

2. **The HOURLY panel is rank-one and its forecast target is reconstructable** (this does NOT apply to `cell_quarter_dataset.parquet`, where every value is an independent real measurement — use that for any forecasting claim) — because one M-Lab curve is shared by every cell, an anchored cell's hourly value is exactly `ookla_static(cell) × curve(hour-of-week)` with no noise term. Verified: `dl`/`rtt` reconstruct with max absolute error `0.0`, and the reconstructed degradation label matches the stored one on **100%** of anchored rows. Any forecast metric computed on this panel measures memorisation. There is also no cell-specific temporal dynamics, and Gi\* hotspot recurrence varies only through the noise injected into unanchored cells. Fixing this needs per-district M-Lab curves, not a modelling change.

3. **`sunway_test_event_dl_raw` has an unverified unit** — it is the raw GNetTrack `TESTDOWNLINK` value, deliberately *not* named `*_mbps`. The source file contradicts an Mbps reading twice: `TESTUPLINK` reads ~10× higher than `TESTDOWNLINK`, and read as Mbps the values (429–558) exceed the same dataset's own passive throughput peak (103.9 Mbps) in the same cells. The Ookla-vs-Sunway correlation previously published in `ookla_vs_sunway_validation.json` has been **withdrawn** — it was invalid on units *and* on sample size (n=8 cells, median 8 test events each). See `data_quality_report.json` → `sunway_ground_truth`.

4. **The M-Lab curve is citywide and not ASN-filtered** — one shape for all 4,918 cells, and the source query did not separate mobile from fixed-broadband tests. Treat it as "real regional diurnal pattern, traffic mix not verified."

5. **The shared-shape assumption is NOT supported by the only test we can run.** Building the panel as `level(cell) × shape(hour)` assumes cells share a relative diurnal shape and differ only in level. Tested against the Sunway field data — the one place with real per-cell timestamped measurements — that assumption **fails for throughput**, the very quantity the shape is applied to:

   | Metric | Median per-cell R² vs shared shape | Dispersion across cells | Verdict |
   |---|---|---|---|
   | Passive downlink throughput | 0.54 | **0.67** | ❌ Not supported |
   | Median RSRP (control) | 0.82 | 0.01 | ✅ Supported |

   Radio conditions are genuinely shared between neighbouring cells; load-driven throughput is not — even inside a single ~6 km² area. The RSRP result acts as a control showing the method detects sharing when it exists, so the throughput failure is not a measurement artefact. **Caveats:** only 4 cells cleared the ≥100-records-per-cell-hour threshold, coverage is 09:00–16:00 only (so the 19:00–00:00 evening trough is untestable), and the throughput metric is a passive activity proxy, not a speed test. Treat this as suggestive evidence against the current construction, not proof. Full output: `data/outputs/separability_test.json`, reproducible via `python scripts/test_sunway_separability.py`.

6. **`is_peak_period` is an assumption, not a measurement** — it hardcodes local 08–10 and 17–20 commuter hours. The real M-Lab curve for this city troughs at **19:00–00:00 (worst 22:00)** with no morning dip at all. The pipeline warns when the configured window doesn't match the measured curve.

7. **Mixed source vintages under one timestamp** — the panel's calendar window is chosen to overlap the ridership file's real coverage; it is *not* when the measurements were taken. Sources span 2020 (WorldPop) to 2026 (OSM, ridership). Per-source `vintage` fields are in `metadata.json`.
8. **No real network capacity/traffic data exists** — demand-pressure features are activity proxies. Say "QoE degradation associated with estimated demand pressure," not "confirmed congestion."
9. **`tower_count` is approximate** — a deduplicated approximation of physical sites from OpenCellID, not an authoritative operator inventory.
10. **Custom QoE index, not ITU-T** — `target_qoe_index` / `target_service_quality_1to5` are a custom composite, not a validated G.107 MOS.
11. **`demand_pressure_index` is circular with the panel** — its historical-QoE component (`lag_24h_rtt_median`) feeds back into the same formula that generated the hourly values. Use `demand_pressure_base` for a non-circular, real-data-only feature.
12. **Anchor-status determines degradation labels** — on the simulated panel, positive `target_is_degraded_next_1h` labels are concentrated in Ookla-anchored cells. `qoe_level_is_ookla_anchored` is a near-trivial predictor and is hard-excluded from the default feature lists. Report metrics split by anchor status or restrict to anchored cells.
13. **Use spatial holdout splits** — `h3_parent_res6` (~130 groups) is on every output table. Group by it and hold out whole groups; a random split on individual `h3_index` rows leaks across spatially autocorrelated neighbours.
