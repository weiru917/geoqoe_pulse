# GeoQoE Pulse — Dynamic Mobile QoE & Demand Pressure Intelligence System

GeoQoE Pulse is an explainable spatial-temporal decision-support system for mobile network operators and regulators. It integrates seven open geospatial data sources — Ookla Open Data, OpenCellID, Google Open Buildings V3, OpenStreetMap, WorldPop, OpenDOSM, and public transport ridership — to detect recurring network quality degradation, estimate local demand pressure, and produce machine-learning-ready modeling matrices.

The system uses real GeoPandas geometry throughout the pipeline: a study-boundary polygon drives H3 grid generation via `h3.polygon_to_cells`, CRS-correct area and distance calculations are performed in a projected coordinate system (EPSG:3375), and all spatial joins use proper GeoPandas operations (`sjoin`, `sjoin_nearest`, `overlay`) rather than per-row approximations. Hotspot detection relies on statistically validated `esda` implementations (Getis-Ord Gi* and Local Moran's I) with z-scores, p-values, and confidence-band classifications. Temporal features are leakage-free — warm-up rows carry an honest `NaN` plus a `*_valid` flag instead of back-filled future values. The entire pipeline is deterministic end-to-end: a single global seed, set once before any randomness, ensures byte-identical numeric output across runs.

---

## 📁 Repository Structure

```text
geoqoe_pulse/
├── config.py                       # Central settings: CRS, H3 resolution, study boundary, seed, weights, thresholds
├── run_pipeline.py                 # 9-stage pipeline orchestrator
├── requirements.txt                # Python dependencies
├── .gitignore
├── README.md
│
├── src/
│   └── data_preprocessing/
│       ├── geometry_utils.py       # H3 polygon generation, study boundary loading, CRS-correct area/perimeter
│       ├── data_downloader.py      # Real spatial joins for all 7 data sources with synthetic-fallback flagging
│       ├── spatial_processor.py    # libpysal spatial weights, Getis-Ord Gi*, Local Moran's I, spatial lags
│       ├── feature_engineer.py     # Leakage-free temporal lags/rolling stats, demand pressure index
│       ├── qoe_target.py           # Composite QoE score, MOS, binary degradation, severity class targets
│       ├── data_validator.py       # Automated structural and statistical validation checks
│       └── metadata_utils.py       # Pipeline provenance metadata.json generation
│
└── data/
    ├── boundary/                   # Study boundary polygon (study_boundary.geojson)
    ├── raw/                        # 7 raw datasets (git-ignored; see Google Drive link below)
    ├── processed/                  # H3 grid geometry (klang_valley_h3_grid_res8.parquet)
    ├── features/                   # Intermediate engineered features
    │   ├── engineered_panel_features.parquet
    │   └── static_spatial_features.parquet
    └── outputs/                    # Final ML-ready deliverables
        ├── cell_time_dataset.parquet
        ├── cell_static_features.parquet
        ├── qoe_targets.parquet
        ├── feature_dictionary.xlsx
        └── metadata.json
```

---

## 🔧 Pipeline Stages

The pipeline is orchestrated by `run_pipeline.py` and executes nine stages in sequence:

| Stage | Description |
| :---: | :--- |
| **1** | **H3 Study Grid** — Loads the study boundary polygon and generates H3 resolution-8 hexagonal cells via `h3.polygon_to_cells`. Exports grid geometry to `data/processed/`. |
| **2** | **Spatial Joins** — Ingests all 7 raw datasets using real GeoPandas spatial joins (`sjoin`, `sjoin_nearest`, zonal stats). Missing sources fall back to documented synthetic formulas; every fallback row is explicitly flagged via a `<feature>_is_synthetic` column. |
| **3** | **Temporal Features** — Computes leakage-free lag and rolling-window statistics (`shift(1)`, `shift(24)`, 24h/7d rolling means). Warm-up rows retain `NaN` with a `*_valid` flag — no back-fill or current-value substitution. |
| **4** | **Demand Pressure Index** — Constructs a 0–100 composite demand index from real static features (population, POI density, built-up ratio, road density, historical QoE, nearby demand) using configurable weights. |
| **5** | **Spatial Features** — Builds H3 1-ring adjacency as a `libpysal.weights.W` object and computes spatial lags, spatial variance, and spatial rolling means across the panel. |
| **6** | **Hotspot Detection** — Runs Getis-Ord Gi* (`esda.G_Local`) and Local Moran's I (`esda.Moran_Local`) with permutation-based inference (z-scores, p-values, 90/95/99% confidence-band classifications, HH/LH/LL/HL quadrant labels). Also computes leakage-free historical hotspot frequency. |
| **7** | **QoE Targets** — Derives ITU-T G.107 & Weber-Fechner inspired composite QoE score (0–100), MOS (1.0–5.0), binary degradation flag, and three-class severity label. |
| **8** | **Validation** — Runs automated checks for duplicate keys, invalid geometry, CRS consistency, out-of-range values, outliers, and timestamp gaps. Structural failures raise `ValueError`; quality issues log warnings. |
| **9** | **Export** — Writes ML-ready parquet files, an Excel feature dictionary (per-column dtype, non-null %, description), and `metadata.json` (pipeline version, timestamp, CRS, seed, git commit hash, per-source real-vs-synthetic fractions). |

---

## 💾 Raw Datasets & Google Drive Access

All raw datasets are hosted on Google Drive:

👉 **[Google Drive Raw Datasets Folder](https://drive.google.com/drive/folders/1EG-etCo-w5oxZ3QlJQFvC-xa1H-_myGX?usp=sharing)**

Download all files and place them into `data/raw/` in your local project directory. The `data/raw/` directory is git-ignored; raw files are not checked in.

---

## 📊 Dataset Inventory & Sources

| Dataset | Description | Source | File in `data/raw/` |
| :--- | :--- | :--- | :--- |
| **Ookla Open Data Mobile** | Mobile download, upload, and latency performance tiles | [AWS S3 `ookla-open-data`](https://github.com/teamookla/ookla-open-data) | `2024-10-01_performance_mobile_tiles.parquet` |
| **Google Open Buildings V3** | High-resolution building footprints & built-up area | [Google Open Buildings](https://sites.research.google/open-buildings/) | `31d_buildings.csv.gz` |
| **OpenCellID Cell Towers** | Cellular base station locations & technology generation | [OpenCellID API](https://opencellid.org/) (MCC 502) | `502.csv.gz` |
| **OpenStreetMap Infrastructure** | Roads, transit stops, and POI activity centres | [Geofabrik / Overpass](https://overpass-api.de/) | `malaysia-singapore-brunei-260727.osm.pbf` |
| **WorldPop Malaysia 2020** | Gridded 100 m population counts | [WorldPop Hub MYS](https://hub.worldpop.org/) | `mys_ppp_2020_UNadj.tif` |
| **OpenDOSM Census Data** | Official Malaysian population statistics | [OpenDOSM](https://data.gov.my/) | `population_malaysia.csv` |
| **Public Transit Ridership** | Daily LRT/MRT/Monorail station ridership surges | [data.gov.my](https://data.gov.my/) | `ridership_headline.csv` |

---

## 🚀 Quick Setup & Installation

### Step 1: Clone Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd geoqoe_pulse
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Place Raw Data
Download raw datasets from the [Google Drive folder](https://drive.google.com/drive/folders/1EG-etCo-w5oxZ3QlJQFvC-xa1H-_myGX?usp=sharing) and place them into `data/raw/`.

### Step 4: Execute the Pipeline
```bash
python run_pipeline.py
```

The pipeline scans `data/raw/`, ingests all 7 datasets, runs the full 9-stage spatial-temporal feature engineering workflow, and writes final outputs to `data/outputs/`.

---

## 📐 Module Reference

### `config.py`
Central configuration file defining CRS settings (WGS84 for storage, EPSG:3375 for distance/area calculations), H3 resolution (res 8 primary, res 9 secondary), study boundary path and fallback bounding box, the global random seed, raw dataset file registry, demand proxy weights, QoE classification thresholds, and pipeline version.

### `geometry_utils.py`
Owns the spatial definition of "what is a cell." Loads the study boundary polygon (or falls back to a bounding-box rectangle with a logged warning), generates the H3 hexagonal grid via `h3.polygon_to_cells`, computes CRS-correct cell area and perimeter in the projected CRS, and exports `grid_geometry.parquet` / `.geojson`.

### `data_downloader.py`
Performs real GeoPandas spatial joins of all 7 raw datasets into the H3 grid. Uses `sjoin` for point-in-polygon aggregation (towers, buildings, POIs), `sjoin_nearest` for nearest-feature distance, `overlay` for area-weighted intersections, and `rasterstats` zonal statistics for WorldPop raster data. Every ingestion function returns a per-cell DataFrame with an explicit `<feature>_is_synthetic` boolean column. If a raw file is missing, the function logs a clear warning, falls back to a documented and seeded synthetic formula, and flags every row it generated.

### `spatial_processor.py`
Builds H3 1-ring adjacency as a `libpysal.weights.W` object, computes vectorized spatial lags and spatial variance across the panel, runs Getis-Ord Gi* (`esda.G_Local`) and Local Moran's I (`esda.Moran_Local`) with permutation-based p-values, and derives confidence-band hotspot categories and HH/LH/LL/HL quadrant labels.

### `feature_engineer.py`
Engineers leakage-free temporal features (1h and 24h lags via `shift()`, 24h and 7d rolling statistics) with explicit `*_valid` flags for warm-up rows. Constructs the demand pressure index from real static GIS features using configurable component weights.

### `qoe_target.py`
Computes ITU-T G.107 and Weber-Fechner inspired composite QoE score (0–100), Mean Opinion Score (1.0–5.0), binary degradation flag, and three-class severity label (normal / degraded / severe) from throughput and latency metrics.

### `data_validator.py`
Runs automated structural checks (duplicate composite keys, invalid geometry, CRS mismatch) and statistical quality checks (out-of-range values, outliers, timestamp gaps). Structural failures raise `ValueError`; quality issues are collected as warnings in a `ValidationReport`.

### `metadata_utils.py`
Generates `metadata.json` documenting pipeline version, generation timestamp, CRS, H3 resolution, global seed, git commit hash, raw source file checksums, feature column list, row counts, and per-source real-vs-synthetic fractions — so consumers of the exported parquet files can trace provenance without re-reading the code.

---

## 🛠 Model Training Instructions

To train machine learning models (XGBoost, LightGBM, Random Forest) on the exported `cell_time_dataset.parquet`:

```python
import pandas as pd
from xgboost import XGBClassifier

# 1. Load Master Dataset
df = pd.read_parquet("data/outputs/cell_time_dataset.parquet")

# 2. Exclude ID & Target columns from predictor matrix X
target_col = "target_is_degraded"
exclude_cols = [
    "h3_index", "timestamp", "dl_median", "rtt_median",
    "target_qoe_score", "target_mos", "target_is_degraded", "target_degradation_class"
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

# 3. Temporal Train/Test Split (80% Train, 20% Test)
df = df.sort_values("timestamp").reset_index(drop=True)
split_point = int(len(df) * 0.80)

X_train, y_train = X.iloc[:split_point], y.iloc[:split_point]
X_test, y_test = X.iloc[split_point:], y.iloc[split_point:]

# 4. Train Model
model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

print("Test Accuracy:", model.score(X_test, y_test))
```

---

## 🔑 Key Design Decisions

- **CRS discipline** — All geometry is stored and exported in WGS84 (EPSG:4326). Distance, area, and length calculations are performed after reprojection to GDM2000 / Peninsular Malaysia RSO (EPSG:3375). Raw WGS84 degree-based area/distance is never used.
- **Synthetic transparency** — Every feature that falls back to synthetic data is explicitly flagged via a `<feature>_is_synthetic` boolean column. Nothing is silently faked; callers can always filter `df[df["poi_is_synthetic"]]` to see exactly what isn't real.
- **Temporal integrity** — Lag and rolling features use `shift()` only — no `bfill()` or current-value substitution. Warm-up period rows retain `NaN` with an accompanying `*_valid` flag (0 = warm-up, 1 = fully backed by real history). The modelling team can choose to impute, mask, or drop these rows with full visibility.
- **Reproducibility** — A single global seed (`config.GLOBAL_RANDOM_SEED = 42`) is set once at the very start of the pipeline, before any random sampling. No mid-pipeline re-seeding.
- **Provenance** — Every pipeline run generates `metadata.json` with version, timestamp, CRS, seed, git commit, source file checksums, and per-source real-vs-synthetic fractions.
