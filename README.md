# GeoQoE Pulse — Dynamic Mobile QoE & Demand Pressure Intelligence System

GeoQoE Pulse is an explainable spatial-temporal decision-support system for mobile network operators and regulators. It integrates multi-source geospatial open data (Ookla Open Data, OpenCellID, Google Open Buildings V3, OpenStreetMap, WorldPop, OpenDOSM, and public transport ridership) to detect recurring network quality degradation, estimate local demand pressure, and prepare machine learning modeling matrices.

---

## 📁 Repository Structure

```text
geoqoe_pulse/
├── config.py               # Shared settings (CRS: EPSG:3375, H3 Res 8, study bbox)
├── run_pipeline.py         # Master entry script
├── requirements.txt        # Python dependencies
├── .gitignore              # Git exclusion file
├── README.md               # Project documentation & guide
├── src/
│   ├── data_preprocessing/ # GIS alignment & spatial-temporal feature engineering
│   │   ├── data_downloader.py
│   │   ├── spatial_processor.py
│   │   ├── feature_engineer.py
│   │   └── qoe_target.py
│   ├── modelling/          # Machine learning model training & evaluation
│   │   └── train.py
│   └── dashboard/          # Interactive decision-support web interface
│       └── app.py
└── data/
    ├── raw/                # 7 Downloaded raw datasets (Ookla, OpenCellID, Buildings, OSM)
    ├── processed/          # Intermediate clean GIS layers (H3 grid checkpoint)
    ├── features/           # Intermediate spatial lag & temporal feature checkpoints
    └── outputs/            # Final master modeling deliverables
        ├── cell_time_dataset.parquet
        ├── cell_static_features.parquet
        ├── qoe_targets.parquet
        └── feature_dictionary.xlsx
```

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

### Step 3: Execute Master Data Pipeline
```bash
python run_pipeline.py
```

Running `python run_pipeline.py` scans `data/raw/`, ingests all 7 real datasets, executes spatial-temporal feature engineering, and updates the final master dataset in `data/outputs/cell_time_dataset.parquet`.

---

## 📊 Dataset Inventory & Sources

| Dataset Name | Description | Source | File Name in `data/raw/` |
| :--- | :--- | :--- | :--- |
| **Ookla Open Data Mobile** | Mobile download, upload, and latency performance tiles | [AWS S3 `ookla-open-data`](https://github.com/teamookla/ookla-open-data) | `2024-10-01_performance_mobile_tiles.parquet` |
| **Google Open Buildings V3** | High-resolution building footprints & built-up area | [Google Open Buildings](https://sites.research.google/open-buildings/) | `31d_buildings.csv.gz` |
| **OpenCellID Cell Towers** | Cellular base station locations & tech generation | [OpenCellID API](https://opencellid.org/) (MCC 502) | `502.csv.gz` |
| **OpenStreetMap Infrastructure** | Roads, transit stops, and POI activity centres | [Geofabrik / Overpass](https://overpass-api.de/) | `malaysia-singapore-brunei-260727.osm.pbf` |
| **WorldPop Malaysia 2020** | Gridded 100m population counts | [WorldPop Hub MYS](https://hub.worldpop.org/) | `mys_ppp_2020_UNadj.tif` |
| **OpenDOSM Census Data** | Official Malaysian population statistics | [OpenDOSM](https://data.gov.my/) | `population_malaysia.csv` |
| **Public Transit Ridership** | Daily LRT/MRT/Monorail station ridership surges | [data.gov.my](https://data.gov.my/) | `ridership_headline.csv` |

---

## 🛠 Model Training Instructions

To train machine learning models (XGBoost, LightGBM, Random Forest) using `cell_time_dataset.parquet`:

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
