"""
GeoQoE Pulse Configuration File
Central configuration for CRS, H3 resolution, spatial bounds, study boundary,
reproducibility, and feature weights.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Base Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BOUNDARY_DIR = DATA_DIR / "boundary"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_DIR = DATA_DIR / "outputs"

# Ensure directories exist
for folder in [RAW_DIR, BOUNDARY_DIR, PROCESSED_DIR, FEATURES_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Spatial Reference Systems
# --------------------------------------------------------------------------
# WGS84 — used for all storage/export (lat/lon, web-map compatible)
CRS_WGS84 = "EPSG:4326"
# GDM2000 / Peninsular Malaysia RSO — projected, metres. Used for any
# distance/area calculation (H3 cell area, nearest-tower distance, road
# length, building density) since WGS84 degrees are not equal-distance.
CRS_PENINSULAR_MYS = "EPSG:3375"
CRS_EAST_MYS = "EPSG:3376"  # GDM2000 / East Malaysia BRSO (not used — Klang Valley only)

# Rule of thumb used throughout src/data_preprocessing/:
#   - Store & export geometry in CRS_WGS84.
#   - Reproject to CRS_PENINSULAR_MYS ("to_crs") immediately before any
#     .area / .length / .distance call, then reproject results (not
#     geometries) back if needed. Never compute area/length/distance in
#     raw WGS84 degrees.

# --------------------------------------------------------------------------
# Study Boundary
# --------------------------------------------------------------------------
# Real GADM level-2 administrative polygon, dissolved. As of this version it
# covers the Klang Valley CORE districts only:
#   Gombak, Hulu Langat, Klang, Petaling, Sepang, Kuala Langat, Putrajaya
# Two districts that are NOT part of any common "Klang Valley" definition —
# Hulu Selangor (rural/interior) and Sabak Bernam (rural, north coast) — were
# removed from the original full-Selangor GADM extract (see
# scripts/trim_boundary.py) because they diluted hotspot detection with very
# low-density, low-POI cells and were never part of the intended study area.
#
# KNOWN GAP: Kuala Lumpur (W.P. Kuala Lumpur) is NOT included. No KL boundary
# file was available when this boundary was built. KL is the single highest
# mobile-demand area in the country and its absence is a real limitation of
# every hotspot/demand result in this project. Before treating this as the
# final study area, download the GADM MYS level-2 "W.P. Kuala Lumpur" polygon
# and merge it in via KLANG_VALLEY_DISTRICT_NAMES below, then re-run the full
# pipeline (H3 grid + every spatial join depends on the boundary).
STUDY_BOUNDARY_PATH = BOUNDARY_DIR / "study_boundary.geojson"

# Defensive filter: if a raw, unfiltered Selangor-wide GADM extract is ever
# reloaded (e.g. someone re-downloads NAME_1 == "Selangor" from scratch),
# geometry_utils.load_study_boundary() uses this list to automatically drop
# out-of-scope districts rather than silently including all of Selangor
# again. Matched against the GADM "NAME_2" column (falls back to "NAME_1"
# for Putrajaya, which has no NAME_2).
KLANG_VALLEY_DISTRICT_NAMES = {
    # Matched against GADM's actual NAME_2 spelling (no spaces).
    "Gombak", "HuluLangat", "Klang", "Petaling", "Sepang", "KualaLangat",
    "Putrajaya", "KualaLumpur",  # KualaLumpur included pre-emptively for when KL is added
}
STUDY_AREA_LABEL = (
    "Klang Valley core districts (Selangor: Gombak, Hulu Langat, Klang, "
    "Petaling, Sepang, Kuala Langat; plus Putrajaya). Kuala Lumpur is NOT "
    "included — see STUDY_BOUNDARY_PATH comment above."
)

# Fallback bounding box — only used if STUDY_BOUNDARY_PATH is missing, and a
# clear warning is logged whenever this fallback fires. Kept for backward
# compatibility with the original rectangular-bbox workflow.
STUDY_BBOX = {
    "min_lon": 101.50,
    "min_lat": 3.00,
    "max_lon": 101.80,
    "max_lat": 3.30
}

# --------------------------------------------------------------------------
# H3 Hexagonal Grid
# --------------------------------------------------------------------------
H3_PRIMARY_RESOLUTION = 8    # ~0.74 km^2 cell area, ~461m edge
H3_SECONDARY_RESOLUTION = 9  # ~0.11 km^2 cell area, ~174m edge

# --------------------------------------------------------------------------
# Temporal Configuration
# --------------------------------------------------------------------------
TIME_FREQ = "1h"
ROLLING_WINDOW_24H = 24
ROLLING_WINDOW_7D = 168
# Rows within this many hours of each cell's series start are a "warm-up"
# window: rolling/lag features are not yet fully backed by real history.
LAG_WARMUP_HOURS = 24

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# Single global seed applied once, at the very start of the pipeline, before
# any random sampling (real-data fallback generation included). Do not seed
# anywhere else — re-seeding mid-pipeline silently breaks the intended
# reproducibility guarantee.
GLOBAL_RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Raw Dataset Registry
# --------------------------------------------------------------------------
# Central place documenting expected raw filenames/schemas, so ingestion
# code and this config never drift apart.
RAW_DATASETS = {
    "ookla": RAW_DIR / "2024-10-01_performance_mobile_tiles.parquet",
    "buildings": RAW_DIR / "31d_buildings.csv.gz",
    "towers": RAW_DIR / "502.csv.gz",
    "osm": RAW_DIR / "malaysia-singapore-brunei-260727.osm.pbf",
    "worldpop": RAW_DIR / "mys_ppp_2020_UNadj.tif",
    "dosm_population": RAW_DIR / "population_malaysia.csv",
    "ridership": RAW_DIR / "ridership_headline.csv",
}

# --------------------------------------------------------------------------
# Demand Proxy Weights
# --------------------------------------------------------------------------
# demand_pressure_base uses only the first four (population/poi/built_area/
# transport) — all real static GIS inputs, zero QoE dependency, safe to use
# as a feature against ANY target, current or simulated.
#
# The full demand_pressure_index additionally folds in "historical_qoe"
# (lag_24h_rtt_median) and "nearby_demand" (spatial lag of itself). Because
# the simulated hourly panel's rtt_median is itself generated as a function
# of population/POI (see data_downloader.build_dynamic_panel), including
# historical_qoe creates a circular path back to the same real static inputs
# that already drive demand_pressure_base — i.e. part of any observed
# demand_pressure_index -> target correlation in this dataset is guaranteed
# by construction, not discovered. HIGH_LEAKAGE_RISK_FEATURES below flags
# this for the modelling team; prefer demand_pressure_base over the full
# demand_pressure_index until real (non-simulated) QoE data is available.
DEMAND_WEIGHTS = {
    "population": 0.30,
    "poi": 0.20,
    "built_area": 0.15,
    "transport": 0.15,
    "historical_qoe": 0.10,
    "nearby_demand": 0.10,
}

# Columns that are mechanically linked to the simulated QoE generator and
# should not be presented as independently-validated predictors while the
# panel remains simulated. Exported into feature_dictionary.xlsx and
# metadata.json so the modelling team sees this without reading source code.
HIGH_LEAKAGE_RISK_FEATURES = {
    "demand_pressure_index": (
        "Circular with the simulated QoE generator via its historical_qoe "
        "component (lag_24h_rtt_median). Use demand_pressure_base instead "
        "for a non-circular, real-data-only feature."
    ),
    "demand_pressure_index_pre_spatial": (
        "Same circularity as demand_pressure_index, pre-spatial-smoothing."
    ),
}

# --------------------------------------------------------------------------
# QoE Thresholds for Classification
# --------------------------------------------------------------------------
# SINGLE source of truth for every degradation-related label in the project.
# target_is_degraded (binary) and target_degradation_class (3-class) are both
# derived from these same two tiers in src/data_preprocessing/qoe_target.py,
# so the two labels can no longer disagree with each other on the same row
# (previously: is_degraded used dl<10/rtt>100 while the class label used a
# separately hardcoded dl<15/rtt>80 — fixed).
QOE_THRESHOLDS = {
    "dl_degraded_mbps": 10.0,   # class >= 1 ("degraded") and is_degraded=1
    "rtt_degraded_ms": 100.0,
    "dl_severe_mbps": 5.0,      # class == 2 ("severely degraded")
    "rtt_severe_ms": 150.0,
    "dl_excellent_mbps": 50.0,
    "rtt_excellent_ms": 30.0,
}

# --------------------------------------------------------------------------
# QoE Index Weights
# --------------------------------------------------------------------------
# Positive (utility) weights MUST sum to 1.0, otherwise the composite index's
# theoretical ceiling silently falls below 100 (previous bug: 0.5 + 0.3 = 0.8
# meant target_qoe_index could never exceed 80, and the derived 1-5 quality
# score could never exceed 4.2, despite being documented as spanning the
# full range). Verified against src/data_preprocessing/qoe_target.py.
QOE_INDEX_WEIGHTS = {
    "download": 0.6,
    "upload": 0.4,
    "rtt_penalty": 0.4,
    "loss_penalty": 0.4,
}

# --------------------------------------------------------------------------
# Modelling target
# --------------------------------------------------------------------------
# The dataset's default target used to be same-hour degradation, i.e. "is
# this cell degraded right now" using features measured in that same hour —
# not a forecast. The recommended primary MVP target is now the 1-hour-ahead
# shifted label, computed by qoe_target.compute_next_hour_targets().
PRIMARY_ML_TARGET = "target_is_degraded_next_1h"
FORECAST_HORIZON_HOURS = 1

# --------------------------------------------------------------------------
# Data honesty / simulation disclosure
# --------------------------------------------------------------------------
# No organiser QoE dataset and no per-hour QoE source of any kind exists for
# this project. Ookla provides one real, STATIC, quarterly average per cell
# for the ~41% of cells it covers — nothing hourly. Every hourly QoE value in
# cell_time_dataset.parquet (dl/ul/rtt/jitter/loss, whether or not a real
# Ookla anchor exists for that cell) is generated by a diurnal-shape formula,
# not observed. This is tracked explicitly (see data_downloader.py and
# metadata_utils.py) rather than left implicit in a single ambiguous flag.
PANEL_QOE_HOURLY_SHAPE_IS_SIMULATED = True
SIMULATION_DISCLAIMER = (
    "The hourly QoE panel (cell_time_dataset.parquet) is a simulated "
    "diurnal-demand scenario, not observed network measurements. It is a "
    "software-integration and dashboard-demo fixture only. For a modelling "
    "claim that stands up to review, use cell_static_qoe_baseline.parquet "
    "(real Ookla-anchored targets, zero synthetic QoE) or obtain real "
    "repeated-timestamp QoE measurements before training a forecast model "
    "and reporting its accuracy as evidence of real-world skill."
)

# --------------------------------------------------------------------------
# Pipeline / Metadata
# --------------------------------------------------------------------------
PIPELINE_VERSION = "3.0.0"  # bumped: target-consistency, leakage, and
                             # boundary-scope fixes applied — see CHANGELOG.md
