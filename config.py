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
# Real, non-rectangular study-area polygon (placeholder approximation of the
# Klang Valley).
STUDY_BOUNDARY_PATH = BOUNDARY_DIR / "study_boundary.geojson"

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
DEMAND_WEIGHTS = {
    "population": 0.30,
    "poi": 0.20,
    "built_area": 0.15,
    "transport": 0.15,
    "historical_qoe": 0.10,
    "nearby_demand": 0.10,
}

# --------------------------------------------------------------------------
# QoE Thresholds for Classification
# --------------------------------------------------------------------------
QOE_THRESHOLDS = {
    "dl_degraded_mbps": 10.0,
    "rtt_degraded_ms": 100.0,
    "dl_excellent_mbps": 50.0,
    "rtt_excellent_ms": 30.0
}

# --------------------------------------------------------------------------
# Pipeline / Metadata
# --------------------------------------------------------------------------
PIPELINE_VERSION = "2.0.0"
