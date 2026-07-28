"""
GeoQoE Pulse Configuration File
Central configuration for CRS, H3 resolution, spatial bounds, and feature weights.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_DIR = DATA_DIR / "outputs"

# Ensure directories exist
for folder in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Spatial Projections
CRS_WGS84 = "EPSG:4326"
CRS_PENINSULAR_MYS = "EPSG:3375"  # GDM2000 / Peninsular Malaysia RSO
CRS_EAST_MYS = "EPSG:3376"        # GDM2000 / East Malaysia BRSO

# Primary Study Area: Kuala Lumpur / Klang Valley Bounding Box
STUDY_BBOX = {
    "min_lon": 101.50,
    "min_lat": 3.00,
    "max_lon": 101.80,
    "max_lat": 3.30
}

# H3 Hexagonal Grid Resolutions
H3_PRIMARY_RESOLUTION = 8   # ~0.74 km² cell area, ~461m edge
H3_SECONDARY_RESOLUTION = 9 # ~0.11 km² cell area, ~174m edge

# Temporal Configurations
TIME_FREQ = "1h"            # 1-hour windows
ROLLING_WINDOW_24H = 24     # 24 steps for 1-hour frequency
ROLLING_WINDOW_7D = 168     # 168 steps for 7-day frequency

# Demand Proxy Weights
DEMAND_WEIGHTS = {
    "population": 0.30,
    "poi": 0.25,
    "built_area": 0.20,
    "transport": 0.15,
    "peak_hour": 0.10
}

# QoE Thresholds for Classification
QOE_THRESHOLDS = {
    "dl_degraded_mbps": 10.0,
    "rtt_degraded_ms": 100.0,
    "dl_excellent_mbps": 50.0,
    "rtt_excellent_ms": 30.0
}
