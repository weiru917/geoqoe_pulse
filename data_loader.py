"""
data_loader.py — GeoQoE Pulse Dashboard data layer.

Tries to load the REAL pipeline outputs from PIPELINE_OUTPUT_DIR (default:
"data/outputs", matching the geoqoe_pulse repo layout). If those parquet
files aren't found, it falls back to a seeded synthetic panel with the same
shape/columns, every value flagged is_synthetic=True, so the dashboard is
demoable before (or during) the pipeline run — same "synthetic transparency"
principle the pipeline itself uses.

COLUMN_MAP
----------
Names marked "confirmed" come directly from the README's model-training
snippet. Names marked "assumed" are best guesses at what feature_engineer.py
/ spatial_processor.py / qoe_target.py would call their outputs. If your real
feature_dictionary.xlsx uses different names, edit COLUMN_MAP only — nothing
else in this file or app.py needs to change.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. CONFIGURATION — edit this block to match your real pipeline outputs
# --------------------------------------------------------------------------

PIPELINE_OUTPUT_DIR = Path(os.environ.get("GEOQOE_OUTPUT_DIR", "data/outputs"))

CELL_TIME_FILE = PIPELINE_OUTPUT_DIR / "cell_time_dataset.parquet"
CELL_STATIC_FILE = PIPELINE_OUTPUT_DIR / "cell_static_features.parquet"
METADATA_FILE = PIPELINE_OUTPUT_DIR / "metadata.json"

COLUMN_MAP = {
    # -- confirmed, from README training snippet --
    "cell_id": "h3_index",
    "timestamp": "timestamp",
    "download_mbps": "dl_median",
    "latency_ms": "rtt_median",
    "qoe_score": "target_qoe_score",
    "mos": "target_mos",
    "is_degraded": "target_is_degraded",
    "degradation_class": "target_degradation_class",
    # -- assumed; rename here if feature_dictionary.xlsx differs --
    "upload_mbps": "ul_median",
    "jitter_ms": "jitter_median",
    "packet_loss_pct": "packet_loss_median",
    "demand_pressure": "demand_pressure_index",
    "gi_star_z": "gi_star_z",
    "gi_star_p": "gi_star_p",
    "gi_confidence_band": "gi_confidence_band",
    "moran_quadrant": "moran_quadrant",
    "hotspot_freq_hist": "hotspot_freq_historical",
    "likely_cause": "likely_cause",
    "population": "population",
    "poi_density": "poi_density",
    "built_up_ratio": "built_up_ratio",
    "road_density": "road_density_km_per_km2",
    "num_towers": "num_towers",
    "profile": "area_profile",
}

STUDY_AREA_NAME = "Klang Valley"
# Rough bounding box covering KL / PJ / Shah Alam / Klang / Subang, as
# (lat, lng) pairs — used only for the synthetic-data fallback grid.
BBOX_LAT_LON = [
    (2.85, 101.35),
    (2.85, 101.85),
    (3.35, 101.85),
    (3.35, 101.35),
]
H3_RESOLUTION = 8
MAX_SYNTHETIC_CELLS = 260
SYNTHETIC_DAYS = 7
GLOBAL_SEED = 42

AREA_PROFILES = ["residential", "office", "campus", "transit_hub", "mixed"]
PROFILE_WEIGHTS = [0.38, 0.22, 0.10, 0.10, 0.20]

CAUSE_LABELS = ["congestion", "indoor_attenuation", "coverage_gap", "backhaul", "none"]

# --------------------------------------------------------------------------
# 2. H3 compatibility shim (works with both h3 v3.x and v4.x APIs)
# --------------------------------------------------------------------------


def _h3_polygon_to_cells(bbox_lat_lon, resolution):
    import h3

    try:  # h3 >= 4
        poly = h3.LatLngPoly(bbox_lat_lon)
        return list(h3.polygon_to_cells(poly, resolution))
    except AttributeError:  # h3 3.x
        geojson = {
            "type": "Polygon",
            "coordinates": [[[lng, lat] for lat, lng in bbox_lat_lon]],
        }
        return list(h3.polyfill(geojson, resolution, geo_json_conformant=True))


def _h3_cell_to_latlng(cell):
    import h3

    try:  # h3 >= 4
        return h3.cell_to_latlng(cell)
    except AttributeError:  # h3 3.x
        return h3.h3_to_geo(cell)


# --------------------------------------------------------------------------
# 3. Synthetic data generation (seeded, clearly flagged)
# --------------------------------------------------------------------------


def _diurnal_factor(hour: int, profile: str) -> float:
    """0-1 demand shape by hour-of-day for a given area profile."""
    curves = {
        "residential": [0.55, 0.45, 0.35, 0.3, 0.3, 0.35, 0.5, 0.6, 0.55, 0.5,
                         0.5, 0.5, 0.55, 0.5, 0.5, 0.55, 0.65, 0.8, 0.95, 1.0,
                         0.95, 0.85, 0.75, 0.65],
        "office": [0.25, 0.2, 0.18, 0.15, 0.15, 0.2, 0.4, 0.7, 0.9, 1.0,
                   0.95, 0.9, 0.85, 0.9, 0.95, 0.9, 0.85, 0.6, 0.4, 0.3,
                   0.28, 0.27, 0.26, 0.25],
        "campus": [0.2, 0.15, 0.12, 0.1, 0.1, 0.15, 0.3, 0.55, 0.8, 0.95,
                   1.0, 0.9, 0.85, 0.95, 1.0, 0.9, 0.7, 0.5, 0.4, 0.35,
                   0.3, 0.28, 0.24, 0.22],
        "transit_hub": [0.2, 0.15, 0.12, 0.1, 0.12, 0.3, 0.7, 1.0, 0.9, 0.5,
                         0.4, 0.4, 0.45, 0.4, 0.4, 0.5, 0.75, 1.0, 0.95, 0.6,
                         0.45, 0.35, 0.28, 0.22],
        "mixed": [0.35, 0.3, 0.25, 0.22, 0.22, 0.28, 0.45, 0.65, 0.75, 0.8,
                  0.8, 0.78, 0.78, 0.78, 0.78, 0.78, 0.8, 0.85, 0.85, 0.8,
                  0.7, 0.6, 0.5, 0.4],
    }
    return curves[profile][hour]


def _generate_static(rng: np.random.Generator) -> pd.DataFrame:
    try:
        cells = _h3_polygon_to_cells(BBOX_LAT_LON, H3_RESOLUTION)
        if len(cells) > MAX_SYNTHETIC_CELLS:
            idx = rng.choice(len(cells), size=MAX_SYNTHETIC_CELLS, replace=False)
            cells = [cells[i] for i in sorted(idx)]
        coords = [_h3_cell_to_latlng(c) for c in cells]
        use_h3 = True
    except ImportError as exc:
        raise ImportError(
            "The 'h3' package is required to build the demo grid "
            "(pip install h3). See requirements.txt."
        ) from exc

    n = len(cells)
    profile = rng.choice(AREA_PROFILES, size=n, p=PROFILE_WEIGHTS)
    # ~8% of cells are seeded as chronic problem spots so the hotspot /
    # demand views have something meaningful to surface in the demo.
    is_chronic = rng.random(n) < 0.08

    population = np.clip(rng.lognormal(mean=6.8, sigma=0.9, size=n), 20, 15000).round()
    poi_density = np.clip(rng.lognormal(mean=1.8, sigma=1.0, size=n), 0, 400).round(1)
    built_up_ratio = np.clip(rng.beta(2, 3, size=n), 0.02, 0.98).round(3)
    road_density = np.clip(rng.normal(6, 2.5, size=n), 0.3, 18).round(2)
    num_towers = rng.poisson(2.2, size=n) + is_chronic.astype(int) * rng.integers(0, 2, size=n)

    df = pd.DataFrame({
        COLUMN_MAP["cell_id"]: cells,
        "lat": [c[0] for c in coords],
        "lon": [c[1] for c in coords],
        COLUMN_MAP["profile"]: profile,
        "is_chronic_hotspot": is_chronic,
        COLUMN_MAP["population"]: population,
        COLUMN_MAP["poi_density"]: poi_density,
        COLUMN_MAP["built_up_ratio"]: built_up_ratio,
        COLUMN_MAP["road_density"]: road_density,
        COLUMN_MAP["num_towers"]: num_towers,
    })
    for col in [COLUMN_MAP["population"], COLUMN_MAP["poi_density"],
                COLUMN_MAP["built_up_ratio"], COLUMN_MAP["road_density"],
                COLUMN_MAP["num_towers"]]:
        df[f"{col}_is_synthetic"] = True
    return df


def _generate_panel(static_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").floor("h")
    timestamps = pd.date_range(end=end, periods=SYNTHETIC_DAYS * 24, freq="h")

    rows = []
    for _, cell in static_df.iterrows():
        profile = cell[COLUMN_MAP["profile"]]
        chronic = bool(cell["is_chronic_hotspot"])
        base_noise = rng.normal(0, 1, size=len(timestamps))
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            weekday = ts.dayofweek
            weekend_damp = 0.7 if weekday >= 5 and profile in ("office", "campus") else 1.0
            demand = _diurnal_factor(hour, profile) * weekend_damp
            demand = np.clip(demand + base_noise[i] * 0.04, 0, 1.3)
            if chronic:
                demand = np.clip(demand * 1.35, 0, 1.5)
            demand_pressure = float(np.clip(demand * 100, 0, 100))

            dl = np.clip(rng.normal(45 - demand_pressure * 0.28, 6), 2, 220)
            ul = np.clip(dl * rng.uniform(0.08, 0.18), 0.5, 40)
            rtt = np.clip(rng.normal(28 + demand_pressure * 0.55, 5), 8, 400)
            jitter = np.clip(rng.normal(4 + demand_pressure * 0.06, 1.5), 0.5, 60)
            loss = np.clip(rng.normal(0.3 + demand_pressure * 0.02, 0.3), 0, 15)

            qoe = float(np.clip(100 - demand_pressure * 0.55 - max(0, rtt - 40) * 0.3
                                 - loss * 2, 0, 100))
            mos = round(1 + (qoe / 100) * 4, 2)
            degraded = qoe < 60
            severity = "severe" if qoe < 40 else ("degraded" if qoe < 60 else "normal")

            if not degraded:
                cause = "none"
            elif chronic and demand_pressure > 70:
                cause = "congestion"
            elif cell[COLUMN_MAP["built_up_ratio"]] > 0.6:
                cause = "indoor_attenuation"
            elif cell[COLUMN_MAP["num_towers"]] <= 1:
                cause = "coverage_gap"
            else:
                cause = rng.choice(["congestion", "backhaul", "indoor_attenuation"],
                                    p=[0.5, 0.25, 0.25])

            z = float(np.clip(rng.normal(3.0 if chronic else 0.1,
                                          1.0 if chronic else 1.2), -3.5, 6))
            p_val = math.erfc(abs(z) / math.sqrt(2))  # proper two-sided normal p-value
            if z > 2.58:
                band, quad = "99%", "HH"
            elif z > 1.96:
                band, quad = "95%", "HH"
            elif z > 1.65:
                band, quad = "90%", "HH"
            elif z < -1.96:
                band, quad = "95%", "LL"
            else:
                band, quad = "not significant", "NS"

            rows.append((
                cell[COLUMN_MAP["cell_id"]], ts, round(dl, 1), round(ul, 1),
                round(rtt, 1), round(jitter, 1), round(loss, 2),
                round(demand_pressure, 1), round(qoe, 1), mos, degraded,
                severity, cause, round(z, 2),
                round(float(np.clip(p_val, 0.0005, 0.99)), 4), band, quad,
            ))

    cols = [
        COLUMN_MAP["cell_id"], COLUMN_MAP["timestamp"], COLUMN_MAP["download_mbps"],
        COLUMN_MAP["upload_mbps"], COLUMN_MAP["latency_ms"], COLUMN_MAP["jitter_ms"],
        COLUMN_MAP["packet_loss_pct"], COLUMN_MAP["demand_pressure"],
        COLUMN_MAP["qoe_score"], COLUMN_MAP["mos"], COLUMN_MAP["is_degraded"],
        COLUMN_MAP["degradation_class"], COLUMN_MAP["likely_cause"],
        COLUMN_MAP["gi_star_z"], COLUMN_MAP["gi_star_p"],
        COLUMN_MAP["gi_confidence_band"], COLUMN_MAP["moran_quadrant"],
    ]
    panel = pd.DataFrame(rows, columns=cols)
    for col in [COLUMN_MAP["download_mbps"], COLUMN_MAP["latency_ms"],
                COLUMN_MAP["demand_pressure"], COLUMN_MAP["qoe_score"]]:
        panel[f"{col}_is_synthetic"] = True
    return panel


def _synthetic_metadata() -> dict:
    return {
        "pipeline_version": "synthetic-fallback",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "crs_storage": "EPSG:4326",
        "crs_metric": "EPSG:3375",
        "h3_resolution": H3_RESOLUTION,
        "global_seed": GLOBAL_SEED,
        "git_commit": None,
        "source_real_fraction": {
            "ookla_open_data": 0.0, "opencellid": 0.0, "google_open_buildings": 0.0,
            "openstreetmap": 0.0, "worldpop": 0.0, "opendosm": 0.0,
            "transit_ridership": 0.0,
        },
        "note": "No files found under PIPELINE_OUTPUT_DIR — showing seeded "
                "synthetic demo data instead of real pipeline output.",
    }


# --------------------------------------------------------------------------
# 4. Backfill — tolerate real pipeline files that lack an "assumed" column
# --------------------------------------------------------------------------


def _backfill_missing(df_static: pd.DataFrame, df_panel: pd.DataFrame):
    """Fill in any COLUMN_MAP column that isn't in the real parquet files with
    a clearly-labelled derived/default value, so the dashboard degrades
    gracefully instead of raising KeyError. Returns (df_static, df_panel,
    filled: list[str]) — `filled` lists what was auto-derived, shown in the
    sidebar so you know what to check against feature_dictionary.xlsx.
    """
    filled = []

    if COLUMN_MAP["cell_id"] not in df_static.columns:
        raise KeyError(
            f"'{COLUMN_MAP['cell_id']}' (cell_id) not found in cell_static_features.parquet. "
            "Fix COLUMN_MAP['cell_id'] in data_loader.py to match your real column name."
        )
    for hard_req in (COLUMN_MAP["cell_id"], COLUMN_MAP["timestamp"]):
        if hard_req not in df_panel.columns:
            raise KeyError(
                f"'{hard_req}' not found in cell_time_dataset.parquet. "
                "Fix COLUMN_MAP in data_loader.py to match your real column names."
            )

    # --- static features ---
    if "lat" not in df_static.columns or "lon" not in df_static.columns:
        latlon = df_static[COLUMN_MAP["cell_id"]].apply(_h3_cell_to_latlng)
        df_static["lat"] = latlon.apply(lambda t: t[0])
        df_static["lon"] = latlon.apply(lambda t: t[1])
        filled.append("lat/lon (derived from h3_index)")

    static_defaults = {
        "profile": "mixed",
        "population": np.nan,
        "poi_density": np.nan,
        "built_up_ratio": np.nan,
        "road_density": np.nan,
        "num_towers": np.nan,
    }
    for key, default in static_defaults.items():
        col = COLUMN_MAP[key]
        if col not in df_static.columns:
            df_static[col] = default
            filled.append(col)

    # --- panel / time-varying features ---
    if COLUMN_MAP["qoe_score"] not in df_panel.columns:
        raise KeyError(
            f"'{COLUMN_MAP['qoe_score']}' not found in cell_time_dataset.parquet. "
            "Fix COLUMN_MAP['qoe_score'] in data_loader.py."
        )

    if COLUMN_MAP["is_degraded"] not in df_panel.columns:
        df_panel[COLUMN_MAP["is_degraded"]] = df_panel[COLUMN_MAP["qoe_score"]] < 60
        filled.append(COLUMN_MAP["is_degraded"])

    if COLUMN_MAP["demand_pressure"] not in df_panel.columns:
        df_panel[COLUMN_MAP["demand_pressure"]] = (
            100 - df_panel[COLUMN_MAP["qoe_score"]]
        ).clip(0, 100)
        filled.append(f"{COLUMN_MAP['demand_pressure']} (proxy: 100 - QoE score)")

    if COLUMN_MAP["gi_star_z"] not in df_panel.columns:
        df_panel[COLUMN_MAP["gi_star_z"]] = 0.0
        filled.append(COLUMN_MAP["gi_star_z"])

    if COLUMN_MAP["gi_confidence_band"] not in df_panel.columns:
        df_panel[COLUMN_MAP["gi_confidence_band"]] = "not significant"
        filled.append(COLUMN_MAP["gi_confidence_band"])

    if COLUMN_MAP["moran_quadrant"] not in df_panel.columns:
        df_panel[COLUMN_MAP["moran_quadrant"]] = "NS"
        filled.append(COLUMN_MAP["moran_quadrant"])

    if COLUMN_MAP["likely_cause"] not in df_panel.columns:
        df_panel[COLUMN_MAP["likely_cause"]] = "unknown"
        filled.append(f"{COLUMN_MAP['likely_cause']} (defaulted to 'unknown')")

    return df_static, df_panel, filled


# --------------------------------------------------------------------------
# 5. Public loader
# --------------------------------------------------------------------------


def load_data():
    """Returns (df_static, df_panel, metadata, is_real_data: bool, filled: list[str])."""
    if CELL_TIME_FILE.exists() and CELL_STATIC_FILE.exists():
        df_panel = pd.read_parquet(CELL_TIME_FILE)
        df_static = pd.read_parquet(CELL_STATIC_FILE)
        metadata = {}
        if METADATA_FILE.exists():
            metadata = json.loads(METADATA_FILE.read_text())
        df_static, df_panel, filled = _backfill_missing(df_static, df_panel)
        return df_static, df_panel, metadata, True, filled

    rng = np.random.default_rng(GLOBAL_SEED)
    df_static = _generate_static(rng)
    df_panel = _generate_panel(df_static, rng)
    metadata = _synthetic_metadata()
    return df_static, df_panel, metadata, False, []