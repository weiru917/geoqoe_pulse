"""
GeoQoE Pulse Data & GIS Pipeline — Orchestrator.

    Raw Data
        v
    H3 Study Grid (real, trimmed Klang-Valley-core boundary — see config.py)
        v
    Real Spatial Joins (deduplicated towers, buildings, Ookla, OSM, WorldPop, DOSM, ridership)
        v
    Temporal Feature Engineering (leakage-free lags/rolling stats)
        v
    Spatial Feature Engineering (real libpysal weights, spatial lags/variance)
        v
    Demand Pressure Index (real static features; historical-QoE component
                            flagged as leakage-risk — see config.HIGH_LEAKAGE_RISK_FEATURES)
        v
    Hotspot Detection (real esda Getis-Ord Gi* + Local Moran's I)
        v
    QoE Target Generation (unified thresholds; SAME-HOUR + 1-HOUR-AHEAD forecast targets)
        v
    Validation (incl. target-consistency and score-range regression guards)
        v
    Export ML-ready datasets + honest static baseline + grid geometry + metadata

IMPORTANT: cell_time_dataset.parquet's hourly QoE columns are a SIMULATED
diurnal-demand scenario (see config.SIMULATION_DISCLAIMER) — there is no
per-hour QoE measurement source in this project. For a modelling claim that
doesn't depend on simulated data, use cell_static_qoe_baseline.parquet
instead (real Ookla-anchored labels only, real static GIS features, zero
fabricated values).

Exported files:
    data/processed/grid_geometry.parquet / .geojson
    data/outputs/cell_time_dataset.parquet          (SIMULATED hourly panel — demo/dashboard fixture)
    data/outputs/cell_static_features.parquet
    data/outputs/cell_static_qoe_baseline.parquet   (REAL, non-simulated — primary modelling deliverable)
    data/outputs/qoe_targets.parquet
    data/outputs/feature_dictionary.xlsx
    data/outputs/metadata.json
"""
import logging
import time

import numpy as np
import pandas as pd

from config import (
    GLOBAL_RANDOM_SEED, PROCESSED_DIR, OUTPUT_DIR, BASE_DIR, H3_PRIMARY_RESOLUTION,
    FORECAST_HORIZON_HOURS, HIGH_LEAKAGE_RISK_FEATURES, PIPELINE_VERSION,
    SIMULATION_DISCLAIMER,
)
from src.data_preprocessing.geometry_utils import generate_h3_study_grid, export_grid_geometry
from src.data_preprocessing.data_downloader import (
    build_static_features, load_ookla, build_dynamic_panel, build_static_qoe_baseline,
)
from src.data_preprocessing.feature_engineer import (
    compute_temporal_shifted_features, compute_demand_pressure_base,
    apply_dynamic_demand_adjustments, finalize_demand_pressure_index,
)
from src.data_preprocessing.spatial_processor import (
    build_h3_spatial_weights, compute_vectorized_spatial_lags, compute_spatial_rolling_mean,
    compute_hotspots, compute_local_moran, compute_historical_hotspot_frequency,
)
from src.data_preprocessing.qoe_target import compute_qoe_targets, compute_next_hour_targets
from src.data_preprocessing.data_validator import run_full_validation
from src.data_preprocessing.metadata_utils import generate_metadata, summarize_synthetic_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

# --------------------------------------------------------------------------
# Feature-exclusion sets — TWO modes, because "what's safe to use as a
# feature" depends on which target you're predicting. This replaces the
# single NON_FEATURE_COLS set from the previous version, which only ever
# supported same-hour prediction and didn't exclude spatial_lag_dl_median/
# spatial_lag_rtt_median (both direct functions of the same-hour target
# inputs) at all.
# --------------------------------------------------------------------------
STRUCTURAL_COLS = {"h3_index", "timestamp", "row_id"}

# Constant / zero-information columns — never useful as a feature regardless
# of target mode.
CONSTANT_COLS = {"qoe_hourly_shape_is_simulated"}

RAW_QOE_OBSERVATION_COLS = {
    "dl_median", "dl_mean", "dl_std", "dl_p95",
    "ul_median", "ul_mean", "ul_std",
    "rtt_median", "rtt_mean", "rtt_std",
    "jitter_median", "loss_median",
}
# Same-hour derivatives of the raw QoE columns that a naive user would NOT
# think to exclude just by reading "exclude dl_median/rtt_median" — flagged
# explicitly here because they were the review's leakage finding.
SAME_HOUR_QOE_DERIVED_COLS = {"spatial_lag_dl_median", "spatial_lag_rtt_median"}

# Ordinal encodings for categorical string columns that would otherwise
# crash tree-based learners (XGBoost, LightGBM) at fit time. The string
# originals are kept in the exported panel for dashboard human-readable
# labels; the _encoded integer versions are what the ML feature lists
# include. Ordering is deliberately semantic (cold → not_significant → hot;
# LL → LH → HL → HH) so ordinal encoding preserves meaningful ordering.
CATEGORICAL_ENCODINGS = {
    "hotspot_category_rtt_median": {
        "cold_99": 0, "cold_95": 1, "cold_90": 2, "not_significant": 3,
        "hot_90": 4, "hot_95": 5, "hot_99": 6,
    },
    "local_moran_quadrant_rtt_median": {
        "not_significant": 0, "LL": 1, "LH": 2, "HL": 3, "HH": 4,
    },
}
# Raw string columns excluded from ML features — use the _encoded versions.
STRING_CATEGORICAL_COLS = set(CATEGORICAL_ENCODINGS.keys())

# Priority-score components are post-processing for the dashboard / ranking
# pipeline, not independent predictors — including them would just add
# multicollinearity with their real-feature inputs (population_day,
# historical_hotspot_freq, qoe_confidence_score).
PRIORITY_SCORE_COLS = {"priority_exposure_score", "priority_base_score"}


def _target_cols(df: pd.DataFrame) -> set:
    return {c for c in df.columns if c.startswith("target_")}


def get_feature_columns(df: pd.DataFrame, target_mode: str) -> list:
    """
    target_mode = "current"   -> predicting target_is_degraded/_class/_index
                                  for THIS hour. Must exclude every raw QoE
                                  column AND their same-hour derivatives
                                  (spatial lags), or the model trivially
                                  reconstructs the label from a same-hour
                                  proxy (e.g. dl_mean = dl_median * 1.05).
    target_mode = "next_hour" -> predicting target_is_degraded_next_1h (the
                                  recommended MVP target). Current-hour QoE
                                  values and their spatial lags ARE legitimate
                                  features here, because they are genuinely
                                  observed before the hour being predicted.
    """
    if target_mode not in ("current", "next_hour"):
        raise ValueError('target_mode must be "current" or "next_hour"')

    exclude = set(STRUCTURAL_COLS) | set(CONSTANT_COLS) | _target_cols(df) | STRING_CATEGORICAL_COLS | PRIORITY_SCORE_COLS
    if target_mode == "current":
        exclude |= RAW_QOE_OBSERVATION_COLS | SAME_HOUR_QOE_DERIVED_COLS

    return [c for c in df.columns if c not in exclude]


COLUMN_DESCRIPTIONS = {
    "h3_index": "Uber H3 res-8 cell identifier (join key).",
    "timestamp": "Hourly timestamp (UTC-naive, local Klang Valley convention).",
    "row_id": "Surrogate unique row identifier (h3_index + timestamp composite, as int).",
    "dl_median": "Median simulated download throughput (Mbps) for this cell-hour. NOT a measurement — see config.SIMULATION_DISCLAIMER.",
    "dl_mean": "dl_median x 1.05 (formula-derived, not an independent statistic). Excluded from features when predicting the SAME-HOUR target (leakage); allowed for next-hour prediction.",
    "dl_std": "dl_median x 0.15 (formula-derived). Same leakage caveat as dl_mean.",
    "dl_p95": "dl_median x 1.30 (formula-derived). Same leakage caveat as dl_mean.",
    "ul_median": "Median simulated upload throughput (Mbps) for this cell-hour. Simulated, see above.",
    "ul_mean": "ul_median x 1.05 (formula-derived).",
    "ul_std": "ul_median x 0.15 (formula-derived).",
    "rtt_median": "Median simulated round-trip latency (ms) for this cell-hour. Simulated, see above.",
    "rtt_mean": "rtt_median x 1.05 (formula-derived).",
    "rtt_std": "rtt_median x 0.20 (formula-derived).",
    "jitter_median": "Simulated jitter (ms), derived from rtt_median plus noise.",
    "loss_median": "Simulated packet loss percentage, derived from the same load_factor as dl/rtt.",
    "test_sample_count": "Simulated HOURLY-equivalent test count. For Ookla-anchored cells this is Poisson-sampled around an hourly rate derived from the REAL quarterly total (ookla_quarter_test_count / ~2160 hours, scaled by demand), NOT the quarterly total itself repeated per hour (that was a bug — see CHANGELOG). Represents measurement VOLUME, not the outcome — safe as a feature in either target mode.",
    "unique_device_count": "~70% of test_sample_count (rounded). Same status as test_sample_count.",
    "ookla_quarter_test_count": "REAL Ookla quarterly test count for this cell (constant across all 168 simulated hours by definition — it is a QUARTERLY total, never a per-hour figure). NaN if the cell has no Ookla tile coverage.",
    "ookla_quarter_device_count": "REAL Ookla quarterly unique-device count for this cell (constant across all 168 simulated hours; quarterly, not hourly). NaN if unavailable in the source tile or no coverage exists.",
    "qoe_baseline_source": "Human-readable provenance string: 'ookla_q4_2024_static_anchor' or 'no_real_anchor_simulated_only'. Complements the qoe_level_is_ookla_anchored boolean.",
    "ridership_daily_surge_factor": "Real daily ridership-derived demand multiplier (normalized to the selected week's mean), summed only over Klang-Valley-relevant transit modes (national/inter-city services like ETS, Intercity, Komuter Utara, and Tebrau are excluded). Previously folded silently into the hourly formula with no standalone column; now exported directly so it can be inspected or used as its own feature.",
    "qoe_level_is_ookla_anchored": "True if a real Ookla quarterly average existed to re-centre this cell's simulated hourly values on (~41% of cells). False does not mean 'less real' shape — the hourly SHAPE is simulated either way.",
    "qoe_hourly_shape_is_simulated": "Always True. Included explicitly so this fact cannot be missed or confused with the anchored/unanchored distinction above. Excluded from ML features (constant, zero information).",
    "target_qoe_index": "Custom composite 0-100 QoE index (log-utility download/upload minus exponential latency/loss penalty). NOT a validated ITU-T G.107 MOS — see qoe_target.py docstring.",
    "target_service_quality_1to5": "1-5 linear rescaling of target_qoe_index. NOT a validated MOS/ACR score.",
    "target_is_degraded": "Binary SAME-HOUR degradation flag, derived from target_degradation_class >= 1 (single shared threshold source — config.QOE_THRESHOLDS — so this can no longer disagree with target_degradation_class).",
    "target_degradation_class": "0=normal, 1=degraded, 2=severe. Same threshold source as target_is_degraded.",
    f"target_is_degraded_next_{FORECAST_HORIZON_HOURS}h": "RECOMMENDED PRIMARY ML TARGET. target_is_degraded shifted -1 per cell (groupby h3_index).shift(-1) — genuinely predicting the future, not classifying the present. NaN/0-valid for each cell's final row (no future observation exists).",
    f"target_degradation_class_next_{FORECAST_HORIZON_HOURS}h": "3-class version of the next-hour forecast target.",
    f"target_qoe_index_next_{FORECAST_HORIZON_HOURS}h": "Continuous version of the next-hour forecast target.",
    f"target_next_{FORECAST_HORIZON_HOURS}h_valid": "1 if a real future observation existed to build the next-hour target from; 0 for each cell's final row. Drop rows where this is 0 before training — do not treat them as a 'not degraded' case.",
    "qoe_confidence_score": "min(1.0, test_sample_count / 10) — higher sample count = more confidence in the cell-hour's QoE estimate.",
    "demand_pressure_base": "0-100 composite of ONLY real static features (population, POI, built-up, roads/transport). Zero QoE dependency — safe to use with any target, any mode.",
    "demand_pressure_index_pre_spatial": f"demand_pressure_base plus a historical-QoE component. {HIGH_LEAKAGE_RISK_FEATURES.get('demand_pressure_index_pre_spatial', '')}",
    "demand_pressure_index": f"Final spatially-smoothed demand pressure index. {HIGH_LEAKAGE_RISK_FEATURES.get('demand_pressure_index', '')}",
    "population_night": "Real WorldPop 2020 zonal-statistics residential population estimate for the cell.",
    "population_day": "Modelled daytime population (residential population reweighted by real OSM office/retail density) — explicitly an assumption, not a raster.",
    "population_density_per_km2": "population_night / real cell area (km^2, projected CRS).",
    "built_up_ratio": "Real building footprint area / real H3 cell area (both from GIS geometry, not a hardcoded constant).",
    "tower_count": "Deduplicated approximate PHYSICAL SITE count (OpenCellID records collapsed by ~11m coordinate rounding) — see opencellid_record_count for the raw, non-deduplicated record count.",
    "opencellid_record_count": "Raw OpenCellID record count per cell BEFORE deduplication — includes multiple sectors/technologies/operators at the same physical site. Use tower_count for a physical-site estimate.",
    "tower_source": "Provenance string: 'opencellid_deduplicated' (real) or 'synthetic_fallback_no_file'.",
    "dist_to_nearest_tower_m": "Real geometric nearest-neighbor distance (sjoin_nearest, projected CRS) from the cell centroid to the nearest deduplicated OpenCellID site.",
    "building_count": "Real Google Open Buildings V3 footprint count per cell (confidence >= 0.65 only).",
    "total_building_area_m2": "Real summed footprint area (m^2) per cell.",
    "building_density_per_km2": "building_count / real cell area (km^2).",
    "avg_building_footprint_m2": "total_building_area_m2 / building_count.",
    "poi_education_count": "Real OSM POI count: schools, universities, colleges, kindergartens.",
    "poi_healthcare_count": "Real OSM POI count: hospitals, clinics, pharmacies, doctors.",
    "poi_retail_food_count": "Real OSM POI count: restaurants, cafes, fast food, shops.",
    "poi_office_count": "Real OSM POI count: tagged office=*.",
    "poi_recreation_count": "Real OSM POI count: leisure=* or tourism=*.",
    "poi_transport_stop_count": "Real OSM POI count of public-transport/rail stops joined into the cell.",
    "road_length_m": "Real road length (m) per cell via polygon overlay in a projected CRS (not a heuristic).",
    "road_density_km_per_km2": "road_length_m / 1000 / real cell area (km^2).",
    "intersection_count": "Real OSM road-network node count per cell.",
    "intersection_density_per_km2": "intersection_count / real cell area (km^2).",
    "dist_to_nearest_transit_m": "Real geometric nearest-neighbor distance to the nearest OSM transit/rail POI (sjoin_nearest, projected CRS).",
    "ookla_dl_mbps": "REAL Ookla tests-weighted mean download speed (Mbps) for the cell, static/quarterly. NaN where no tile coverage exists (not imputed).",
    "ookla_rtt_ms": "REAL Ookla tests-weighted mean latency (ms) for the cell, static/quarterly. NaN where no tile coverage exists.",
    "ookla_tests": "REAL summed Ookla test count backing ookla_dl_mbps/ookla_rtt_ms for the cell.",
    "ookla_is_synthetic": "True if the cell fell outside all Ookla tile coverage (no real anchor at all for this cell).",
    "has_real_ookla_label": "True if ookla_dl_mbps is non-null — i.e. this row has a REAL label usable for training in cell_static_qoe_baseline.parquet.",
    "qoe_risk_label": "high_risk/moderate_risk/low_risk from REAL ookla_dl_mbps thresholds. None where has_real_ookla_label is False (deliberately not guessed).",
    "gi_star_zscore_rtt_median": "Real Getis-Ord Gi* z-score for latency (esda.G_Local, star=True) — the full Gi* expectation/variance formulation, not a plain standardized neighbor average.",
    "hotspot_category_rtt_median": "Hot/cold spot classification at 90/95/99% confidence, from real esda.G_Local p-values.",
    "local_moran_quadrant_rtt_median": "HH/LH/LL/HL spatial cluster/outlier classification (real esda.Moran_Local).",
    "historical_hotspot_freq_rtt_median": "Rolling (shift(1)-based, leakage-free) fraction of past hours this cell was a real (esda-confirmed) hotspot.",
    "hotspot_category_rtt_median_encoded": "Ordinal-encoded hotspot confidence band (0=cold_99, 1=cold_95, 2=cold_90, 3=not_significant, 4=hot_90, 5=hot_95, 6=hot_99). ML-compatible integer version of hotspot_category_rtt_median (string, kept for dashboard).",
    "local_moran_quadrant_rtt_median_encoded": "Ordinal-encoded Local Moran's I spatial cluster/outlier class (0=not_significant, 1=LL, 2=LH, 3=HL, 4=HH). ML-compatible integer version of local_moran_quadrant_rtt_median (string, kept for dashboard).",
    "priority_exposure_score": "MinMax-scaled population_day floored at 0.01 — the population-exposure component of the Priority Score. Not an ML feature (derived from population_day).",
    "priority_base_score": "priority_exposure_score × historical_hotspot_freq × max(0.1, qoe_confidence_score) — pre-computed Priority Score base without model risk probability. Dashboard: final_priority = model_predicted_risk × priority_base_score. Not an ML feature.",
}


def _describe_column(col: str) -> str:
    if col in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[col]
    if col.endswith("_is_synthetic"):
        return f"True if {col.replace('_is_synthetic', '')} used a synthetic fallback (raw source file missing)."
    if col.startswith("lag_1h_"):
        return f"Value of {col[7:]} exactly 1 hour prior (shift(1), leakage-free)."
    if col.startswith("lag_2h_"):
        return f"Value of {col[7:]} exactly 2 hours prior (shift(2), leakage-free)."
    if col.startswith("lag_24h_"):
        return f"Value of {col[8:]} exactly 24 hours prior (shift(24), leakage-free)."
    if col.startswith("rolling_mean_"):
        return f"Trailing rolling mean of {col[len('rolling_mean_'):]}, computed after shift(1) (leakage-free)."
    if col.startswith("rolling_std_"):
        return f"Trailing rolling standard deviation of {col[len('rolling_std_'):]}, computed after shift(1) (leakage-free)."
    if col.startswith("spatial_lag_"):
        return f"Row-standardized H3 1-ring neighbor average of {col[12:]} at the same timestamp (real libpysal weights)."
    if col.startswith("spatial_var_"):
        return f"Row-standardized H3 1-ring neighbor variance of {col[12:]} at the same timestamp (real libpysal weights)."
    if col.startswith("gi_star_pvalue_"):
        return f"Permutation-based p-value (99 permutations) for the Getis-Ord Gi* statistic on {col[len('gi_star_pvalue_'):]}."
    if col.startswith("local_moran_i_"):
        return f"Local Moran's I statistic value for {col[len('local_moran_i_'):]} (real esda.Moran_Local)."
    if col.startswith("local_moran_pvalue_"):
        return f"Permutation-based p-value (99 permutations) for Local Moran's I on {col[len('local_moran_pvalue_'):]}."
    if col.endswith("_valid"):
        return "1 if the corresponding lag/rolling/forecast column is backed by real history/future, 0 otherwise (warm-up or end-of-series). Filter to 1 before training on the corresponding column."
    return "Undocumented — flag for the Data/GIS lead rather than guessing; do not assume this column is safe to use as-is."


def build_feature_dictionary(panel_df: pd.DataFrame, static_df: pd.DataFrame,
                              baseline_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tables = (
        ("cell_time_dataset", panel_df),
        ("cell_static_features", static_df),
        ("cell_static_qoe_baseline", baseline_df),
        ("qoe_targets", targets_df),
    )
    for name, df in tables:
        current_features = set(get_feature_columns(df, "current")) if "timestamp" in df.columns else set(df.columns)
        next_hour_features = set(get_feature_columns(df, "next_hour")) if "timestamp" in df.columns else set(df.columns)
        for col in df.columns:
            rows.append({
                "file": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "pct_non_null": round(100 * df[col].notna().mean(), 2),
                "is_feature_for_current_target": col in current_features,
                "is_feature_for_next_hour_target": col in next_hour_features,
                "leakage_risk_note": HIGH_LEAKAGE_RISK_FEATURES.get(col, ""),
                "description": _describe_column(col),
            })
    return pd.DataFrame(rows)


def main(days: int = 7):
    t_start = time.time()
    np.random.seed(GLOBAL_RANDOM_SEED)  # single global seed, set once, before ANY randomness
    logger.info("=== GeoQoE Pulse Data & GIS Pipeline v%s ===", PIPELINE_VERSION)
    logger.info(SIMULATION_DISCLAIMER)

    # --- 1. Real H3 study grid -------------------------------------------------
    logger.info("[1/10] Generating H3 study grid from real, trimmed study boundary...")
    grid_gdf = generate_h3_study_grid(res=H3_PRIMARY_RESOLUTION)
    export_grid_geometry(grid_gdf, PROCESSED_DIR)

    # --- 2. Real spatial joins ---------------------------------------------------
    logger.info("[2/10] Real spatial joins: deduplicated towers, buildings, OSM, WorldPop, DOSM...")
    static_df = build_static_features(grid_gdf)

    logger.info("[2b/10] Real Ookla spatial join (static per-cell baseline)...")
    ookla_df = load_ookla(grid_gdf)

    logger.info("[2c/10] Building the HONEST static QoE baseline (real Ookla labels, no fabrication)...")
    baseline_df = build_static_qoe_baseline(static_df, ookla_df)

    logger.info("[2d/10] Assembling SIMULATED hourly panel (%d days) for dashboard/demo use...", days)
    panel_df = build_dynamic_panel(grid_gdf, static_df, ookla_df, days=days)

    # --- 3. Temporal feature engineering (leakage-free) --------------------------
    logger.info("[3/10] Temporal features (leakage-free lags/rolling stats)...")
    panel_df = compute_temporal_shifted_features(panel_df)

    # Bring static features into the panel so per-timestamp spatial lags of
    # e.g. population are possible, and so the exported panel is self-contained.
    static_cols_for_panel = [c for c in static_df.columns if c != "h3_index"]
    panel_df = panel_df.merge(static_df[["h3_index"] + static_cols_for_panel], on="h3_index", how="left")

    # --- 4. Demand pressure index (real features; historical-QoE term flagged) ---
    logger.info("[4/10] Demand pressure index (real static features + flagged historical-QoE term)...")
    demand_base_df = compute_demand_pressure_base(static_df)
    panel_df = apply_dynamic_demand_adjustments(panel_df, demand_base_df)
    static_df = static_df.merge(demand_base_df, on="h3_index", how="left")  # also export in static features

    # --- 5. Spatial weights + spatial features ------------------------------------
    logger.info("[5/10] Building real libpysal spatial weights (H3 1-ring adjacency)...")
    w, h3_order = build_h3_spatial_weights(grid_gdf["h3_index"])

    logger.info("[5b/10] Spatial lags, spatial variance, spatial rolling mean...")
    panel_df = compute_vectorized_spatial_lags(
        panel_df, w, h3_order,
        target_cols=["dl_median", "rtt_median", "population_night", "demand_pressure_index_pre_spatial"],
    )
    panel_df = finalize_demand_pressure_index(panel_df)
    panel_df = compute_spatial_rolling_mean(panel_df, w, h3_order, col="dl_median", window=24)

    # --- 6. Hotspot detection (real esda implementations) -------------------------
    logger.info("[6/10] Getis-Ord Gi* + Local Moran's I (esda, real statistics with p-values)...")
    panel_df = compute_hotspots(panel_df, w, h3_order, col="rtt_median", permutations=99)
    panel_df = compute_local_moran(panel_df, w, h3_order, col="rtt_median", permutations=99)
    panel_df = compute_historical_hotspot_frequency(panel_df, col="rtt_median", window=168)

    # --- 6b. Encode categorical string columns for ML compatibility -----------
    logger.info("[6b/10] Encoding categorical columns (ordinal integers for ML, strings kept for dashboard)...")
    for _cat_col, _cat_mapping in CATEGORICAL_ENCODINGS.items():
        if _cat_col in panel_df.columns:
            panel_df[f"{_cat_col}_encoded"] = panel_df[_cat_col].map(_cat_mapping).fillna(3).astype(int)

    # --- 7. QoE targets: same-hour AND next-hour forecast ---------------------------
    logger.info("[7/10] QoE targets (unified thresholds; same-hour + %dh-ahead forecast)...", FORECAST_HORIZON_HOURS)
    panel_df = compute_qoe_targets(panel_df)
    panel_df = compute_next_hour_targets(panel_df, horizon_hours=FORECAST_HORIZON_HOURS)

    # --- 7b. Pre-compute priority score components ----------------------------
    logger.info("[7b/10] Pre-computing priority score components (exposure × recurrence × confidence)...")
    _pop = panel_df["population_day"].copy()
    _pmin, _pmax = _pop.min(), _pop.max()
    panel_df["priority_exposure_score"] = np.clip(
        (_pop - _pmin) / max(_pmax - _pmin, 1e-9), 0.01, 1.0
    )
    panel_df["priority_base_score"] = (
        panel_df["priority_exposure_score"]
        * panel_df["historical_hotspot_freq_rtt_median"].fillna(0.0)
        * panel_df["qoe_confidence_score"].clip(lower=0.1)
    )

    panel_df = panel_df.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)
    panel_df.insert(0, "row_id", panel_df.index)

    suffix = f"next_{FORECAST_HORIZON_HOURS}h"
    targets_df = panel_df[[
        "row_id", "h3_index", "timestamp",
        "target_qoe_index", "target_service_quality_1to5", "target_is_degraded", "target_degradation_class",
        f"target_is_degraded_{suffix}", f"target_degradation_class_{suffix}", f"target_qoe_index_{suffix}",
        f"target_{suffix}_valid",
        "qoe_confidence_score",
    ]].copy()

    # --- 8. Validation ----------------------------------------------------------------
    logger.info("[8/10] Running automated data validation (incl. target-consistency guards)...")
    report = run_full_validation(grid_gdf, static_df, panel_df, targets_df)
    if report.warnings:
        logger.warning("Validation passed with %d warning(s) — see log above for details.", len(report.warnings))

    # --- 9. Feature dictionary --------------------------------------------------------
    logger.info("[9/10] Building feature dictionary (per-target-mode feature flags + leakage notes)...")
    feature_dict_df = build_feature_dictionary(panel_df, static_df, baseline_df, targets_df)

    # --- 10. Export ---------------------------------------------------------------------
    logger.info("[10/10] Exporting ML-ready datasets, feature dictionary, and metadata...")
    panel_path = OUTPUT_DIR / "cell_time_dataset.parquet"
    static_path = OUTPUT_DIR / "cell_static_features.parquet"
    baseline_path = OUTPUT_DIR / "cell_static_qoe_baseline.parquet"
    targets_path = OUTPUT_DIR / "qoe_targets.parquet"

    panel_df.to_parquet(panel_path, index=False, compression="zstd")
    static_df.to_parquet(static_path, index=False, compression="zstd")
    baseline_df.to_parquet(baseline_path, index=False, compression="zstd")
    targets_df.to_parquet(targets_path, index=False, compression="zstd")
    feature_dict_df.to_excel(OUTPUT_DIR / "feature_dictionary.xlsx", index=False)

    synthetic_summary = summarize_synthetic_fraction(static_df, panel_df)
    generate_metadata(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        feature_columns_current=get_feature_columns(panel_df, "current"),
        feature_columns_next_hour=get_feature_columns(panel_df, "next_hour"),
        row_counts={"grid_cells": len(grid_gdf), "static_rows": len(static_df),
                    "panel_rows": len(panel_df), "target_rows": len(targets_df),
                    "static_qoe_baseline_rows": len(baseline_df),
                    "static_qoe_baseline_real_labels": int(baseline_df["has_real_ookla_label"].sum())},
        synthetic_summary=synthetic_summary,
    )

    elapsed = time.time() - t_start
    logger.info(
        "=== Pipeline complete in %.1fs — %d cells x %d hours = %d panel rows (SIMULATED), "
        "%d cells in the static QoE baseline (%d with a REAL Ookla label). "
        "Synthetic-fallback fractions: %s ===",
        elapsed, len(grid_gdf), len(panel_df) // len(grid_gdf), len(panel_df),
        len(baseline_df), int(baseline_df["has_real_ookla_label"].sum()), synthetic_summary,
    )
    return grid_gdf, static_df, panel_df, baseline_df, targets_df


if __name__ == "__main__":
    main()
