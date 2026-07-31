"""
GeoQoE Pulse Data & GIS Pipeline — Orchestrator.

    Raw Data
        v
    H3 Study Grid (real boundary polygon, real geometry)
        v
    Real Spatial Joins (towers, buildings, Ookla, OSM, WorldPop, DOSM, ridership)
        v
    Temporal Feature Engineering (leakage-free lags/rolling stats)
        v
    Spatial Feature Engineering (real libpysal weights, spatial lags/variance)
        v
    Demand Pressure Index (real static features + historical QoE + nearby demand)
        v
    Hotspot Detection (real esda Getis-Ord Gi* + Local Moran's I)
        v
    QoE Target Generation
        v
    Validation
        v
    Export ML-ready datasets + grid geometry + metadata

Exported files (unchanged filenames/schema-compatible with the modelling
team's existing workflow, per the brief):
    data/processed/grid_geometry.parquet / .geojson   
    data/outputs/cell_time_dataset.parquet
    data/outputs/cell_static_features.parquet
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
)
from src.data_preprocessing.geometry_utils import generate_h3_study_grid, export_grid_geometry
from src.data_preprocessing.data_downloader import build_static_features, load_ookla, build_dynamic_panel
from src.data_preprocessing.feature_engineer import (
    compute_temporal_shifted_features, compute_demand_pressure_base,
    apply_dynamic_demand_adjustments, finalize_demand_pressure_index,
)
from src.data_preprocessing.spatial_processor import (
    build_h3_spatial_weights, compute_vectorized_spatial_lags, compute_spatial_rolling_mean,
    compute_hotspots, compute_local_moran, compute_historical_hotspot_frequency,
)
from src.data_preprocessing.qoe_target import compute_composite_qoe_targets
from src.data_preprocessing.data_validator import run_full_validation
from src.data_preprocessing.metadata_utils import generate_metadata, summarize_synthetic_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

NON_FEATURE_COLS = {
    "h3_index", "timestamp", "row_id",
    "target_qoe_score", "target_mos", "target_is_degraded", "target_degradation_class",
    "dl_median", "dl_mean", "dl_std", "dl_p95", "ul_median", "ul_mean", "ul_std",
    "rtt_median", "rtt_mean", "rtt_std", "jitter_median", "loss_median",
}

COLUMN_DESCRIPTIONS = {
    "h3_index": "Uber H3 res-8 cell identifier (join key).",
    "timestamp": "Hourly timestamp (UTC-naive, local Klang Valley convention).",
    "row_id": "Surrogate unique row identifier (h3_index + timestamp composite, as int).",
    "dl_median": "Median download throughput (Mbps) for this cell-hour.",
    "rtt_median": "Median round-trip latency (ms) for this cell-hour.",
    "qoe_is_synthetic": "True if this cell-hour's QoE values are model-generated rather than Ookla-anchored.",
    "target_qoe_score": "Composite 0-100 QoE score (ITU-inspired).",
    "target_mos": "Mean Opinion Score estimate (1.0-5.0).",
    "target_is_degraded": "Binary degradation flag (dl below threshold OR rtt above threshold).",
    "target_degradation_class": "0=normal, 1=degraded, 2=severe.",
    "demand_pressure_index": "0-100 composite demand pressure (population, POI, built-up, roads, historical QoE, nearby demand).",
    "population_night": "WorldPop-derived residential population estimate for the cell.",
    "population_day": "Modelled daytime population (residential population reweighted by real OSM office/retail density).",
    "built_up_ratio": "Real building footprint area / real H3 cell area (both from GIS geometry, not a hardcoded constant).",
    "gi_star_zscore_rtt_median": "Getis-Ord Gi* z-score for latency (esda.G_Local, star=True).",
    "hotspot_category_rtt_median": "Hot/cold spot classification at 90/95/99% confidence (esda.G_Local p-values).",
    "local_moran_quadrant_rtt_median": "HH/LH/LL/HL spatial cluster/outlier classification (esda.Moran_Local).",
    "historical_hotspot_freq_rtt_median": "Rolling (shift(1)-based, leakage-free) fraction of past hours this cell was a real hotspot.",
}


def _describe_column(col: str) -> str:
    if col in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[col]
    if col.endswith("_is_synthetic"):
        return f"True if {col.replace('_is_synthetic', '')} used a synthetic fallback (raw source file missing)."
    if col.startswith("lag_1h_"):
        return f"Value of {col[7:]} exactly 1 hour prior (shift(1), leakage-free)."
    if col.startswith("lag_24h_"):
        return f"Value of {col[8:]} exactly 24 hours prior (shift(24), leakage-free)."
    if col.startswith("spatial_lag_"):
        return f"Row-standardized H3 1-ring neighbor average of {col[12:]} at the same timestamp."
    if col.endswith("_valid"):
        return "1 if the corresponding lag/rolling column is backed by real history, 0 during warm-up (NaN)."
    return "See src/data_preprocessing/ module docstrings."


def build_feature_dictionary(panel_df: pd.DataFrame, static_df: pd.DataFrame,
                              targets_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in (("cell_time_dataset", panel_df), ("cell_static_features", static_df), ("qoe_targets", targets_df)):
        for col in df.columns:
            rows.append({
                "file": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "pct_non_null": round(100 * df[col].notna().mean(), 2),
                "is_feature": col not in NON_FEATURE_COLS and not col.startswith("target_"),
                "description": _describe_column(col),
            })
    return pd.DataFrame(rows)


def main(days: int = 7):
    t_start = time.time()
    np.random.seed(GLOBAL_RANDOM_SEED)  # single global seed, set once, before ANY randomness
    logger.info("=== GeoQoE Pulse Data & GIS Pipeline v2.0.0 ===")

    # --- 1. Real H3 study grid -------------------------------------------------
    logger.info("[1/9] Generating H3 study grid from real study boundary...")
    grid_gdf = generate_h3_study_grid(res=H3_PRIMARY_RESOLUTION)
    export_grid_geometry(grid_gdf, PROCESSED_DIR)

    # --- 2. Real spatial joins ---------------------------------------------------
    logger.info("[2/9] Real spatial joins: towers, buildings, OSM, WorldPop, DOSM...")
    static_df = build_static_features(grid_gdf)

    logger.info("[2b/9] Real Ookla spatial join (static per-cell baseline)...")
    ookla_df = load_ookla(grid_gdf)

    logger.info("[2c/9] Assembling hourly panel (%d days)...", days)
    panel_df = build_dynamic_panel(grid_gdf, static_df, ookla_df, days=days)

    # --- 3. Temporal feature engineering (leakage-free) --------------------------
    logger.info("[3/9] Temporal features (leakage-free lags/rolling stats)...")
    panel_df = compute_temporal_shifted_features(panel_df)

    # Bring static features into the panel so per-timestamp spatial lags of
    # e.g. population are possible, and so the exported panel is self-contained.
    static_cols_for_panel = [c for c in static_df.columns if c != "h3_index"]
    panel_df = panel_df.merge(static_df[["h3_index"] + static_cols_for_panel], on="h3_index", how="left")

    # --- 4. Demand pressure index (real features, two-pass to fold in spatial term) ---
    logger.info("[4/9] Demand pressure index (real static features + historical QoE)...")
    demand_base_df = compute_demand_pressure_base(static_df)
    panel_df = apply_dynamic_demand_adjustments(panel_df, demand_base_df)

    # --- 5. Spatial weights + spatial features ------------------------------------
    logger.info("[5/9] Building real libpysal spatial weights (H3 1-ring adjacency)...")
    w, h3_order = build_h3_spatial_weights(grid_gdf["h3_index"])

    logger.info("[5b/9] Spatial lags, spatial variance, spatial rolling mean...")
    panel_df = compute_vectorized_spatial_lags(
        panel_df, w, h3_order,
        target_cols=["dl_median", "rtt_median", "population_night", "demand_pressure_index_pre_spatial"],
    )
    panel_df = finalize_demand_pressure_index(panel_df)
    panel_df = compute_spatial_rolling_mean(panel_df, w, h3_order, col="dl_median", window=24)

    # --- 6. Hotspot detection (real esda implementations) -------------------------
    logger.info("[6/9] Getis-Ord Gi* + Local Moran's I (esda, real statistics with p-values)...")
    panel_df = compute_hotspots(panel_df, w, h3_order, col="rtt_median", permutations=99)
    panel_df = compute_local_moran(panel_df, w, h3_order, col="rtt_median", permutations=99)
    panel_df = compute_historical_hotspot_frequency(panel_df, col="rtt_median", window=168)

    # --- 7. QoE targets -------------------------------------------------------------
    logger.info("[7/9] Composite QoE targets (score, MOS, degradation class)...")
    panel_df = compute_composite_qoe_targets(panel_df)

    panel_df = panel_df.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)
    panel_df.insert(0, "row_id", panel_df.index)

    targets_df = panel_df[[
        "row_id", "h3_index", "timestamp",
        "target_qoe_score", "target_mos", "target_is_degraded", "target_degradation_class",
        "qoe_confidence_score",
    ]].copy()

    # --- 8. Validation ----------------------------------------------------------------
    logger.info("[8/9] Running automated data validation...")
    report = run_full_validation(grid_gdf, static_df, panel_df, targets_df)
    if report.warnings:
        logger.warning("Validation passed with %d warning(s) — see log above for details.", len(report.warnings))

    # --- 9. Export ----------------------------------------------------------------------
    logger.info("[9/9] Exporting ML-ready datasets, feature dictionary, and metadata...")
    panel_path = OUTPUT_DIR / "cell_time_dataset.parquet"
    static_path = OUTPUT_DIR / "cell_static_features.parquet"
    targets_path = OUTPUT_DIR / "qoe_targets.parquet"

    panel_df.to_parquet(panel_path, index=False, compression="zstd")
    static_df.to_parquet(static_path, index=False, compression="zstd")
    targets_df.to_parquet(targets_path, index=False, compression="zstd")

    feature_dict_df = build_feature_dictionary(panel_df, static_df, targets_df)
    feature_dict_df.to_excel(OUTPUT_DIR / "feature_dictionary.xlsx", index=False)

    synthetic_summary = summarize_synthetic_fraction(static_df, panel_df)
    generate_metadata(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        feature_columns=[c for c in panel_df.columns if c not in NON_FEATURE_COLS and not c.startswith("target_")],
        row_counts={"grid_cells": len(grid_gdf), "static_rows": len(static_df),
                    "panel_rows": len(panel_df), "target_rows": len(targets_df)},
        synthetic_summary=synthetic_summary,
    )

    elapsed = time.time() - t_start
    logger.info(
        "=== Pipeline complete in %.1fs — %d cells x %d hours = %d panel rows. "
        "Synthetic-fallback fractions: %s ===",
        elapsed, len(grid_gdf), len(panel_df) // len(grid_gdf), len(panel_df), synthetic_summary,
    )
    return grid_gdf, static_df, panel_df, targets_df


if __name__ == "__main__":
    main()
