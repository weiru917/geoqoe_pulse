"""
GeoQoE Pulse Metadata Module.

Generates metadata.json documenting pipeline version, generation timestamp,
CRS, source file versions/hashes, feature list, and git commit
"""
import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    CRS_WGS84, CRS_PENINSULAR_MYS, H3_PRIMARY_RESOLUTION, GLOBAL_RANDOM_SEED,
    PIPELINE_VERSION, RAW_DATASETS, RAW_DATASET_DIRS, DERIVED_LAYERS,
    SOURCE_VINTAGES, STUDY_BOUNDARY_PATH, STUDY_AREA_LABEL,
    SIMULATION_DISCLAIMER, PANEL_COMPONENT_PROVENANCE,
    PANEL_STRUCTURE_DISCLOSURE, HIGH_LEAKAGE_RISK_FEATURES, PRIMARY_ML_TARGET,
    PRIMARY_ML_TABLE, PANEL_DEMO_TARGET, FORECAST_HORIZON_HOURS,
)

logger = logging.getLogger(__name__)


def _git_hash(base_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(base_dir),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unavailable (not a git repository or git not installed)"


def _file_fingerprint(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    stat = path.stat()
    # Hash only the first 1MB for large raw files — enough to catch an
    # accidental swap/re-download without paying the cost of hashing
    # multi-GB files on every pipeline run.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_000_000))
    return {
        "present": True,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256_first_1mb": h.hexdigest(),
    }


def _dir_fingerprint(path: Path) -> dict:
    """Fingerprint a directory of raster tiles/granules: file count and
    total bytes. Hashing multi-GB satellite files on every run isn't worth
    the time; count+size still catches an accidental swap or a partial
    re-download."""
    if not path.exists() or not path.is_dir():
        return {"present": False}
    files = [f for f in path.rglob("*") if f.is_file()]
    return {
        "present": True,
        "n_files": len(files),
        "total_bytes": sum(f.stat().st_size for f in files),
        "filenames_sample": sorted(f.name for f in files)[:6],
    }


def generate_metadata(base_dir: Path, output_dir: Path,
                       feature_columns_current: list, feature_columns_next_hour: list,
                       row_counts: dict, synthetic_summary: dict,
                       fitted_scalers: dict = None, coverage_bias: dict = None,
                       application_profiles: dict = None) -> dict:
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_hash(base_dir),
        "random_seed": GLOBAL_RANDOM_SEED,
        "crs": {
            "storage_export": CRS_WGS84,
            "distance_area_computation": CRS_PENINSULAR_MYS,
        },
        "h3_resolution": H3_PRIMARY_RESOLUTION,
        "study_area": STUDY_AREA_LABEL,
        "study_boundary_source": (
            str(STUDY_BOUNDARY_PATH.name) if STUDY_BOUNDARY_PATH.exists()
            else "FALLBACK: rectangular STUDY_BBOX (real boundary file missing)"
        ),
        "row_counts": row_counts,
        "source_datasets": {
            **{
                name: {
                    "expected_filename": path.name,
                    "vintage": SOURCE_VINTAGES.get(name, "unrecorded"),
                    **_file_fingerprint(path),
                }
                for name, path in RAW_DATASETS.items()
            },
            **{
                name: {
                    "expected_directory": path.name,
                    "vintage": SOURCE_VINTAGES.get(name, "unrecorded"),
                    **_dir_fingerprint(path),
                }
                for name, path in RAW_DATASET_DIRS.items()
            },
        },
        "derived_layers": {
            name: {"path": str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else str(path),
                   **_file_fingerprint(path)}
            for name, path in DERIVED_LAYERS.items()
        },
        "source_vintages_note": (
            "cell_time_dataset.parquet's timestamps are a synthetic calendar "
            "window chosen so the panel overlaps the ridership file's real "
            "coverage. They are NOT the dates the underlying measurements "
            "were taken — see the per-source `vintage` fields above, which "
            "span 2020 (population) to 2026 (OSM/ridership)."
        ),
        "synthetic_fallback_summary": synthetic_summary,
        "primary_ml_target": PRIMARY_ML_TARGET,
        "primary_ml_table": PRIMARY_ML_TABLE,
        "panel_demo_target": PANEL_DEMO_TARGET,
        "panel_demo_target_warning": (
            "Do NOT report forecast skill on panel_demo_target. The hourly "
            "panel is rank-one for Ookla-anchored cells, so that label is "
            "reconstructable from (cell, hour-of-week) by memorisation. "
            "Benchmark on primary_ml_target in primary_ml_table instead."
        ),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "feature_columns": {
            "for_current_target": sorted(feature_columns_current),
            "for_next_hour_target": sorted(feature_columns_next_hour),
            "note": (
                "Two different lists because same-hour QoE columns and their "
                "spatial lags are leakage when predicting the SAME hour's "
                "label, but legitimate features when predicting NEXT hour's "
                "label. See run_pipeline.get_feature_columns()."
            ),
        },
        "fitted_scalers": {
            "note": (
                "Min-max scalers are fitted on the earliest 80% of the panel's timestamps only, "
                "not on all rows, so a temporal-holdout model's scaler has not seen the test "
                "period. Values outside the fitted range are clipped to [0,1]. These parameters "
                "are recorded because a scaler is part of the data contract, not an "
                "implementation detail — reproduce them exactly if you re-scale downstream."
            ),
            "scalers": fitted_scalers or {},
        },
        "coverage_bias": coverage_bias or {
            "status": "not computed (scikit-learn unavailable or baseline missing)",
        },
        "application_profiles": application_profiles or {"status": "not computed"},
        "high_leakage_risk_features": HIGH_LEAKAGE_RISK_FEATURES,
        "simulation_disclosure": {
            "explanation": SIMULATION_DISCLAIMER,
            "per_component_provenance": PANEL_COMPONENT_PROVENANCE,
            "real_ookla_anchored_cell_fraction": synthetic_summary.get("qoe_level_ookla_anchored"),
            "recommended_defensible_dataset": "cell_static_qoe_baseline.parquet (real Ookla labels only, zero simulated values)",
        },
        "panel_structure_disclosure": {
            "structure": "rank_one_separable",
            "explanation": PANEL_STRUCTURE_DISCLOSURE,
            "verified_reconstruction": (
                "dl/rtt for Ookla-anchored cells reconstruct from "
                "ookla_static x mlab_curve with max absolute error 0.0, and the "
                "reconstructed degradation target matches the stored target on "
                "100% of anchored rows."
            ),
        },
        "known_limitations": [
            "No per-CELL per-hour QoE measurement source exists for this study area. "
            "cell_time_dataset.parquet combines a REAL citywide hourly shape (M-Lab NDT, converted "
            "from UTC to Asia/Kuala_Lumpur at load) with a REAL per-cell level (Ookla quarterly) "
            "where tile coverage exists, and a formula-derived level where it does not. Jitter, "
            "packet loss and upload remain simulated. Because ONE shape is shared by every cell, "
            "the anchored panel is rank-one — see panel_structure_disclosure before reporting any "
            "forecast metric from it.",
            "The M-Lab curve is a single CITY-WIDE curve and was not ASN-filtered, so it may mix "
            "fixed-broadband and mobile tests. Per-district curves (which would break the rank-one "
            "structure) require re-querying M-Lab grouped by client geolocation.",
            "No real network capacity/traffic/utilisation data exists. demand_pressure_* features "
            "are exposure/activity PROXIES, not a measurement of actual load vs. capacity — do not "
            "claim 'confirmed congestion', only 'QoE degradation associated with estimated demand pressure'.",
            f"Study area is {STUDY_AREA_LABEL}",
            "On the SIMULATED hourly panel (cell_time_dataset.parquet), positive "
            "target_is_degraded_next_1h labels are concentrated in Ookla-anchored cells — see "
            "run_full_validation's validate_target_anchor_correlation warnings for the exact current "
            "split. qoe_level_is_ookla_anchored can be a near-trivial predictor of the target on this "
            "simulated panel; report metrics split by anchor status, or restrict training/eval to "
            "anchored cells, rather than trusting one aggregate PR-AUC.",
            "tower_count is a deduplicated approximation of physical sites (rounding-based clustering of "
            "OpenCellID records), not an authoritative operator site inventory.",
            "target_qoe_index / target_service_quality_1to5 are a custom composite index, not a validated "
            "ITU-T G.107 E-model MOS.",
            "h3_parent_res6 (res-6 H3 parent, ~130 groups over the current grid) is provided on every "
            "output table for SPATIAL HOLDOUT splits — group by it and hold out whole groups; do not "
            "randomly sample individual h3_index rows for a test set, spatially neighbouring res-8 "
            "cells are autocorrelated and a random split will leak.",
        ],
        "pandas_version": pd.__version__,
    }

    out_path = output_dir / "metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Wrote metadata -> %s", out_path.name)
    return metadata


def summarize_synthetic_fraction(*dataframes) -> dict:
    """Scan any number of dataframes for `*_is_synthetic` columns and report
    the fraction of rows/cells relying on a synthetic fallback per source —
    the single-source-of-truth number for "how much of this is real data".

    The QoE panel is handled separately and explicitly rather than through
    the generic `*_is_synthetic` pattern, because its components differ in
    provenance: the hourly SHAPE is real (M-Lab), the LEVEL is real only
    for Ookla-anchored cells, and jitter/loss/upload are simulated
    throughout. Each is reported under its own key so nobody mistakes
    "anchored" for "measured" or "real shape" for "real per-cell data"."""
    summary = {}
    for df in dataframes:
        for col in df.columns:
            if col.endswith("_is_synthetic"):
                source = col.replace("_is_synthetic", "")
                summary[source] = float(df[col].mean())
        if "qoe_jitter_loss_is_simulated" in df.columns:
            summary["qoe_jitter_loss_simulated"] = float(df["qoe_jitter_loss_is_simulated"].mean())
        if "qoe_hourly_shape_source" in df.columns:
            sources = sorted(set(df["qoe_hourly_shape_source"].dropna().unique()))
            summary["qoe_hourly_shape_sources"] = sources
        if "qoe_level_is_ookla_anchored" in df.columns:
            summary["qoe_level_ookla_anchored"] = float(df["qoe_level_is_ookla_anchored"].mean())
    return summary
