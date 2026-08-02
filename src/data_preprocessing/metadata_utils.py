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
    PIPELINE_VERSION, RAW_DATASETS, STUDY_BOUNDARY_PATH, STUDY_AREA_LABEL,
    SIMULATION_DISCLAIMER, HIGH_LEAKAGE_RISK_FEATURES, PRIMARY_ML_TARGET,
    FORECAST_HORIZON_HOURS,
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


def generate_metadata(base_dir: Path, output_dir: Path,
                       feature_columns_current: list, feature_columns_next_hour: list,
                       row_counts: dict, synthetic_summary: dict) -> dict:
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
            name: {"expected_filename": path.name, **_file_fingerprint(path)}
            for name, path in RAW_DATASETS.items()
        },
        "synthetic_fallback_summary": synthetic_summary,
        "primary_ml_target": PRIMARY_ML_TARGET,
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
        "high_leakage_risk_features": HIGH_LEAKAGE_RISK_FEATURES,
        "simulation_disclosure": {
            "panel_qoe_hourly_shape_is_simulated": True,
            "explanation": SIMULATION_DISCLAIMER,
            "real_ookla_anchored_cell_fraction": synthetic_summary.get("qoe_level_ookla_anchored"),
            "recommended_defensible_dataset": "cell_static_qoe_baseline.parquet (real Ookla labels only, zero simulated values)",
        },
        "known_limitations": [
            "No real per-hour/repeated-timestamp QoE measurement source exists in this project "
            "(no organiser data was supplied). cell_time_dataset.parquet's hourly QoE values are "
            "a simulated diurnal-demand scenario, useful for pipeline/dashboard testing only.",
            "No real network capacity/traffic/utilisation data exists. demand_pressure_* features "
            "are exposure/activity PROXIES, not a measurement of actual load vs. capacity — do not "
            "claim 'confirmed congestion', only 'QoE degradation associated with estimated demand pressure'.",
            f"Study area is {STUDY_AREA_LABEL} Kuala Lumpur itself is not covered by any layer in this "
            "pipeline (no KL boundary file was available) — every hotspot/demand result excludes the "
            "country's highest-demand urban core.",
            "tower_count is a deduplicated approximation of physical sites (rounding-based clustering of "
            "OpenCellID records), not an authoritative operator site inventory.",
            "target_qoe_index / target_service_quality_1to5 are a custom composite index, not a validated "
            "ITU-T G.107 E-model MOS.",
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
    the generic `*_is_synthetic` pattern: `qoe_hourly_shape_is_simulated` is
    always True (reported as such, not averaged away), and
    `qoe_level_is_ookla_anchored` is reported under its own key so nobody
    mistakes "anchored" for "measured"."""
    summary = {}
    for df in dataframes:
        for col in df.columns:
            if col.endswith("_is_synthetic"):
                source = col.replace("_is_synthetic", "")
                summary[source] = float(df[col].mean())
        if "qoe_hourly_shape_is_simulated" in df.columns:
            summary["qoe_hourly_shape_simulated"] = float(df["qoe_hourly_shape_is_simulated"].mean())
        if "qoe_level_is_ookla_anchored" in df.columns:
            summary["qoe_level_ookla_anchored"] = float(df["qoe_level_is_ookla_anchored"].mean())
    return summary
