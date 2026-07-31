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
    PIPELINE_VERSION, RAW_DATASETS, STUDY_BOUNDARY_PATH,
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


def generate_metadata(base_dir: Path, output_dir: Path, feature_columns: list,
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
        "feature_columns": sorted(feature_columns),
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
    the single-source-of-truth number for "how much of this is real data"."""
    summary = {}
    for df in dataframes:
        for col in df.columns:
            if col.endswith("_is_synthetic"):
                source = col.replace("_is_synthetic", "")
                summary[source] = float(df[col].mean())
    return summary
