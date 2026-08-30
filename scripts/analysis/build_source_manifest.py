"""
Record of exactly which source files were used, for provenance.

The 14 raw Ookla quarterly downloads (~2.75 GB) are deleted after their
per-cell extraction by design — keeping them would bloat the repo for no
analytical gain, since the extraction is deterministic and the files are
re-fetchable from a public S3 path. This manifest is the evidence trail
that replaces them: which quarters were ingested, how many real tiles and
tests each contributed, and the exact URL to re-fetch any of them.
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OUTPUT_DIR, PROCESSED_DIR, RAW_DATASETS, RAW_DATASET_DIRS, SOURCE_VINTAGES
from src.data_preprocessing.ookla_quarterly import quarter_url

OUT = OUTPUT_DIR / "source_manifest.json"


def main():
    q = pd.read_parquet(PROCESSED_DIR / "ookla_quarterly_by_h3.parquet")
    per_q = q.groupby(["quarter_label", "year", "quarter"]).agg(
        cells_covered=("h3_index", "nunique"),
        tiles=("ookla_tile_count", "sum"),
        real_tests=("ookla_tests", "sum"),
    ).reset_index().sort_values(["year", "quarter"])

    quarters = [{
        "quarter": r.quarter_label,
        "cells_covered": int(r.cells_covered),
        "ookla_tiles": int(r.tiles),
        "real_speed_tests": int(r.real_tests),
        "source_url": quarter_url(int(r.year), int(r.quarter)),
    } for r in per_q.itertuples()]

    manifest = {
        "purpose": "Provenance record of every source actually used, including files "
                    "deleted after extraction.",
        "ookla_quarterly": {
            "quarters_ingested": len(quarters),
            "span": f"{quarters[0]['quarter']} to {quarters[-1]['quarter']}",
            "total_real_speed_tests": int(per_q.real_tests.sum()),
            "distinct_cells_ever_covered": int(q.h3_index.nunique()),
            "raw_files_retained": False,
            "why_deleted": "~2.75 GB of raw tiles for a few hundred KB of per-cell extract. "
                            "Deterministic and re-fetchable from the URLs below.",
            "extract_retained_at": "data/processed/ookla_quarterly_by_h3.parquet",
            "quarters": quarters,
        },
        "other_sources": {
            name: {"file": path.name, "present": path.exists(),
                    "vintage": SOURCE_VINTAGES.get(name, "unrecorded")}
            for name, path in RAW_DATASETS.items()
        },
        "source_directories": {
            name: {"directory": path.name, "present": path.exists(),
                    "n_files": len(list(path.rglob("*"))) if path.exists() else 0,
                    "vintage": SOURCE_VINTAGES.get(name, "unrecorded")}
            for name, path in RAW_DATASET_DIRS.items()
        },
    }
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=2)

    o = manifest["ookla_quarterly"]
    print(f"Ookla: {o['quarters_ingested']} quarters, {o['span']}, "
          f"{o['total_real_speed_tests']:,} real speed tests, "
          f"{o['distinct_cells_ever_covered']:,} distinct cells")
    print(f"\n{'quarter':10s} {'cells':>7s} {'tiles':>8s} {'real tests':>12s}")
    for r in quarters:
        print(f"{r['quarter']:10s} {r['cells_covered']:7d} {r['ookla_tiles']:8d} {r['real_speed_tests']:12,d}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
