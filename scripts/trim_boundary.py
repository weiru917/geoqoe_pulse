"""
One-off, reproducible fix for the study-boundary scope bug.

The original data/boundary/study_boundary.geojson was a full-Selangor GADM
level-2 extract (9 districts) plus Putrajaya — labelled "Klang Valley" in the
README even though it included Hulu Selangor (rural interior) and Sabak
Bernam (rural north coast), neither of which is part of any common Klang
Valley definition, while it entirely EXCLUDED Kuala Lumpur (no KL boundary
file was available at build time).

This script re-applies that trim from a fresh/raw GADM Selangor extract, so
the fix is reproducible rather than a one-time hand-edit. It was already
applied once directly to the shipped data/boundary/study_boundary.geojson;
re-run this only if you re-download a raw extract from scratch.

Usage:
    python scripts/trim_boundary.py path/to/raw_selangor_gadm.geojson

If/when a real "W.P. Kuala Lumpur" GADM polygon becomes available, download
it, concatenate its single feature into the FeatureCollection this script
produces, and re-run the full pipeline (H3 grid + every spatial join
depends on the boundary) — see config.py's STUDY_BOUNDARY_PATH comment.
"""
import json
import sys
from pathlib import Path

from config import KLANG_VALLEY_DISTRICT_NAMES, STUDY_BOUNDARY_PATH


def trim(raw_path: Path, out_path: Path = STUDY_BOUNDARY_PATH) -> None:
    with open(raw_path) as f:
        data = json.load(f)

    before = len(data["features"])
    data["features"] = [
        feat for feat in data["features"]
        if feat["properties"].get("NAME_2") in KLANG_VALLEY_DISTRICT_NAMES
        or feat["properties"].get("NAME_1") in KLANG_VALLEY_DISTRICT_NAMES
    ]
    after = len(data["features"])

    with open(out_path, "w") as f:
        json.dump(data, f)

    kept = sorted(f["properties"].get("NAME_2", f["properties"].get("NAME_1")) for f in data["features"])
    print(f"{raw_path} -> {out_path}: {before} -> {after} features kept: {kept}")
    if after == before:
        print("WARNING: nothing was dropped — check that the raw file's NAME_2 column "
              "actually contains district names outside KLANG_VALLEY_DISTRICT_NAMES.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    trim(Path(sys.argv[1]))
