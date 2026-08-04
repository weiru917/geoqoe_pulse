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

UPDATE: a real "W.P. Kuala Lumpur" polygon (source: nullifye/
malaysia.geojson, state code KUL — state-level, since GADM has no level-2
subdivision for a federal territory) has since been merged into the shipped
data/boundary/study_boundary.geojson directly, alongside the Selangor
districts this script filters. If you re-run this script against a fresh
raw Selangor extract (e.g. NAME_1 == "Selangor" from scratch), it will
output ONLY the Selangor districts + Putrajaya again — you must
re-concatenate the KL feature back in afterwards (or merge from
data/boundary/study_boundary.geojson's existing KL row) before treating the
result as the real study boundary. See config.py's STUDY_BOUNDARY_PATH
comment for exactly how the KL polygon was sourced and validated.
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
