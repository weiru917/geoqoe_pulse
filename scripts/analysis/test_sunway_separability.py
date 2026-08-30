"""
Test the separability assumption underlying cell_time_dataset.parquet.

THE ASSUMPTION
--------------
The hourly panel is built as  value(cell, hour) = level(cell) x shape(hour).
That is a downscaling method, and it rests on one testable claim:

    Within an area, all cells share the same RELATIVE diurnal shape,
    and differ only in level.

If that holds, applying one shape to many cells is a defensible
approximation. If it does not, the panel is smoothing away exactly the
spatial-temporal interaction the project claims to detect.

WHAT THIS SCRIPT CAN AND CANNOT TEST
------------------------------------
The Sunway field campaign (Mendeley dx5xyyfz2y) is the only place in the
study area with real, timestamped, per-cell measurements. It covers 10
adjacent H3 res-8 cells. Its limits are hard:

  - Hours present: 09:00-16:00 ONLY, concentrated 11:00-14:00.
    -> CANNOT test the evening peak (19:00-00:00), which is precisely
       where the M-Lab curve says the real trough is.
  - 18 distinct calendar days over Dec 2024 - Apr 2025.
  - All 10 cells are in ONE small area (~6 km^2), so this tests
    WITHIN-area separability. It says nothing about whether the CBD and
    a distant suburb share a shape — which is the assumption that
    actually matters for a citywide curve.

So a positive result here is weak supporting evidence for a narrow claim,
not a validation of the panel. A negative result is strong evidence
against the assumption, because failing even within one small area is
worse than failing across the city.

METRIC CHOICE
-------------
throughput_dl_mbps is a passive continuous log dominated by idle
moments, so its per-cell-hour MEDIAN is ~0 and useless as a shape. This
script uses the per-cell-hour MEAN throughput (an activity//load proxy)
and, separately, median RSRP (radio conditions). Neither is a speed test.
Both are real measurements, reported for what they are.

Run: python scripts/test_sunway_separability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUTPUT_DIR, PROCESSED_DIR

MIN_RECORDS_PER_CELL_HOUR = 100
MIN_HOURS_PER_CELL = 5
OUT_PATH = OUTPUT_DIR / "separability_test.json"


def build_profiles(points: pd.DataFrame, value_col: str, agg: str) -> pd.DataFrame:
    """Per (cell, hour) aggregate, kept only where enough real records back it."""
    points = points.copy()
    points["hour"] = points["timestamp"].dt.hour

    grouped = points.groupby(["h3_index", "hour"]).agg(
        value=(value_col, agg), n=(value_col, "size"),
    ).reset_index()
    grouped = grouped[grouped["n"] >= MIN_RECORDS_PER_CELL_HOUR]

    wide = grouped.pivot(index="h3_index", columns="hour", values="value")
    # Keep cells observed across enough hours for a shape to mean anything.
    wide = wide[wide.notna().sum(axis=1) >= MIN_HOURS_PER_CELL]
    return wide


def normalise_shape(wide: pd.DataFrame) -> pd.DataFrame:
    """Divide each cell's profile by its own mean -> pure SHAPE, level removed.
    This is exactly the operation the panel assumes is shared."""
    return wide.div(wide.mean(axis=1), axis=0)


def analyse(wide: pd.DataFrame, label: str) -> dict:
    if wide.empty or len(wide) < 2:
        return {"metric": label, "status": "INSUFFICIENT_DATA",
                "n_cells": int(len(wide))}

    shapes = normalise_shape(wide)
    shared = shapes.mean(axis=0)  # the common shape the panel would apply

    # How much of each cell's variation does the shared shape explain?
    # R^2 of cell_shape ~ shared_shape, computed only on hours both have.
    r2_per_cell, corr_per_cell = {}, {}
    for cell, row in shapes.iterrows():
        both = row.notna() & shared.notna()
        if both.sum() < MIN_HOURS_PER_CELL:
            continue
        y, f = row[both].values, shared[both].values
        ss_res = float(((y - f) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2_per_cell[cell] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if np.std(y) > 0 and np.std(f) > 0:
            corr_per_cell[cell] = float(np.corrcoef(y, f)[0, 1])

    r2 = pd.Series(r2_per_cell).dropna()
    corr = pd.Series(corr_per_cell).dropna()

    # Spread of the cells around the shared shape, per hour. If separability
    # holds this is small; if cells genuinely differ in shape it is large.
    dispersion = shapes.std(axis=0) / shapes.mean(axis=0).abs()

    return {
        "metric": label,
        "status": "COMPLETED",
        "n_cells": int(len(shapes)),
        "hours_covered": [int(h) for h in sorted(shapes.columns)],
        "shared_shape_by_hour": {int(h): round(float(v), 4) for h, v in shared.dropna().items()},
        "per_cell_r2_vs_shared_shape": {
            "median": round(float(r2.median()), 4) if len(r2) else None,
            "min": round(float(r2.min()), 4) if len(r2) else None,
            "max": round(float(r2.max()), 4) if len(r2) else None,
            "n_cells_scored": int(len(r2)),
            "n_cells_negative_r2": int((r2 < 0).sum()),
        },
        "per_cell_correlation_vs_shared_shape": {
            "median": round(float(corr.median()), 4) if len(corr) else None,
            "min": round(float(corr.min()), 4) if len(corr) else None,
            "max": round(float(corr.max()), 4) if len(corr) else None,
        },
        "relative_dispersion_across_cells_by_hour": {
            int(h): round(float(v), 4) for h, v in dispersion.dropna().items()
        },
        "median_relative_dispersion": round(float(dispersion.median()), 4),
    }


def verdict(result: dict) -> str:
    if result.get("status") != "COMPLETED":
        return "NOT_TESTABLE"
    r2 = result["per_cell_r2_vs_shared_shape"]["median"]
    disp = result["median_relative_dispersion"]
    if r2 is None:
        return "NOT_TESTABLE"
    if r2 >= 0.5 and disp <= 0.30:
        return "SUPPORTED_WITHIN_DAYTIME_WINDOW"
    if r2 >= 0.0 and disp <= 0.60:
        return "WEAK_PARTIAL_SUPPORT"
    return "NOT_SUPPORTED"


def main():
    points_path = PROCESSED_DIR / "sunway_real_points.parquet"
    if not points_path.exists():
        raise FileNotFoundError(
            f"{points_path} not found — run "
            "`python -m src.data_preprocessing.ingest_sunway_real` first."
        )
    points = pd.read_parquet(points_path)

    results = {
        "what_this_tests": (
            "Whether cells within one small area share a common RELATIVE diurnal shape, "
            "which is the assumption behind building the hourly panel as "
            "level(cell) x shape(hour) with a single shared shape."
        ),
        "hard_limits": {
            "hours_available": "09:00-16:00 only, concentrated 11:00-14:00",
            "cannot_test": (
                "The 19:00-00:00 evening trough the M-Lab curve identifies as this city's "
                "real congestion window. No Sunway data exists after 16:00."
            ),
            "spatial_scope": (
                "All 10 cells lie within ~6 km^2. This tests WITHIN-area separability only. "
                "It does not test whether the CBD and a distant suburb share a shape — the "
                "assumption that actually matters for a citywide curve."
            ),
            "calendar_days": int(points["timestamp"].dt.date.nunique()),
            "min_records_per_cell_hour": MIN_RECORDS_PER_CELL_HOUR,
        },
    }

    for value_col, agg, label in [
        ("throughput_dl_mbps", "mean", "mean_passive_downlink_throughput"),
        ("rsrp_dbm", "median", "median_rsrp_dbm"),
    ]:
        wide = build_profiles(points, value_col, agg)
        res = analyse(wide, label)
        res["verdict"] = verdict(res)
        results[label] = res
        print(f"\n=== {label} ===")
        print(f"  cells with a usable profile : {res.get('n_cells')}")
        if res.get("status") == "COMPLETED":
            print(f"  hours covered               : {res['hours_covered']}")
            print(f"  median per-cell R^2 vs shared shape : {res['per_cell_r2_vs_shared_shape']['median']}")
            print(f"  range                              : {res['per_cell_r2_vs_shared_shape']['min']} .. {res['per_cell_r2_vs_shared_shape']['max']}")
            print(f"  cells with NEGATIVE R^2            : {res['per_cell_r2_vs_shared_shape']['n_cells_negative_r2']}")
            print(f"  median relative dispersion         : {res['median_relative_dispersion']}")
        print(f"  VERDICT                     : {res['verdict']}")

    results["overall_interpretation"] = (
        "Read the verdicts as evidence about a NARROW claim only. A supported result means "
        "cells in one small area share a daytime shape; it does not license applying one "
        "curve citywide, and it says nothing about the evening peak, which this data cannot "
        "reach. A NOT_SUPPORTED result is strong evidence against the panel's construction, "
        "because failing within a single 6 km^2 area is worse than failing across the city."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
