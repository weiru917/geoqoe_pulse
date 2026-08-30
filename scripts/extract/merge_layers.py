"""
Merge the VIIRS nighttime-lights and ESA WorldCover per-H3 extractions
into one file the main pipeline consumes.

Run this AFTER scripts/extract/viirs.py and scripts/extract/worldcover.py have both run.
run_pipeline.py picks the output up automatically (see
config.DERIVED_LAYERS["new_static_layers"]) and folds it into
cell_static_features.parquet BEFORE the static QoE baseline is built, so
these columns reach the modelling dataset, the hourly panel and the
feature dictionary — not just the static features file.

WHAT CHANGED HERE
-----------------
The previous version forwarded only 7 of ESA WorldCover's 11 classes,
silently dropping mangroves (mean 3.9% of grid area, 100% in some cells,
>10% in 305 cells) and wetland_herbaceous. Those are the real coastal
Klang / Kuala Langat cells. The effect downstream was that 480 cells had
land-cover percentages summing to <99%, and some summed to 0.00 — which
reads as "no land-cover data" when the truth is "100% mangrove". Coastal
mangrove is genuine signal for a network-planning model (low demand,
hard to serve), not noise. All 11 classes are now forwarded, along with
the coverage flag and pixel counts, and the sums are asserted.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Repo root on the path so these standalone scripts can import config.
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

VIIRS_PATH = PROCESSED_DIR / "viirs_by_h3.parquet"
WORLDCOVER_PATH = PROCESSED_DIR / "worldcover_by_h3.parquet"
OUTPUT_PATH = PROCESSED_DIR / "static_new_layers_by_h3.parquet"

LANDCOVER_CLASSES = [
    "tree_cover", "shrubland", "grassland", "cropland", "built_up",
    "bare_sparse_veg", "snow_ice", "water", "wetland_herbaceous",
    "mangroves", "moss_lichen",
]


def main():
    for path in (VIIRS_PATH, WORLDCOVER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run scripts/extract/viirs.py and "
                "scripts/extract/worldcover.py first."
            )

    viirs = pd.read_parquet(VIIRS_PATH)
    wc = pd.read_parquet(WORLDCOVER_PATH)

    pct_cols = [f"landcover_{name}_pct" for name in LANDCOVER_CLASSES]
    wc_cols = ["h3_index"] + pct_cols + [
        "worldcover_tile_covered", "worldcover_pixel_count",
        "worldcover_n_tiles", "worldcover_source",
    ]
    missing = [c for c in wc_cols if c not in wc.columns]
    if missing:
        raise ValueError(
            f"WorldCover extraction is missing columns {missing} — re-run "
            "scripts/extract/worldcover.py with the current version (it now retains "
            "all 11 classes and accumulates counts across tiles)."
        )

    # Recover the per-month VIIRS values. scripts/extract/viirs.py computes one
    # column per monthly composite and merge_new_layers previously forwarded
    # only their MEAN — the same silent-drop pattern that lost the mangrove
    # land-cover class. The 6 months cannot become a time-varying feature
    # (they span 3 of the panel's 14 quarters, and 6 months is too short a
    # baseline for real activity growth to exceed retrieval wobble), but
    # they do support a genuine STABILITY measure: a cell whose nighttime
    # lights fluctuate month to month behaves differently from one that is
    # steady. Both are computed on complete cases only — a cell missing a
    # month (cloud cover; Nov 2024 has 1,037 such cells) gets NaN rather
    # than a variability figure computed over a different set of months
    # than its neighbours.
    month_cols = sorted(c for c in viirs.columns
                        if c.startswith("viirs_avg_rad_") and c != "viirs_avg_rad_mean")
    # Requiring EVERY month was right when there were 6; with 43 monthly
    # composites over an equatorial study area, cloud cover means almost no
    # cell has a complete series and a complete-case rule would empty the
    # column. A minimum count keeps the statistic honest without throwing
    # the layer away — and viirs_n_months_observed travels with it so a
    # consumer can filter harder if they want.
    MIN_MONTHS = 12
    viirs = viirs.copy()
    n_obs = viirs[month_cols].notna().sum(axis=1)
    enough = n_obs >= min(MIN_MONTHS, len(month_cols))
    viirs["viirs_monthly_std"] = viirs[month_cols].std(axis=1).where(enough)
    viirs["viirs_monthly_cv"] = (
        viirs["viirs_monthly_std"] / viirs[month_cols].mean(axis=1).abs()
    ).where(enough)
    viirs["viirs_n_months_observed"] = n_obs
    viirs["viirs_months_covered"] = f"{len(month_cols)} monthly composites"

    merged = viirs[["h3_index", "viirs_avg_rad_mean", "viirs_monthly_std", "viirs_monthly_cv",
                    "viirs_n_months_observed", "viirs_months_covered", "viirs_source"]].merge(
        wc[wc_cols], on="h3_index", how="left",
    )

    if not merged["viirs_avg_rad_mean"].notna().all():
        raise ValueError("VIIRS coverage incomplete — check scripts/extract/viirs.py output")
    if not merged["landcover_built_up_pct"].notna().all():
        raise ValueError("WorldCover coverage incomplete — check scripts/extract/worldcover.py output")

    sums = merged.loc[merged["worldcover_tile_covered"], pct_cols].sum(axis=1)
    if not np.allclose(sums, 100.0, atol=0.01):
        raise ValueError(
            f"Land-cover percentages do not sum to 100 after the merge "
            f"(min {sums.min():.4f}, max {sums.max():.4f}). A class column was "
            "dropped — that is the exact bug this file was rewritten to fix."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {merged.shape} -> {OUTPUT_PATH}")
    print(f"  all {len(LANDCOVER_CLASSES)} land-cover classes retained; "
          f"every covered cell sums to 100%")
    print(f"  mangroves: mean {merged['landcover_mangroves_pct'].mean():.2f}%, "
          f"{int((merged['landcover_mangroves_pct'] > 10).sum())} cells above 10% "
          "(previously dropped entirely)")


if __name__ == "__main__":
    main()
