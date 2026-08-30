"""
Extract real OSM building-height / storey information per H3 cell.

NO DOWNLOAD REQUIRED — this reads the Geofabrik PBF already in
data/raw/. Vertical density is a better indoor-demand proxy than
footprint area alone: two cells with identical built-up ratio differ
enormously in population and traffic if one is warehouses and the other
is 30-storey towers.

HONESTY: OSM height tagging is voluntary and sparse in Malaysia. This
script MEASURES the fill rate and refuses to emit a feature that is too
sparse to be meaningful (see MIN_TAGGED_FRACTION) rather than shipping a
column that is 98% NaN and inviting a model to read noise. Whatever it
finds is reported honestly, including "not usable".

Run: python scripts/extract/building_heights.py
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Repo root on the path so these standalone scripts can import config.
sys.path.insert(0, str(PROJECT_ROOT))
GRID_PARQUET = PROJECT_ROOT / "data" / "processed" / "grid_geometry.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "building_heights_by_h3.parquet"
OSM_PBF = PROJECT_ROOT / "data" / "raw" / "malaysia-singapore-brunei-260727.osm.pbf"

# Below this fraction of buildings carrying usable height information, the
# derived features are emitted as all-NaN with a flag rather than as a
# feature. A column populated for 2% of buildings is not a height feature,
# it is a sampling artefact of which mapper visited where.
MIN_TAGGED_FRACTION = 0.02

# Typical storey height (m) for converting building:levels to metres.
METRES_PER_LEVEL = 3.2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_height(value):
    """OSM `height` is free text: '12', '12 m', '12.5m'. Return metres or NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower().replace("m", "").replace("meters", "").strip()
    try:
        h = float(text)
    except ValueError:
        return np.nan
    # Implausible values are mis-tags, not buildings.
    return h if 1.5 <= h <= 900.0 else np.nan


def _parse_levels(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        lv = float(str(value).strip())
    except ValueError:
        return np.nan
    return lv if 1 <= lv <= 200 else np.nan


def main():
    if not GRID_PARQUET.exists():
        raise FileNotFoundError(f"{GRID_PARQUET} not found — run the pipeline's grid step first.")
    if not OSM_PBF.exists():
        raise FileNotFoundError(f"{OSM_PBF} not found.")

    import pyrosm

    grid = gpd.read_parquet(GRID_PARQUET)
    min_lon, min_lat, max_lon, max_lat = grid.total_bounds
    pad = 0.02
    logger.info("Reading buildings from the existing PBF over the study bounds...")
    osm = pyrosm.OSM(str(OSM_PBF),
                     bounding_box=[min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad])
    buildings = osm.get_buildings()
    if buildings is None or buildings.empty:
        raise ValueError("No buildings returned from the PBF over the study bounds.")
    logger.info("  %d buildings in the study bounds", len(buildings))

    b = buildings.copy()
    b["_levels"] = b["building:levels"].map(_parse_levels) if "building:levels" in b.columns else np.nan
    b["_height_m"] = b["height"].map(_parse_height) if "height" in b.columns else np.nan
    # Prefer an explicit height; fall back to levels x typical storey height.
    b["height_m"] = b["_height_m"].fillna(b["_levels"] * METRES_PER_LEVEL)

    n_total = len(b)
    n_levels = int(b["_levels"].notna().sum())
    n_height = int(b["_height_m"].notna().sum())
    n_any = int(b["height_m"].notna().sum())
    frac = n_any / max(n_total, 1)

    logger.info(
        "Height tagging in the study area: building:levels on %d (%.2f%%), height on %d (%.2f%%), "
        "either on %d (%.2f%%) of %d buildings.",
        n_levels, 100 * n_levels / n_total, n_height, 100 * n_height / n_total,
        n_any, 100 * frac, n_total,
    )

    usable = frac >= MIN_TAGGED_FRACTION
    if not usable:
        logger.warning(
            "Only %.2f%% of buildings carry usable height information, below the %.0f%% threshold. "
            "The per-cell columns will be emitted as NaN with "
            "building_height_is_usable=False. A column this sparse reflects which mapper visited "
            "where, not how tall the buildings are — do not model on it.",
            100 * frac, 100 * MIN_TAGGED_FRACTION,
        )

    tagged = b[b["height_m"].notna()].copy()
    if usable and not tagged.empty:
        if tagged.crs is None:
            tagged = tagged.set_crs(grid.crs)
        # Centroids MUST be computed in a projected CRS. Taking a centroid
        # in WGS84 degrees is the same discipline error the pipeline audits
        # for elsewhere; at building scale the displacement is small, but it
        # is still wrong and can push a footprint across a cell boundary.
        from config import CRS_PENINSULAR_MYS, CRS_WGS84
        tagged["geometry"] = (
            tagged.geometry.to_crs(CRS_PENINSULAR_MYS).centroid.to_crs(CRS_WGS84)
        )
        tagged = tagged.set_geometry("geometry")
        joined = gpd.sjoin(tagged[["height_m", "geometry"]],
                            grid[["h3_index", "geometry"]], how="inner", predicate="within")
        agg = joined.groupby("h3_index").agg(
            building_height_mean_m=("height_m", "mean"),
            building_height_max_m=("height_m", "max"),
            building_height_p90_m=("height_m", lambda s: s.quantile(0.90)),
            building_n_with_height=("height_m", "size"),
        ).reset_index()
    else:
        agg = pd.DataFrame(columns=["h3_index", "building_height_mean_m", "building_height_max_m",
                                     "building_height_p90_m", "building_n_with_height"])

    out = grid[["h3_index"]].merge(agg, on="h3_index", how="left")
    out["building_n_with_height"] = out["building_n_with_height"].fillna(0).astype(int)
    out["building_height_is_usable"] = bool(usable)
    out["building_height_tagged_fraction"] = float(frac)
    # Selection-bias warning that travels with the data. OSM height tags are
    # concentrated where mappers are active — city centre, malls, landmarks —
    # which is the same kind of non-random coverage that makes Ookla labels
    # biased (see coverage_bias.py). A model can therefore read this column
    # as "is this a well-mapped area" rather than "how tall is it". Weight or
    # interpret accordingly, and never impute the NaNs.
    out["building_height_coverage_is_biased"] = True
    out["building_height_source"] = (
        f"REAL (OpenStreetMap building:levels/height tags, {100 * frac:.2f}% of buildings tagged; "
        f"levels converted at {METRES_PER_LEVEL} m/storey)"
    )

    n_cells = int((out["building_n_with_height"] > 0).sum())
    logger.info("Per-cell coverage: %d of %d cells have at least one height-tagged building (%.1f%%).",
                n_cells, len(out), 100 * n_cells / len(out))

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Saved -> %s", OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
