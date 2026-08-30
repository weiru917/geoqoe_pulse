"""
Extract real terrain statistics per H3 cell from Copernicus DEM GLO-30.

NO ACCOUNT REQUIRED. The Copernicus 30 m DEM is served from a public,
unauthenticated S3 bucket, one COG per 1x1 degree tile. Four tiles cover
the Klang Valley study area (N02/N03 x E101/E102, ~160 MB total). This
script downloads any that are missing, then runs zonal statistics over
the H3 grid — the same operation scripts/extract/worldcover.py uses.

WHY TERRAIN MATTERS HERE: the proposal names terrain and obstruction as
context for coverage, and the pipeline had no elevation layer at all.
Ruggedness in particular distinguishes cells that are hard to serve for
physical reasons from cells that are merely busy — which is exactly the
persistent-limitation vs demand-pressure distinction the project is
trying to make.

IMPORTANT — THIS IS A SURFACE MODEL (DSM), NOT BARE EARTH (DTM).
Copernicus GLO-30 includes buildings and vegetation canopy. In dense
urban cells the values partly reflect built structures rather than
ground. That is fine, arguably better, for radio-obstruction reasoning,
but it means `elevation_mean` must NOT be read as ground height, and it
correlates with the building layers by construction. Flagged in the
output so the distinction travels with the data.

Run: python scripts/extract/dem.py
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
DEM_DIR = PROJECT_ROOT / "data" / "raw" / "copernicus_dem"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "dem_by_h3.parquet"

DEM_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
# 1x1 degree tiles covering the study bounds (lat 2.60-3.40, lon 101.20-101.97).
DEM_TILES = [
    ("N02", "E101"), ("N02", "E102"), ("N03", "E101"), ("N03", "E102"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def tile_name(lat: str, lon: str) -> str:
    return f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"


def download_tiles(dest_dir: Path = DEM_DIR, only=None) -> list:
    """Fetch any missing tiles. Public bucket — no credentials needed."""
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for lat, lon in (only if only is not None else DEM_TILES):
        name = tile_name(lat, lon)
        dest = dest_dir / f"{name}.tif"
        if dest.exists():
            logger.info("%s already present (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            paths.append(dest)
            continue
        url = f"{DEM_S3_BASE}/{name}/{name}.tif"
        logger.info("Downloading %s ...", name)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        logger.info("  %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        paths.append(dest)
    return paths


def main():
    if not GRID_PARQUET.exists():
        raise FileNotFoundError(f"{GRID_PARQUET} not found — run the pipeline's grid step first.")
    from rasterstats import zonal_stats

    grid = gpd.read_parquet(GRID_PARQUET)
    # Only fetch tiles that actually intersect the grid. The full 2x2 tile
    # block is 160 MB but the study bounds touch only the E101 column, so
    # the E102 tiles were 97 MB of pointless download.
    min_lon, min_lat, max_lon, max_lat = grid.total_bounds
    needed = [(la, lo) for la, lo in DEM_TILES
              if int(lo[1:]) <= max_lon and int(lo[1:]) + 1 > min_lon
              and int(la[1:]) <= max_lat and int(la[1:]) + 1 > min_lat]
    logger.info("Study bounds need %d of %d tiles: %s", len(needed), len(DEM_TILES),
                [f'{la}{lo}' for la, lo in needed])
    tiles = download_tiles(only=needed)

    # Select each cell's tile GEOGRAPHICALLY, not by pixel count.
    #
    # Bug this replaces: the Copernicus COGs declare `nodata=None`, so a
    # read outside a tile's footprint returns 0.0 AND counts those pixels.
    # Every cell therefore appeared to have ~901 valid pixels in every
    # tile, and a "whichever tile has the most pixels" rule picked an
    # arbitrary one — handing 2,562 cells a phantom elevation of exactly
    # 0.0 m. Matching the cell centroid against each tile's real bounds is
    # deterministic and cannot produce that failure.
    #
    # (This is a different failure mode from the WorldCover tile seam,
    # where tiles genuinely each held part of a cell. Here a cell belongs
    # to exactly one 1-degree tile, and the DEM's 30 m pixels are far
    # finer than an H3 res-8 cell, so no cell needs stitching.)
    import rasterio

    tile_bounds = {}
    for path in tiles:
        with rasterio.open(path) as src:
            tile_bounds[path] = src.bounds

    def tile_for(lon, lat):
        for path, b in tile_bounds.items():
            if b.left <= lon < b.right and b.bottom <= lat < b.top:
                return path
        return None

    grid = grid.copy()
    grid["_tile"] = [tile_for(lon, lat) for lon, lat
                     in zip(grid["centroid_lon"], grid["centroid_lat"])]
    unassigned = int(grid["_tile"].isna().sum())
    if unassigned:
        logger.warning("%d cell(s) fall outside every downloaded DEM tile — they get NaN.", unassigned)
    logger.info("Cells per tile: %s",
                {p.name[-22:-8]: int(n) for p, n in grid["_tile"].value_counts().items()})

    best = {}
    for path in tiles:
        subset = grid[grid["_tile"] == path]
        if subset.empty:
            continue
        logger.info("Zonal statistics against %s (%d cells) ...", path.name, len(subset))
        stats = zonal_stats(
            subset, str(path), stats=["mean", "min", "max", "std", "count"],
        )
        for (_, row), s in zip(subset.iterrows(), stats):
            if s and s.get("count"):
                best[row["h3_index"]] = s

    rows = []
    for _, row in grid.iterrows():
        h3_index = row["h3_index"]
        s = best.get(h3_index)
        rows.append({
            "h3_index": h3_index,
            "elevation_mean_m": s["mean"] if s else np.nan,
            "elevation_min_m": s["min"] if s else np.nan,
            "elevation_max_m": s["max"] if s else np.nan,
            # Standard deviation of surface height within the cell — the
            # ruggedness measure. High values mean broken terrain or tall
            # structures, both of which obstruct propagation.
            "terrain_ruggedness_m": s["std"] if s else np.nan,
            "elevation_range_m": (s["max"] - s["min"]) if s else np.nan,
            "dem_pixel_count": int(s["count"]) if s else 0,
        })

    out = pd.DataFrame(rows)
    out["dem_is_surface_model"] = True
    out["dem_source"] = "REAL (Copernicus DEM GLO-30, DSM — includes buildings and canopy)"

    covered = int((out["dem_pixel_count"] > 0).sum())
    logger.info("DEM coverage: %d of %d cells (%.1f%%).", covered, len(out), 100 * covered / len(out))
    if covered < len(out):
        logger.warning(
            "%d cell(s) have no DEM pixels — they get NaN, never an imputed elevation.",
            len(out) - covered,
        )
    logger.info(
        "Elevation: median %.1f m, range %.1f-%.1f m. Ruggedness: median %.1f m.",
        out["elevation_mean_m"].median(), out["elevation_mean_m"].min(),
        out["elevation_mean_m"].max(), out["terrain_ruggedness_m"].median(),
    )

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Saved -> %s", OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
