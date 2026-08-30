"""
Extract real ESA WorldCover 2021 land cover fractions per H3 cell.

Searches recursively under WORLDCOVER_DIR for the two tiles by filename
pattern (N00E099 / N03E099), so it doesn't matter if your zip extraction
created a nested/odd folder structure — it just needs both real .tif
tiles to exist somewhere under that root.

TILE-SEAM HANDLING — the bug this script used to have
------------------------------------------------------
Tiles N00E099 and N03E099 meet at latitude 3.0, and 100 cells of the
Klang Valley grid straddle that line. The previous version processed each
tile separately and did `combined[h3] = (stats, total)`, so for a
straddling cell whichever tile was processed SECOND overwrote the first.
The cell's composition was then computed from only the sliver of its area
inside that tile — and because the percentages were normalised within
that sliver they still summed to 100, which hid the error from any
sum-based sanity check.

This version ACCUMULATES per-class pixel counts across tiles first and
normalises once at the end, so a straddling cell is described by its
whole area. Cells entirely inside one tile are unaffected (the other
tile contributes an empty count dict).

Run: python scripts/extract/worldcover.py
"""
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Repo root on the path so these standalone scripts can import config.
sys.path.insert(0, str(PROJECT_ROOT))
WORLDCOVER_DIR = PROJECT_ROOT / "data" / "raw" / "worldcover"

GRID_PARQUET = PROJECT_ROOT / "data" / "processed" / "grid_geometry.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "worldcover_by_h3.parquet"

CLASS_MAP = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse_veg", 70: "snow_ice", 80: "water",
    90: "wetland_herbaceous", 95: "mangroves", 100: "moss_lichen",
}


def find_tiles(root: Path) -> dict:
    """Recursively find both required tiles regardless of nesting/naming quirks."""
    all_files = glob.glob(str(root / "**" / "*"), recursive=True)
    tiles = {}
    for f in all_files:
        p = Path(f)
        if not p.is_file():
            continue
        name_upper = p.name.upper()
        if "N00E099" in name_upper:
            tiles["N00E099"] = p
        elif "N03E099" in name_upper:
            tiles["N03E099"] = p
        elif p.suffix.lower() == ".tif" or p.suffix == "":
            # Unnamed / renamed download (e.g. a browser-saved file with no
            # extension) — identify it by its real raster bounds instead.
            try:
                import rasterio
                with rasterio.open(p) as src:
                    b = src.bounds
                    if abs(b.bottom - 3.0) < 0.01 and abs(b.top - 6.0) < 0.01:
                        tiles.setdefault("N03E099", p)
                    elif abs(b.bottom - 0.0) < 0.01 and abs(b.top - 3.0) < 0.01:
                        tiles.setdefault("N00E099", p)
            except Exception:
                pass

    missing = {"N00E099", "N03E099"} - set(tiles.keys())
    if missing:
        raise FileNotFoundError(
            f"Could not find tile(s) {missing} under {root}. Found so far: "
            f"{ {k: str(v) for k, v in tiles.items()} }. Check the files are "
            "present and are real GeoTIFFs."
        )
    return tiles


def main():
    if not GRID_PARQUET.exists():
        raise FileNotFoundError(
            f"{GRID_PARQUET} not found — run the base pipeline's grid "
            "generation step first."
        )
    grid = gpd.read_parquet(GRID_PARQUET)

    tiles = find_tiles(WORLDCOVER_DIR)
    print("Using tiles:")
    for name, path in tiles.items():
        print(f"  {name}: {path}")

    # Process tiles SEPARATELY (not mosaicked) to avoid loading two full
    # 36000x36000 rasters into memory at once — but ACCUMULATE their pixel
    # counts per cell rather than letting the second tile overwrite the
    # first (see the TILE-SEAM HANDLING note in the module docstring).
    pixel_counts = {}   # h3_index -> {class_code: pixel_count}
    tiles_touching = {}  # h3_index -> set of tile names contributing pixels

    for tile_name, path in tiles.items():
        print(f"Processing {tile_name}...")
        stats = zonal_stats(grid, str(path), categorical=True, nodata=0)
        n_new = 0
        for i, s in enumerate(stats):
            if not s:
                continue
            h3_index = grid.iloc[i]["h3_index"]
            acc = pixel_counts.setdefault(h3_index, {})
            if not acc:
                n_new += 1
            for code, count in s.items():
                acc[int(code)] = acc.get(int(code), 0) + int(count)
            tiles_touching.setdefault(h3_index, set()).add(tile_name)
        print(f"  contributed pixels to {len(stats) and sum(1 for s in stats if s)} cells "
              f"({n_new} of them not seen in a previous tile); cumulative {len(pixel_counts)}")

    n_seam = sum(1 for names in tiles_touching.values() if len(names) > 1)
    print(f"\nCells receiving pixels from MORE THAN ONE tile (the seam): {n_seam}")
    print("  These are the cells the previous version described using only a fraction")
    print("  of their area. Their class counts are now summed across both tiles.")

    rows = []
    for _, row_grid in grid.iterrows():
        h3_index = row_grid["h3_index"]
        row = {"h3_index": h3_index}
        acc = pixel_counts.get(h3_index)
        total = sum(acc.values()) if acc else 0
        if total > 0:
            for code, name in CLASS_MAP.items():
                row[f"landcover_{name}_pct"] = acc.get(code, 0) / total * 100.0
            row["worldcover_tile_covered"] = True
            row["worldcover_pixel_count"] = total
            row["worldcover_n_tiles"] = len(tiles_touching.get(h3_index, ()))
        else:
            for code, name in CLASS_MAP.items():
                row[f"landcover_{name}_pct"] = np.nan
            row["worldcover_tile_covered"] = False
            row["worldcover_pixel_count"] = 0
            row["worldcover_n_tiles"] = 0
        rows.append(row)

    out = pd.DataFrame(rows)
    covered = int(out["worldcover_tile_covered"].sum())
    print(f"\nFINAL: {covered}/{len(out)} cells covered ({covered / len(out) * 100:.1f}%)")
    if covered < len(out):
        print("WARNING: not all cells covered — check whether both real tiles were found correctly above.")

    # Completeness guard: with all 11 classes retained and counts summed
    # across tiles, every covered cell's percentages must total 100.
    pct_cols = [f"landcover_{name}_pct" for name in CLASS_MAP.values()]
    sums = out.loc[out["worldcover_tile_covered"], pct_cols].sum(axis=1)
    if len(sums) and not np.allclose(sums, 100.0, atol=0.01):
        raise ValueError(
            f"Land-cover percentages do not sum to 100 for every covered cell "
            f"(min {sums.min():.4f}, max {sums.max():.4f}). This should be impossible "
            "with all 11 classes retained — investigate before using the output."
        )
    print(f"Completeness check OK: all {len(sums)} covered cells sum to 100%.")

    out["worldcover_source"] = "REAL (ESA WorldCover 2021 v200, tiles N00E099+N03E099, counts summed across tiles)"
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved -> {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
