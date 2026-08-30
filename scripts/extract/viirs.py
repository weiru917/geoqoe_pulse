"""
Extract real VIIRS/VNP46A3 nighttime lights per H3 cell.

This is the h5 -> H3 zonal-stats "conversion" step. No reprojection
library is needed: VNP46A3's lat/lon fields are a plain regular grid
(Linear Lat/Lon Grid product), not the MODIS sinusoidal projection some
other products use — confirmed by inspecting the real file structure.

Uses AllAngle_Composite_Snow_Free (not NearNadir) — for this AOI,
near-nadir observations were essentially absent (57% zero-observation
pixels) due to monsoon cloud cover and viewing geometry near the equator.
AllAngle had real, usable coverage. This is a documented, deliberate
choice, not a downgrade.

Run: python scripts/extract/viirs.py
"""
from pathlib import Path
import glob
import re
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterstats import zonal_stats

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Repo root on the path so these standalone scripts can import config.
sys.path.insert(0, str(PROJECT_ROOT))
# All VNP46A3 download folders. LAADS names each order directory by
# timestamp, so a backfill arrives as a second folder rather than merging
# into the first. Both are scanned and deduplicated by acquisition id, so
# re-ordering an overlapping range is harmless.
VIIRS_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "VNP46A3_2-20260826_125341",
    PROJECT_ROOT / "data" / "raw" / "VNP46A3_2-20260829_142049",
]

GRID_PARQUET = PROJECT_ROOT / "data" / "processed" / "grid_geometry.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "viirs_by_h3.parquet"
# Long-format per-month values, for joining to the QUARTERLY panel. The
# wide file above stays the static summary; this one is what makes VIIRS
# a genuinely time-varying feature at CELL resolution.
LONG_OUTPUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "viirs_monthly_long.parquet"
TMP_DIR = PROJECT_ROOT / "data" / "raw" / "_viirs_tmp"

# AOI bounding box with a small buffer — matches Klang Valley study area
LAT_MIN, LAT_MAX = 2.5, 3.5
LON_MIN, LON_MAX = 101.1, 102.05


def find_h5_files(viirs_dirs=None) -> list:
    """
    Scan every download folder, deduplicating by acquisition id so an
    overlapping re-order does not double-count a month. Where the same
    acquisition appears twice, the file with the later PROCESSING stamp
    (the trailing number in the NASA filename) wins — that is the more
    recently reprocessed version.
    """
    viirs_dirs = viirs_dirs if viirs_dirs is not None else VIIRS_DIRS
    by_acq = {}
    for d in viirs_dirs:
        if not d.exists():
            continue
        for f in sorted(glob.glob(str(d / "**" / "*.h5"), recursive=True)):
            name = Path(f).name
            if "VNP46A3" not in name:
                continue
            acq = acquisition_id(f)
            prev = by_acq.get(acq)
            if prev is None or Path(f).name > Path(prev).name:
                by_acq[acq] = f
    if not by_acq:
        raise FileNotFoundError(f"No VNP46A3 .h5 files found under any of {viirs_dirs}.")
    return [by_acq[a] for a in sorted(by_acq)]


def acquisition_id(filepath: str) -> str:
    """Pull the AYYYYDDD acquisition id regardless of dot/underscore separators."""
    m = re.search(r"A(\d{7})", Path(filepath).name)
    if not m:
        raise ValueError(f"Could not parse acquisition date from {filepath}")
    return m.group(1)


def main():
    import geopandas as gpd

    if not GRID_PARQUET.exists():
        raise FileNotFoundError(
            f"{GRID_PARQUET} not found — run the base pipeline's grid "
            "generation step (or run_pipeline.py) first so the real H3 "
            "grid exists to join against."
        )
    grid = gpd.read_parquet(GRID_PARQUET)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    files = find_h5_files()
    print(f"Found {len(files)} distinct VNP46A3 acquisitions across {len(VIIRS_DIRS)} folder(s).")

    rad_results, num_results = {}, {}

    for fp in files:
        acq = acquisition_id(fp)
        f = h5py.File(fp, "r")
        grp = f["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"]

        rad = grp["AllAngle_Composite_Snow_Free"][:].astype("float32")
        num = grp["AllAngle_Composite_Snow_Free_Num"][:].astype("float32")
        lat = grp["lat"][:]
        lon = grp["lon"][:]
        fill = grp["AllAngle_Composite_Snow_Free"].attrs["_FillValue"][0]
        rad = np.where(rad == fill, np.nan, rad)

        lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
        lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
        lat_sub, lon_sub = lat[lat_mask], lon[lon_mask]
        rad_sub = rad[np.ix_(lat_mask, lon_mask)]
        num_sub = num[np.ix_(lat_mask, lon_mask)]

        res_lat = abs(lat_sub[1] - lat_sub[0])
        res_lon = abs(lon_sub[1] - lon_sub[0])
        top = lat_sub.max() + res_lat / 2
        left = lon_sub.min() - res_lon / 2
        transform = from_origin(left, top, res_lon, res_lat)

        if lat_sub[0] < lat_sub[-1]:
            rad_sub = rad_sub[::-1, :]
            num_sub = num_sub[::-1, :]

        for name, arr in [("rad", rad_sub), ("num", num_sub)]:
            tmp = TMP_DIR / f"_tmp_{name}_{acq}.tif"
            with rasterio.open(
                tmp, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
            ) as dst:
                dst.write(arr.astype("float32"), 1)
            stats = zonal_stats(grid, str(tmp), stats=["mean"], nodata=np.nan)
            vals = [s["mean"] for s in stats]
            if name == "rad":
                rad_results[acq] = vals
            else:
                num_results[acq] = vals

        covered = sum(1 for v in rad_results[acq] if v is not None)
        print(f"  {acq}: radiance covered {covered}/{len(grid)}  median real obs/cell: {np.nanmedian(num_results[acq]):.1f}")

    out = pd.DataFrame({"h3_index": grid["h3_index"]})
    for acq in rad_results:
        out[f"viirs_avg_rad_{acq}"] = rad_results[acq]
    rad_cols = [c for c in out.columns if c.startswith("viirs_avg_rad")]
    out["viirs_avg_rad_mean"] = out[rad_cols].mean(axis=1)
    out["viirs_source"] = f"REAL (VIIRS VNP46A3 AllAngle_Composite_Snow_Free, {len(files)} months)"

    out.to_parquet(OUTPUT_PARQUET)
    print(f"\nSaved -> {OUTPUT_PARQUET}")
    print(f"Cells with a real value in every month: {(~out[rad_cols].isna().any(axis=1)).sum()}/{len(out)}")

    write_long_format(out)


def write_long_format(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the wide per-month table into one row per (cell, month), tagged
    with the quarter it belongs to.

    This is what turns VIIRS from a static feature into a genuinely
    time-varying one at CELL resolution — finer than the district-level
    population series, and the only per-cell demand signal that changes
    across the panel's span. Split out as its own function so it can be
    regenerated from an existing wide extraction without repeating the
    ~25-minute HDF5 pass.
    """
    import datetime as _dt

    month_cols = [c for c in wide.columns
                  if c.startswith("viirs_avg_rad_") and c != "viirs_avg_rad_mean"]
    frames = []
    for col in month_cols:
        acq = col.rsplit("_", 1)[-1]
        year, doy = int(acq[:4]), int(acq[4:])
        date = _dt.date(year, 1, 1) + _dt.timedelta(days=doy - 1)
        quarter = (date.month - 1) // 3 + 1
        frames.append(pd.DataFrame({
            "h3_index": wide["h3_index"].values,
            "viirs_avg_rad": wide[col].values,
            "viirs_month": date.strftime("%Y-%m"),
            "year": date.year,
            "quarter": quarter,
            "quarter_label": f"{date.year}Q{quarter}",
            "quarter_index": date.year * 4 + (quarter - 1),
        }))

    long_df = pd.concat(frames, ignore_index=True)
    # Drop cell-months with no valid retrieval rather than carrying NaN
    # rows — cloud cover is heavy here and an absent retrieval is an absent
    # observation, not a zero.
    long_df = long_df[long_df["viirs_avg_rad"].notna()].reset_index(drop=True)
    long_df.to_parquet(LONG_OUTPUT_PARQUET, index=False)
    print(f"Saved long format -> {LONG_OUTPUT_PARQUET} "
          f"({len(long_df):,} rows, {long_df['quarter_label'].nunique()} quarters, "
          f"{long_df['viirs_month'].nunique()} months)")
    return long_df


if __name__ == "__main__":
    main()
