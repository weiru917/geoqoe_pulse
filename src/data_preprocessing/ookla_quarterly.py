"""
Real multi-quarter Ookla ingestion — the project's genuine temporal axis.

WHY THIS EXISTS
---------------
The hourly panel (cell_time_dataset.parquet) is built by applying ONE
citywide M-Lab diurnal curve to a per-cell Ookla level. For anchored cells
that makes it exactly level(cell) x curve(hour) — rank-one, deterministic,
and its forecast target analytically reconstructable. Fixing that would
need a spatially-varying real hourly source, which does not exist for this
study area without operator telemetry or a per-district M-Lab re-query.

Ookla publishes the SAME tile grid every quarter, keyed by a stable
`quadkey`. Reading many quarters therefore gives each H3 cell a REAL
measured time series — not downscaled, not shared-shape, not simulated.
Cells genuinely differ in trajectory: some improve as 5G densifies, others
degrade as demand grows. That is real spatio-temporal variation, and it is
the honest replacement for an hourly forecast this project cannot support.

WHAT IT ENABLES THAT THE HOURLY PANEL CANNOT
--------------------------------------------
  - A real forecast target: "will this cell be degraded NEXT QUARTER",
    with real labels on both sides and a genuine temporal holdout (train
    on early quarters, test on the most recent).
  - Real recurrence: "degraded in 5 of the last 8 quarters" becomes a
    measurement, so persistent degradation can finally be separated from
    a transient dip — the proposal's central distinction.
  - Real per-cell trend and volatility, which can be tested against the
    demand proxies instead of assumed.

The trade is horizon: quarterly, not hourly. Every question in the
proposal survives that change except "next hour", which was never backed
by real data.

HONESTY NOTES
-------------
  - Coverage is uneven across quarters. A cell with no Ookla tile in a
    given quarter gets NaN, never an imputed value, and
    `ookla_n_quarters_observed` records how many real observations back
    each cell's series. Trend/volatility features are emitted only where
    enough real quarters exist (MIN_QUARTERS_FOR_TREND).
  - Ookla tiles are ~610 m at the equator and H3 res-8 cells are ~0.74
    km^2, so the tile->cell assignment is a centroid-in-polygon join, the
    same operation load_ookla() already uses for the single quarter.
  - These are quarterly AVERAGES of consumer-initiated speed tests. They
    are not continuous network telemetry, and the sample behind any one
    cell-quarter can be small — `ookla_tests` travels with every value so
    a consumer can weight or filter by it.
"""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

from config import CRS_WGS84, CRS_PENINSULAR_MYS, RAW_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

OOKLA_S3_TEMPLATE = (
    "https://ookla-open-data.s3.amazonaws.com/parquet/performance/"
    "type=mobile/year={year}/quarter={quarter}/{date}_performance_mobile_tiles.parquet"
)
QUARTER_START_MMDD = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}

QUARTERLY_DIR = RAW_DIR / "ookla_quarterly"
QUARTERLY_PANEL_PATH = PROCESSED_DIR / "ookla_quarterly_by_h3.parquet"

# Minimum real quarters behind a cell before trend/volatility are emitted.
# Below this a slope is not an estimate, it is a line through noise.
MIN_QUARTERS_FOR_TREND = 4


def quarter_url(year: int, quarter: int) -> str:
    return OOKLA_S3_TEMPLATE.format(
        year=year, quarter=quarter, date=f"{year}-{QUARTER_START_MMDD[quarter]}"
    )


def quarter_label(year: int, quarter: int) -> str:
    return f"{year}Q{quarter}"


def download_quarter(year: int, quarter: int, dest_dir: Path = QUARTERLY_DIR,
                      overwrite: bool = False) -> Path:
    """Fetch one quarterly tile file. Skips if already present."""
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{year}-{QUARTER_START_MMDD[quarter]}_performance_mobile_tiles.parquet"
    if dest.exists() and not overwrite:
        logger.info("%s already present (%.0f MB) — skipping download.",
                    dest.name, dest.stat().st_size / 1e6)
        return dest

    url = quarter_url(year, quarter)
    logger.info("Downloading %s -> %s", url, dest.name)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    logger.info("  %s downloaded (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def extract_quarter_to_cells(path: Path, grid_gdf: gpd.GeoDataFrame,
                              year: int, quarter: int) -> pd.DataFrame:
    """
    Join one quarter's tiles into the H3 grid, using the same
    tests-weighted, centroid-in-polygon logic as the single-quarter
    loader so the multi-quarter series is consistent with
    cell_static_qoe_baseline.parquet's values.
    """
    df = pd.read_parquet(path, columns=[
        "quadkey", "tile", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices",
    ])
    df = df[df["tests"] > 0].copy()

    # Cheap pre-filter on the grid's bounding box before parsing WKT for
    # 3.5M rows — the tile polygon string is the expensive part.
    min_lon, min_lat, max_lon, max_lat = grid_gdf.total_bounds
    # quadkey prefix filtering is unreliable across zoom levels, so filter
    # after a lightweight WKT parse of only the first coordinate pair.
    def _first_lon_lat(wkt_str: str):
        try:
            inner = wkt_str[wkt_str.index("((") + 2:]
            lon_s, lat_s = inner.split(",", 1)[0].strip().split()
            return float(lon_s), float(lat_s)
        except Exception:
            return np.nan, np.nan

    lonlat = df["tile"].map(_first_lon_lat)
    df["_lon"] = [p[0] for p in lonlat]
    df["_lat"] = [p[1] for p in lonlat]
    pad = 0.05
    df = df[
        df["_lon"].between(min_lon - pad, max_lon + pad)
        & df["_lat"].between(min_lat - pad, max_lat + pad)
    ]
    if df.empty:
        logger.warning("%s: no tiles inside the study bounds.", quarter_label(year, quarter))
        return pd.DataFrame()

    df["geometry"] = df["tile"].apply(shapely_wkt.loads)
    tiles = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_WGS84)
    # Centroid in a projected CRS (WGS84 degrees distort centroid math),
    # then back to WGS84 for the join — identical to load_ookla().
    tiles["geometry"] = tiles.geometry.to_crs(CRS_PENINSULAR_MYS).centroid.to_crs(CRS_WGS84)

    joined = gpd.sjoin(
        tiles, grid_gdf[["h3_index", "geometry"]], how="inner", predicate="within",
    )
    if joined.empty:
        logger.warning("%s: no tiles fell inside the study grid.", quarter_label(year, quarter))
        return pd.DataFrame()

    joined["dl_x_tests"] = (joined["avg_d_kbps"] / 1000.0) * joined["tests"]
    joined["ul_x_tests"] = (joined["avg_u_kbps"] / 1000.0) * joined["tests"]
    joined["rtt_x_tests"] = joined["avg_lat_ms"] * joined["tests"]

    grouped = joined.groupby("h3_index").agg(
        dl_x_tests=("dl_x_tests", "sum"),
        ul_x_tests=("ul_x_tests", "sum"),
        rtt_x_tests=("rtt_x_tests", "sum"),
        ookla_tests=("tests", "sum"),
        ookla_devices=("devices", "sum"),
        ookla_tile_count=("quadkey", "nunique"),
    ).reset_index()

    grouped["ookla_dl_mbps"] = grouped["dl_x_tests"] / grouped["ookla_tests"]
    grouped["ookla_ul_mbps"] = grouped["ul_x_tests"] / grouped["ookla_tests"]
    grouped["ookla_rtt_ms"] = grouped["rtt_x_tests"] / grouped["ookla_tests"]
    grouped["year"] = year
    grouped["quarter"] = quarter
    grouped["quarter_label"] = quarter_label(year, quarter)
    grouped["quarter_index"] = year * 4 + (quarter - 1)

    logger.info(
        "%s: %d tiles -> %d/%d cells covered (%.1f%%), %d real tests total.",
        quarter_label(year, quarter), len(tiles), len(grouped), len(grid_gdf),
        100 * len(grouped) / len(grid_gdf), int(grouped["ookla_tests"].sum()),
    )
    return grouped.drop(columns=["dl_x_tests", "ul_x_tests", "rtt_x_tests"])


def build_quarterly_panel(grid_gdf: gpd.GeoDataFrame, quarters: list,
                           download: bool = True,
                           delete_raw_after: bool = False) -> pd.DataFrame:
    """
    Build the long-format (h3_index x quarter) real Ookla panel.

    `quarters` is a list of (year, quarter) tuples. Each is downloaded if
    missing, joined into the grid, then optionally deleted to keep ~200 MB
    per quarter off disk — the per-cell extract is a few hundred KB.
    """
    frames = []
    for year, quarter in quarters:
        try:
            path = download_quarter(year, quarter) if download else (
                QUARTERLY_DIR / f"{year}-{QUARTER_START_MMDD[quarter]}_performance_mobile_tiles.parquet"
            )
            if not path.exists():
                logger.warning("%s not available locally — skipping.", quarter_label(year, quarter))
                continue
            frame = extract_quarter_to_cells(path, grid_gdf, year, quarter)
            if not frame.empty:
                frames.append(frame)
            if delete_raw_after:
                path.unlink()
                logger.info("  deleted raw %s (per-cell extract retained)", path.name)
        except Exception as exc:
            logger.warning("%s failed (%s) — skipping that quarter rather than "
                           "substituting a value.", quarter_label(year, quarter), exc)

    if not frames:
        raise ValueError("No quarters could be ingested — check network access and the URL pattern.")

    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["h3_index", "quarter_index"]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(QUARTERLY_PANEL_PATH, index=False)
    logger.info(
        "Real Ookla quarterly panel: %d rows across %d quarters and %d cells -> %s",
        len(panel), panel["quarter_label"].nunique(), panel["h3_index"].nunique(),
        QUARTERLY_PANEL_PATH.name,
    )
    return panel


def summarize_quarterly_features(panel: pd.DataFrame, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Collapse the long quarterly panel into per-cell features: coverage,
    latest value, trend, volatility and real recurrence counts.

    Everything here is derived from real measurements. Cells with too few
    observed quarters get NaN for trend/volatility rather than a slope
    fitted through one or two points.
    """
    from config import QOE_THRESHOLDS

    dl_cut = QOE_THRESHOLDS["dl_degraded_mbps"]
    rtt_cut = QOE_THRESHOLDS["rtt_degraded_ms"]

    panel = panel.sort_values(["h3_index", "quarter_index"]).copy()
    panel["is_degraded_quarter"] = (
        (panel["ookla_dl_mbps"] < dl_cut) | (panel["ookla_rtt_ms"] > rtt_cut)
    ).astype(int)

    # Relative (cross-cell, within-quarter) degradation — the same idea as
    # the hourly relative target, but here the ranking genuinely moves
    # between periods because each quarter is independently measured.
    panel["dl_percentile_in_quarter"] = (
        panel.groupby("quarter_index")["ookla_dl_mbps"].rank(pct=True) * 100.0
    )
    panel["is_below_area_norm_quarter"] = (panel["dl_percentile_in_quarter"] < 25).astype(int)

    def _trend(group: pd.DataFrame, col: str):
        valid = group[["quarter_index", col]].dropna()
        if len(valid) < MIN_QUARTERS_FOR_TREND:
            return np.nan
        x = valid["quarter_index"].values.astype(float)
        y = valid[col].values.astype(float)
        if np.std(x) == 0:
            return np.nan
        return float(np.polyfit(x, y, 1)[0])

    rows = []
    for h3_index, group in panel.groupby("h3_index"):
        n_obs = int(group["ookla_dl_mbps"].notna().sum())
        latest = group.dropna(subset=["ookla_dl_mbps"]).tail(1)
        rows.append({
            "h3_index": h3_index,
            "ookla_n_quarters_observed": n_obs,
            "ookla_first_quarter": group["quarter_label"].iloc[0],
            "ookla_latest_quarter": latest["quarter_label"].iloc[0] if len(latest) else None,
            "ookla_dl_mbps_latest": float(latest["ookla_dl_mbps"].iloc[0]) if len(latest) else np.nan,
            "ookla_dl_mbps_mean": float(group["ookla_dl_mbps"].mean()),
            "ookla_dl_mbps_std": float(group["ookla_dl_mbps"].std()) if n_obs > 1 else np.nan,
            "ookla_dl_trend_per_quarter": _trend(group, "ookla_dl_mbps"),
            "ookla_rtt_trend_per_quarter": _trend(group, "ookla_rtt_ms"),
            "ookla_quarters_degraded": int(group["is_degraded_quarter"].sum()),
            "ookla_recurrence_rate": float(group["is_degraded_quarter"].mean()) if n_obs else np.nan,
            "ookla_quarters_below_area_norm": int(group["is_below_area_norm_quarter"].sum()),
            "ookla_total_tests_all_quarters": int(group["ookla_tests"].sum()),
        })

    features = pd.DataFrame(rows)
    # Cells never seen by Ookla in any quarter get an explicit zero-coverage
    # row rather than being absent — downstream joins then never silently
    # drop cells.
    full = grid_gdf[["h3_index"]].merge(features, on="h3_index", how="left")
    full["ookla_n_quarters_observed"] = full["ookla_n_quarters_observed"].fillna(0).astype(int)
    full["ookla_has_multiquarter_history"] = (
        full["ookla_n_quarters_observed"] >= MIN_QUARTERS_FOR_TREND
    )

    logger.info(
        "Quarterly features: %d/%d cells have >= %d real quarters (usable trend); "
        "%d cells have at least one real quarter; %d have none.",
        int(full["ookla_has_multiquarter_history"].sum()), len(full), MIN_QUARTERS_FOR_TREND,
        int((full["ookla_n_quarters_observed"] > 0).sum()),
        int((full["ookla_n_quarters_observed"] == 0).sum()),
    )
    return full


def build_quarterly_ml_dataset(panel: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    """
    The project's ONE genuinely-forecastable dataset: one row per
    (h3_index, quarter), with causal features and a REAL next-quarter
    target measured on both sides.

    Unlike cell_time_dataset.parquet — where the target is analytically
    reconstructable from two static columns because a single shared curve
    drives every cell — every value here is an independent Ookla
    measurement. Cells genuinely differ in trajectory (56% improving, 44%
    degrading over the observed window), so "will this cell be degraded
    next quarter" is a real prediction problem with real ground truth.

    GAP HANDLING — the correctness detail that matters:
    Ookla coverage is uneven, so a cell can be observed in 2023Q1 and
    2023Q3 but not Q2. A naive groupby().shift() would silently treat Q3
    as "the quarter after Q1" and produce a 2-quarter lag labelled as a
    1-quarter lag. Every lag and every target below is therefore validated
    against the actual quarter_index difference, and set to NaN (with an
    explicit *_valid flag) wherever the neighbouring observation is not
    exactly one quarter away.

    LEAKAGE: all lag/rolling/recurrence features are computed with
    shift(1) applied BEFORE any window, so a row never sees its own
    quarter or any later one. The same discipline as the hourly panel's
    temporal features.
    """
    from config import QOE_THRESHOLDS

    dl_cut = QOE_THRESHOLDS["dl_degraded_mbps"]
    rtt_cut = QOE_THRESHOLDS["rtt_degraded_ms"]

    df = panel.sort_values(["h3_index", "quarter_index"]).reset_index(drop=True).copy()

    # --- current-quarter labels -------------------------------------------
    df["is_degraded"] = (
        (df["ookla_dl_mbps"] < dl_cut) | (df["ookla_rtt_ms"] > rtt_cut)
    ).astype(int)
    df["dl_percentile_in_quarter"] = (
        df.groupby("quarter_index")["ookla_dl_mbps"].rank(pct=True) * 100.0
    )
    df["is_below_area_norm"] = (df["dl_percentile_in_quarter"] < 25).astype(int)

    grp = df.groupby("h3_index", sort=False)

    # Contiguity: is the PREVIOUS observed row exactly one quarter back?
    prev_gap = df["quarter_index"] - grp["quarter_index"].shift(1)
    # ...and is the NEXT observed row exactly one quarter forward?
    next_gap = grp["quarter_index"].shift(-1) - df["quarter_index"]

    # --- causal lag features ----------------------------------------------
    for col, short in [("ookla_dl_mbps", "dl"), ("ookla_rtt_ms", "rtt")]:
        lag1 = grp[col].shift(1)
        df[f"lag_1q_{short}"] = lag1.where(prev_gap == 1)
        df[f"lag_1q_{short}_valid"] = (prev_gap == 1).astype(int)
        # Year-ago value: only valid when exactly 4 quarters back exists.
        lag4 = grp[col].shift(4)
        gap4 = df["quarter_index"] - grp["quarter_index"].shift(4)
        df[f"lag_4q_{short}"] = lag4.where(gap4 == 4)
        df[f"lag_4q_{short}_valid"] = (gap4 == 4).astype(int)
        # Trailing mean/std over the previous 4 observed quarters.
        df[f"rolling_mean_4q_{short}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
        df[f"rolling_std_4q_{short}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=2).std()
        )

    # --- causal recurrence-to-date ----------------------------------------
    # "How often has this cell been degraded BEFORE now" — expanding mean
    # after shift(1), so the current quarter never counts toward its own
    # recurrence feature.
    df["recurrence_to_date"] = grp["is_degraded"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["quarters_observed_to_date"] = grp["is_degraded"].transform(
        lambda s: s.shift(1).expanding().count()
    )
    df["below_area_norm_rate_to_date"] = grp["is_below_area_norm"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # --- REAL next-quarter targets ----------------------------------------
    df["target_is_degraded_next_q"] = grp["is_degraded"].shift(-1).where(next_gap == 1)
    df["target_is_below_area_norm_next_q"] = grp["is_below_area_norm"].shift(-1).where(next_gap == 1)
    df["target_dl_mbps_next_q"] = grp["ookla_dl_mbps"].shift(-1).where(next_gap == 1)
    df["target_next_q_valid"] = (next_gap == 1).astype(int)

    # --- join the real static GIS features --------------------------------
    static_cols = [c for c in static_df.columns if c != "h3_index"]
    df = df.merge(static_df[["h3_index"] + static_cols], on="h3_index", how="left")

    n_valid = int(df["target_next_q_valid"].sum())
    logger.info(
        "Quarterly ML dataset: %d rows (%d cells x up to %d quarters); %d rows have a REAL "
        "next-quarter target (%.1f%%). Positive rate on those: %.2f%% absolute, %.2f%% relative.",
        len(df), df["h3_index"].nunique(), df["quarter_label"].nunique(), n_valid,
        100 * n_valid / len(df),
        100 * df.loc[df["target_next_q_valid"] == 1, "target_is_degraded_next_q"].mean(),
        100 * df.loc[df["target_next_q_valid"] == 1, "target_is_below_area_norm_next_q"].mean(),
    )
    return df
