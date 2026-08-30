"""
VIIRS nighttime lights as a per-CELL, per-QUARTER demand signal.

WHY THIS COMPLETES THE PICTURE
------------------------------
The project's demand features have been static from the start: population,
POIs, built-up ratio and the original 6-month VIIRS average were all
constant per cell across every quarter. The DOSM district population
series fixed that partially — real, temporal, but shared by every cell in
a district (8 districts over 4,918 cells).

With 43 monthly VNP46A3 granules spanning 2022-12 to 2026-06, VIIRS now
covers the whole panel and becomes a demand signal that varies per CELL
per QUARTER. That is the finest-resolution time-varying demand proxy in
the project, and it is what makes "did QoE degrade where activity grew?"
answerable at hexagon level rather than district level.

WHAT IS AND IS NOT CLAIMED
--------------------------
Nighttime radiance is a proxy for economic and human activity, not a
measurement of mobile demand. It responds to lighting, construction and
land use. Growth in radiance is consistent with growth in activity; it is
not traffic, and it is not users. Say "associated with rising activity",
never "caused by rising demand".

RETRIEVAL NOISE — HANDLED, NOT IGNORED
--------------------------------------
Monthly VIIRS retrievals are affected by cloud cover, which in this
equatorial study area is substantial (some months lose over 20% of cells).
Two consequences are handled explicitly:

  1. A quarter's value is the mean of whichever months in it produced a
     valid retrieval, and `viirs_n_months_in_quarter` records how many.
     A quarter backed by one cloudy month is not the same evidence as one
     backed by three, and the column lets a consumer weight or filter.
  2. Year-on-year change is preferred over quarter-on-quarter for the
     growth feature. Consecutive quarters differ partly by season
     (monsoon); the same quarter a year apart does not, so YoY isolates
     real change far better at this latitude.
"""
import logging

import numpy as np
import pandas as pd

from config import PROCESSED_DIR

logger = logging.getLogger(__name__)

VIIRS_LONG_PATH = PROCESSED_DIR / "viirs_monthly_long.parquet"

# A quarter needs at least this many valid monthly retrievals before its
# growth features are emitted. One month out of three is a value, but it is
# not a reliable basis for a change measure.
MIN_MONTHS_PER_QUARTER_FOR_GROWTH = 2


def load_viirs_quarterly(path=VIIRS_LONG_PATH) -> pd.DataFrame:
    """Collapse the monthly long table to one row per (cell, quarter)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/extract/viirs.py` first (it writes the "
            "long-format monthly file alongside the wide static summary)."
        )

    monthly = pd.read_parquet(path)
    q = monthly.groupby(["h3_index", "quarter_index"]).agg(
        viirs_rad_quarter=("viirs_avg_rad", "mean"),
        viirs_n_months_in_quarter=("viirs_avg_rad", "size"),
        quarter_label=("quarter_label", "first"),
        year=("year", "first"),
        quarter=("quarter", "first"),
    ).reset_index()

    logger.info(
        "VIIRS quarterly: %d cell-quarters across %d quarters (%s-%s); median %d months "
        "backing each quarter.",
        len(q), q["quarter_label"].nunique(),
        q["quarter_label"].min(), q["quarter_label"].max(),
        int(q["viirs_n_months_in_quarter"].median()),
    )
    return q


def build_viirs_growth_features(q: pd.DataFrame) -> pd.DataFrame:
    """
    Year-on-year radiance change per cell.

    YoY rather than quarter-on-quarter deliberately: at this latitude the
    monsoon makes consecutive quarters differ by season as much as by
    activity, while the same quarter one year earlier is seasonally
    comparable. The 4-quarter lag is validated against the actual
    quarter_index gap, so a cell with a missing quarter does not silently
    get a 3- or 5-quarter comparison labelled as year-on-year.
    """
    df = q.sort_values(["h3_index", "quarter_index"]).copy()
    g = df.groupby("h3_index", sort=False)

    lag4 = g["viirs_rad_quarter"].shift(4)
    gap4 = df["quarter_index"] - g["quarter_index"].shift(4)
    months4 = g["viirs_n_months_in_quarter"].shift(4)

    valid = (
        (gap4 == 4)
        & (df["viirs_n_months_in_quarter"] >= MIN_MONTHS_PER_QUARTER_FOR_GROWTH)
        & (months4 >= MIN_MONTHS_PER_QUARTER_FOR_GROWTH)
        & (lag4 > 0)
    )
    df["viirs_rad_yoy_growth_pct"] = np.where(
        valid, (df["viirs_rad_quarter"] / lag4 - 1.0) * 100.0, np.nan
    )
    # Percentage change explodes on a dim baseline: a cell going from 0.1 to
    # 1.3 reads as +1,200% growth, which is arithmetically true and
    # analytically useless. 1.37% of valid rows exceed +100%, and their
    # median baseline radiance is 1.26 against 11.92 for all rows — so the
    # tail is real but skewed toward dark cells. Rather than threshold the
    # data away (a cut at baseline >= 2.0 still leaves a +1,152% row, so it
    # would discard real signal without fixing the tail), two better-behaved
    # representations of the same change ship alongside:
    #
    #   _change_abs : difference in radiance units. Unaffected by a small
    #                 denominator; the right choice when comparing cells of
    #                 very different brightness.
    #   _log_ratio  : log(current / year-earlier). The natural scale for
    #                 multiplicative growth — symmetric (a halving and a
    #                 doubling are equal and opposite) and tail-tamed. This
    #                 is the one to model on.
    df["viirs_rad_yoy_change_abs"] = np.where(valid, df["viirs_rad_quarter"] - lag4, np.nan)
    df["viirs_rad_yoy_log_ratio"] = np.where(
        valid & (df["viirs_rad_quarter"] > 0), np.log(df["viirs_rad_quarter"] / lag4), np.nan
    )
    df["viirs_yoy_is_valid"] = valid.astype(int)

    n_valid = int(valid.sum())
    logger.info(
        "VIIRS year-on-year growth computed for %d of %d cell-quarters (%.1f%%). The rest lack a "
        "seasonally-comparable quarter 4 periods earlier, or rest on fewer than %d valid monthly "
        "retrievals — NaN, never imputed.",
        n_valid, len(df), 100 * n_valid / max(len(df), 1), MIN_MONTHS_PER_QUARTER_FOR_GROWTH,
    )
    return df


def attach_to_quarterly_panel(quarterly_df: pd.DataFrame, viirs_q: pd.DataFrame) -> pd.DataFrame:
    """
    Join per-cell per-quarter VIIRS onto the Ookla quarterly panel.

    Both are keyed on (h3_index, quarter_index) and both are real
    measurements of the same period, so this is a straight temporal join
    with no carry-forward and no imputation. Cells or quarters VIIRS could
    not observe stay NaN.
    """
    cols = ["h3_index", "quarter_index", "viirs_rad_quarter",
            "viirs_n_months_in_quarter", "viirs_rad_yoy_growth_pct",
            "viirs_rad_yoy_change_abs", "viirs_rad_yoy_log_ratio", "viirs_yoy_is_valid"]
    before = len(quarterly_df)
    merged = quarterly_df.merge(viirs_q[cols], on=["h3_index", "quarter_index"], how="left")
    if len(merged) != before:
        raise ValueError(f"Row count changed joining VIIRS quarterly ({before} -> {len(merged)}).")

    n_val = int(merged["viirs_rad_quarter"].notna().sum())
    n_growth = int(merged["viirs_rad_yoy_growth_pct"].notna().sum())
    logger.info(
        "VIIRS joined to the quarterly panel: %d/%d rows have a per-cell per-quarter radiance "
        "value (%.1f%%), %d have year-on-year growth (%.1f%%). This is the project's first "
        "demand feature that varies over time at CELL resolution.",
        n_val, len(merged), 100 * n_val / len(merged),
        n_growth, 100 * n_growth / len(merged),
    )
    return merged
