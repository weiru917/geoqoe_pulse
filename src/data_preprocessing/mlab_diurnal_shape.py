"""
Real hourly QoE shape, derived from M-Lab NDT measurements.

Replaces the synthetic sine-wave diurnal formula previously used in
data_downloader.build_dynamic_panel(). This module does NOT invent an
hourly shape — it reads it from data/raw/mlab.csv, which is the output of
a real BigQuery aggregation over `measurement-lab.ndt.unified_downloads`
for the Klang Valley bounding box.

TIMEZONE — the bug this module used to have, and how it is now handled
-----------------------------------------------------------------------
mlab.csv is produced by BigQuery's EXTRACT(HOUR/DAYOFWEEK ...), which
returns **UTC**. The hourly panel's timestamps are naive local Klang
Valley time (UTC+8). The previous version looked the UTC-keyed curve up
with a local-time hour, so the entire diurnal pattern was applied EIGHT
HOURS out of phase: the panel's worst hour landed at 14:00 "local" and
its best at 21:00 "local", and `is_peak_period` (local 08-10 / 17-20)
ended up NEGATIVELY associated with degradation.

The CSV proves its own timezone. Reading the raw file as UTC:
    busiest hour (lowest download, most tests) = 14 UTC = 22:00 MYT
    quietest hour (highest download, fewest tests) = 20-21 UTC = 04-05 MYT
which is exactly the expected mobile pattern. Read as local time it would
mean Klang Valley runs most of its speed tests at 2pm and almost none at
8pm, which is not credible.

`load_mlab_shape_lookup()` therefore converts the curve from UTC keys to
LOCAL (Asia/Kuala_Lumpur) keys once, at load time — 168 rows, O(1) cost —
so every downstream caller can keep passing naive local timestamps and
get the correct factor. Malaysia has a fixed UTC+8 offset and observes no
daylight saving, so a constant shift is exact, not an approximation.

data_validator.validate_diurnal_alignment() asserts after the fact that
the assembled panel's worst hour actually lands in the evening, so this
cannot silently regress.

OTHER HONESTY NOTES — read before treating this as a full fix:
  1. This is a CITY-WIDE curve (168 rows: hour x day-of-week), not a
     per-cell one. The same shape multiplier is applied to every H3 cell.
     It replaces a synthetic *shape*; it does not add spatial resolution.
     Consequence: for Ookla-anchored cells the panel factorises exactly
     as level(cell) x curve(hour), i.e. it is rank-one and the hourly
     target is analytically reconstructable. See metadata.json's
     panel_structure_disclosure block.
  2. The underlying BigQuery query did not filter by ASN, so the sample
     may include non-cellular (fixed/WiFi) NDT tests alongside mobile
     ones. Treat the shape as "real regional network diurnal pattern,
     traffic mix not fully verified" rather than "real mobile-only".
  3. If the panel's date range needs a (day, hour) combination outside
     the CSV's 168 rows, this module falls back to a neutral factor of
     1.0 rather than inventing one. With a complete 24x7 table that
     cannot happen; the fallback exists for a truncated future pull.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MLAB_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "mlab.csv"

# The timezone the raw CSV's hour_of_day / day_of_week columns are in.
MLAB_SOURCE_TIMEZONE = "UTC"
# The timezone the pipeline's naive panel timestamps represent.
PANEL_TIMEZONE = "Asia/Kuala_Lumpur"
# Fixed offset, in hours, from MLAB_SOURCE_TIMEZONE to PANEL_TIMEZONE.
# Malaysia is UTC+8 year-round with no daylight saving, so this is exact.
MLAB_UTC_TO_PANEL_OFFSET_HOURS = 8

HOURS_PER_WEEK = 168


# BigQuery's EXTRACT(DAYOFWEEK ...) convention: 1=Sunday ... 7=Saturday.
# pandas Timestamp.dayofweek convention: 0=Monday ... 6=Sunday.
def _pandas_dow_to_bigquery_dow(pandas_dow: int) -> int:
    return ((pandas_dow + 1) % 7) + 1


def _shift_week_key(bq_dow: int, hour: int, offset_hours: int) -> tuple:
    """
    Shift a (BigQuery day-of-week, hour) key by `offset_hours`, rolling
    over the week boundary correctly. Returns a (bq_dow, hour) key in the
    shifted timezone.
    """
    absolute_hour = (bq_dow - 1) * 24 + hour
    shifted = (absolute_hour + offset_hours) % HOURS_PER_WEEK
    return shifted // 24 + 1, shifted % 24


def load_mlab_shape_lookup(csv_path: Path = MLAB_CSV_PATH) -> dict:
    """
    Build a {(bigquery_day_of_week, hour_of_day): (dl_factor, rtt_factor)}
    lookup from the real M-Lab hourly curve, **keyed in local
    (Asia/Kuala_Lumpur) time** so callers can pass the pipeline's naive
    local timestamps directly.

    Factors are normalized against the CSV's OWN overall mean (not
    per-day-of-week), so that a factor of 1.0 means "typical for the week",
    and applying these factors to a static (already-averaged) Ookla
    baseline does not silently rescale that baseline's overall level.
    Normalization happens before the timezone shift and is unaffected by
    it — shifting keys permutes the 168 rows, it does not change their mean.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Real M-Lab diurnal curve not found at {csv_path}. "
            "This module refuses to fall back to a synthetic formula — "
            "either place the real CSV there, or use the synthetic path "
            "explicitly if you truly intend a simulated shape."
        )

    df = pd.read_csv(csv_path)
    required = {"hour_of_day", "day_of_week", "median_download_mbps", "median_min_rtt_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"M-Lab CSV is missing expected columns: {missing}")

    overall_mean_dl = df["median_download_mbps"].mean()
    overall_mean_rtt = df["median_min_rtt_ms"].mean()

    df["dl_factor"] = df["median_download_mbps"] / overall_mean_dl
    df["rtt_factor"] = df["median_min_rtt_ms"] / overall_mean_rtt

    lookup = {}
    for row in df.itertuples():
        local_key = _shift_week_key(
            int(row.day_of_week), int(row.hour_of_day), MLAB_UTC_TO_PANEL_OFFSET_HOURS
        )
        lookup[local_key] = (float(row.dl_factor), float(row.rtt_factor))

    if len(lookup) != HOURS_PER_WEEK:
        raise ValueError(
            f"Expected {HOURS_PER_WEEK} (hour, day-of-week) combinations in "
            f"the M-Lab CSV, found {len(lookup)}. Refusing to proceed with "
            "a partial real shape table — check the source query/export."
        )

    return lookup


# --------------------------------------------------------------------------
# Per-district curves (optional upgrade over the single citywide curve)
# --------------------------------------------------------------------------
# WHY THIS EXISTS: applying ONE curve to every cell makes the anchored
# panel rank-one — value(cell, hour) = level(cell) x shape(hour) — so the
# forecast target is analytically reconstructable and no model can learn
# that one area degrades earlier than another. Giving each district its
# own measured curve breaks that, because two cells with the same Ookla
# level in different districts then have genuinely different weeks.
#
# THIS MODULE DOES NOT INVENT DISTRICT CURVES. It reads them from an
# optional real file (see MLAB_DISTRICT_CSV_PATH). If that file is absent,
# every cell falls back to the citywide curve and is LABELLED as having
# done so via the returned scope string — the panel then behaves exactly
# as it does today, with no silent difference.
#
# To produce the file, re-run the M-Lab BigQuery aggregation grouped by
# rounded client geolocation as well as hour and day-of-week, extracting
# time in Asia/Kuala_Lumpur. Run the geolocation diagnostic first: M-Lab
# warns its IP geolocation is not street-level, and Malaysian carriers use
# CGNAT pools that may collapse to a handful of points. If fewer than ~5
# distinct locations come back, this upgrade is not supported by the data
# and the citywide fallback is the honest answer.
#
# EVIDENCE THAT ONE SHARED SHAPE IS NOT ENOUGH: see
# data/outputs/separability_test.json. Tested against the only real
# per-cell timestamped data in the study area (Sunway, 10 adjacent cells),
# the shared-shape assumption FAILED for passive downlink throughput
# (median per-cell R^2 0.54 but relative dispersion 0.67 across cells) while
# passing cleanly for RSRP (R^2 0.82, dispersion 0.01). Radio conditions are
# shared; load-driven throughput is not — even within 6 km^2.
MLAB_DISTRICT_CSV_PATH = MLAB_CSV_PATH.parent / "mlab_by_district.csv"

# H3 resolution used to bucket cells into "districts" for per-district
# curves. Res 5 is ~20 km across and gives 26 groups over the Klang Valley
# grid — coarse enough that M-Lab's 2.05M tests still leave ~470 tests per
# (district, hour, day-of-week) bucket.
DISTRICT_SHAPE_RESOLUTION = 5

# Minimum real tests behind a (district, hour, day-of-week) bucket before
# its factor is trusted. Thinner buckets fall back to the citywide curve
# rather than contributing a noisy shape.
MIN_TESTS_PER_DISTRICT_BUCKET = 30

SCOPE_DISTRICT = "district_mlab_curve"
SCOPE_CITYWIDE = "citywide_mlab_curve_fallback"


def load_district_shape_lookups(csv_path: Path = MLAB_DISTRICT_CSV_PATH) -> dict:
    """
    Build {h3_parent_res5: {(bq_dow, hour): (dl_factor, rtt_factor)}} from a
    real per-district M-Lab export, keyed in LOCAL time like the citywide
    lookup.

    Returns an EMPTY dict if the file does not exist — that is the normal,
    expected state until the per-district BigQuery aggregation has been
    run, and callers fall back to the citywide curve.

    Expected columns: h3_parent_res5 (or latitude/longitude to derive it),
    hour_of_day, day_of_week, n_tests, median_download_mbps,
    median_min_rtt_ms. hour_of_day/day_of_week must be in the SAME
    timezone as the citywide CSV (UTC) — the same conversion is applied.

    Each district's factors are normalised against THAT DISTRICT's own
    mean, so a factor of 1.0 means "typical for this district" and
    applying it to a cell's Ookla level does not rescale that level.
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    required = {"hour_of_day", "day_of_week", "median_download_mbps", "median_min_rtt_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Per-district M-Lab CSV is missing expected columns: {missing}")

    if "h3_parent_res5" not in df.columns:
        if not {"latitude", "longitude"} <= set(df.columns):
            raise ValueError(
                "Per-district M-Lab CSV needs either an h3_parent_res5 column or "
                "latitude/longitude columns to derive it from."
            )
        import h3
        df["h3_parent_res5"] = [
            h3.latlng_to_cell(lat, lon, DISTRICT_SHAPE_RESOLUTION)
            for lat, lon in zip(df["latitude"], df["longitude"])
        ]

    if "n_tests" in df.columns:
        before = len(df)
        df = df[df["n_tests"] >= MIN_TESTS_PER_DISTRICT_BUCKET]
        if len(df) < before:
            logger.info(
                "Per-district M-Lab: dropped %d of %d buckets with fewer than %d real tests — "
                "those (district, hour) combinations fall back to the citywide curve.",
                before - len(df), before, MIN_TESTS_PER_DISTRICT_BUCKET,
            )

    lookups = {}
    for parent, group in df.groupby("h3_parent_res5"):
        mean_dl = group["median_download_mbps"].mean()
        mean_rtt = group["median_min_rtt_ms"].mean()
        if not (mean_dl > 0 and mean_rtt > 0):
            continue
        table = {}
        for row in group.itertuples():
            local_key = _shift_week_key(
                int(row.day_of_week), int(row.hour_of_day), MLAB_UTC_TO_PANEL_OFFSET_HOURS
            )
            table[local_key] = (
                float(row.median_download_mbps) / float(mean_dl),
                float(row.median_min_rtt_ms) / float(mean_rtt),
            )
        lookups[parent] = table

    logger.info(
        "Loaded real per-district M-Lab curves for %d district(s) at H3 res-%d. Cells outside "
        "these districts — and (district, hour) buckets below the test threshold — use the "
        "citywide curve and are labelled '%s'.",
        len(lookups), DISTRICT_SHAPE_RESOLUTION, SCOPE_CITYWIDE,
    )
    return lookups


def get_shape_factor_for_cell(citywide_lookup: dict, district_lookups: dict,
                               district_key, timestamp: pd.Timestamp) -> tuple:
    """
    Return (dl_factor, rtt_factor, scope) for one cell at one hour.

    Prefers the cell's own district curve where a real one exists for that
    (district, hour, day-of-week); otherwise falls back to the citywide
    curve. `scope` records which was used, per row, so the panel never
    implies district resolution it does not have.
    """
    bq_dow = _pandas_dow_to_bigquery_dow(timestamp.dayofweek)
    key = (bq_dow, timestamp.hour)

    district_table = district_lookups.get(district_key) if district_lookups else None
    if district_table is not None:
        factors = district_table.get(key)
        if factors is not None:
            return factors[0], factors[1], SCOPE_DISTRICT

    dl_f, rtt_f = citywide_lookup.get(key, (1.0, 1.0))
    return dl_f, rtt_f, SCOPE_CITYWIDE


def describe_shape_source() -> str:
    """Provenance string stamped onto every panel row."""
    return (
        f"real_mlab_citywide_curve_{MLAB_SOURCE_TIMEZONE.lower()}"
        f"_converted_to_{PANEL_TIMEZONE.split('/')[-1].lower()}"
    )


def get_shape_factor(lookup: dict, timestamp: pd.Timestamp) -> tuple:
    """
    Look up (dl_factor, rtt_factor) for a given naive LOCAL pandas
    Timestamp. The lookup returned by load_mlab_shape_lookup() is already
    keyed in local time, so no per-call timezone conversion is needed —
    which also keeps this cheap enough to call once per panel row.

    Falls back to (1.0, 1.0) — a neutral, no-op multiplier — only if the
    exact (day, hour) key is somehow absent, which cannot happen given the
    completeness check in load_mlab_shape_lookup().
    """
    bq_dow = _pandas_dow_to_bigquery_dow(timestamp.dayofweek)
    return lookup.get((bq_dow, timestamp.hour), (1.0, 1.0))
