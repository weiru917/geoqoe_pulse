"""
GeoQoE Pulse Data Validation Module.

Automated checks run at the end of the pipeline, before export. Raises
`ValueError` for structural problems that would break the modelling
pipeline (duplicate keys, invalid geometry, bad CRS), and logs `warning`
for statistical/quality issues that are worth a human's attention but
shouldn't halt the run (out-of-range values, outliers, timestamp gaps).
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CRS_WGS84

logger = logging.getLogger(__name__)


class ValidationReport:
    """Collects errors (fatal) and warnings (non-fatal) from a validation
    pass so run_pipeline can decide what to do with them in one place."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg, *args):
        self.errors.append(msg % args if args else msg)
        logger.error(msg, *args)

    def warn(self, msg, *args):
        self.warnings.append(msg % args if args else msg)
        logger.warning(msg, *args)

    def raise_if_errors(self):
        if self.errors:
            raise ValueError(
                f"Data validation failed with {len(self.errors)} fatal error(s):\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )

    def summary(self) -> dict:
        return {"n_errors": len(self.errors), "n_warnings": len(self.warnings),
                "errors": self.errors, "warnings": self.warnings}


def validate_grid_geometry(grid_gdf: gpd.GeoDataFrame, report: ValidationReport) -> None:
    if grid_gdf.crs is None or str(grid_gdf.crs) != CRS_WGS84:
        report.error("Grid geometry CRS is %s, expected %s.", grid_gdf.crs, CRS_WGS84)

    n_dupes = grid_gdf["h3_index"].duplicated().sum()
    if n_dupes:
        report.error("%d duplicate h3_index value(s) in grid geometry.", n_dupes)

    invalid = ~grid_gdf.geometry.is_valid
    if invalid.any():
        report.error("%d invalid geometr(y/ies) in grid geometry (self-intersections etc.).", invalid.sum())

    empty = grid_gdf.geometry.is_empty
    if empty.any():
        report.error("%d empty geometr(y/ies) in grid geometry.", empty.sum())

    zero_area = grid_gdf["area_m2"] <= 0
    if zero_area.any():
        report.error("%d cell(s) with non-positive area_m2.", zero_area.sum())


def validate_static_features(static_df: pd.DataFrame, report: ValidationReport) -> None:
    n_dupes = static_df["h3_index"].duplicated().sum()
    if n_dupes:
        report.error("%d duplicate h3_index value(s) in static features.", n_dupes)

    numeric_cols = static_df.select_dtypes(include=[np.number]).columns
    null_counts = static_df[numeric_cols].isnull().sum()
    for col, n in null_counts.items():
        if n:
            report.warn("%d null value(s) in static feature '%s'.", n, col)

    for ratio_col in ("built_up_ratio",):
        if ratio_col in static_df.columns:
            out_of_range = ~static_df[ratio_col].between(0, 1)
            if out_of_range.any():
                report.error("%d row(s) with %s outside [0, 1].", out_of_range.sum(), ratio_col)

    synthetic_cols = [c for c in static_df.columns if c.endswith("_is_synthetic")]
    for col in synthetic_cols:
        frac = static_df[col].mean()
        if frac > 0:
            report.warn("%.1f%% of cells use a SYNTHETIC fallback for '%s' (raw file missing).",
                        100 * frac, col.replace("_is_synthetic", ""))

    # Regression guard for the extraction-bounding-box mismatch bug: real
    # (is_synthetic == False) sources should still have a real, uneven
    # spatial distribution of zero counts, not a suspiciously high FLAT
    # fraction of zeros across the whole grid — that pattern is what
    # "raw records silently pre-filtered out by a too-small bounding box"
    # looks like (every cell outside the box reads as a confirmed-real
    # zero, not a fallback). Real infrastructure data is uneven but rarely
    # this uniform; >60% is a conservative trigger, not a hard proof.
    zero_check_cols = {
        "building_count": "building_is_synthetic",
        "tower_count": "tower_is_synthetic",
        "road_length_m": "osm_is_synthetic",
    }
    for count_col, synthetic_col in zero_check_cols.items():
        if count_col in static_df.columns and synthetic_col in static_df.columns:
            real_mask = ~static_df[synthetic_col].astype(bool)
            if real_mask.sum() == 0:
                continue
            zero_frac = (static_df.loc[real_mask, count_col] == 0).mean()
            if zero_frac > 0.60:
                report.warn(
                    "%.1f%% of REAL (non-synthetic) cells have %s == 0 — this can be genuine sparse "
                    "infrastructure, but it's also the exact symptom of a raw-file extraction bounding "
                    "box that's smaller than the actual study grid (every cell outside the box reads as "
                    "a confirmed-real zero rather than a fallback). Sanity-check extraction_bounds() "
                    "against grid_gdf.total_bounds if this fires unexpectedly.",
                    100 * zero_frac, count_col,
                )


def validate_panel(panel_df: pd.DataFrame, expected_cells: int, report: ValidationReport,
                    freq: str = "1h") -> None:
    n_dupes = panel_df.duplicated(subset=["h3_index", "timestamp"]).sum()
    if n_dupes:
        report.error("%d duplicate (h3_index, timestamp) row(s) in panel.", n_dupes)

    n_cells = panel_df["h3_index"].nunique()
    if n_cells != expected_cells:
        report.error("Panel has %d unique h3_index values, expected %d.", n_cells, expected_cells)

    for cell, group in panel_df.groupby("h3_index"):
        ts = group["timestamp"].sort_values()
        expected_range = pd.date_range(ts.min(), ts.max(), freq=freq)
        if len(ts) != len(expected_range):
            report.warn("Cell %s has %d timestamps but expected %d over its own range — gap detected.",
                        cell, len(ts), len(expected_range))
            break  # one representative warning is enough; avoid spamming per-cell

    for col in ("dl_median", "ul_median"):
        if col in panel_df.columns and (panel_df[col] < 0).any():
            report.error("Negative values found in '%s'.", col)
    if "rtt_median" in panel_df.columns and (panel_df["rtt_median"] < 0).any():
        report.error("Negative values found in 'rtt_median'.")

    for col in ("dl_median", "rtt_median"):
        if col not in panel_df.columns:
            continue
        q1, q99 = panel_df[col].quantile([0.01, 0.99])
        iqr_low, iqr_high = panel_df[col].quantile([0.25, 0.75])
        iqr = iqr_high - iqr_low
        lower, upper = iqr_low - 3 * iqr, iqr_high + 3 * iqr
        n_outliers = ((panel_df[col] < lower) | (panel_df[col] > upper)).sum()
        if n_outliers:
            report.warn("%d extreme outlier(s) in '%s' (outside 3xIQR fence [%.2f, %.2f]).",
                        n_outliers, col, lower, upper)


EVENING_HOURS = {17, 18, 19, 20, 21, 22, 23, 0, 1}


def validate_diurnal_alignment(panel_df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Regression guard for the M-Lab timezone bug (see
    mlab_diurnal_shape.py's TIMEZONE section).

    mlab.csv is keyed in UTC; the panel's timestamps are naive local
    (UTC+8) Klang Valley time. A previous version looked the UTC curve up
    with a local hour, applying the whole diurnal pattern 8 hours out of
    phase — the panel's worst hour landed at 14:00 "local" and its best at
    21:00 "local", and `is_peak_period` came out NEGATIVELY associated with
    degradation.

    The invariant asserted here is deliberately about network behaviour,
    not about code: for a mobile network in this city, the hour with the
    lowest median download must fall in the evening/night. If it lands in
    the morning or early afternoon, the curve has been applied in the
    wrong timezone. This is a hard error rather than a warning because
    every downstream temporal claim — the dashboard's time slider, the
    "peak period" narrative, the next-hour forecast target — is wrong in
    the same direction when it fires.

    If the shape source is ever legitimately replaced (a different city, a
    different measurement source), update EVENING_HOURS rather than
    deleting this check.
    """
    if "timestamp" not in panel_df.columns or "dl_median" not in panel_df.columns:
        return

    by_hour = panel_df.groupby(panel_df["timestamp"].dt.hour)["dl_median"].mean()
    if by_hour.empty:
        return

    worst_hour = int(by_hour.idxmin())
    best_hour = int(by_hour.idxmax())

    if worst_hour not in EVENING_HOURS:
        report.error(
            "Diurnal alignment check FAILED: the panel's lowest-throughput hour is %02d:00 local "
            "(best hour %02d:00). For a mobile network in Klang Valley the busiest, slowest hour must "
            "fall in the evening/night (%s). This is the exact signature of the M-Lab UTC-vs-local "
            "timezone bug — mlab.csv is keyed in UTC and must be converted to Asia/Kuala_Lumpur "
            "before lookup (see mlab_diurnal_shape.load_mlab_shape_lookup). Every downstream temporal "
            "claim is shifted by 8 hours while this fails.",
            worst_hour, best_hour, sorted(EVENING_HOURS),
        )
    else:
        logger.info(
            "Diurnal alignment OK: lowest-throughput hour is %02d:00 local, highest is %02d:00 — "
            "consistent with a real evening-peak mobile network pattern.",
            worst_hour, best_hour,
        )

    # Secondary, non-fatal: is_peak_period is a hand-written assumption
    # (local 08-10 and 17-20 commuter hours), NOT derived from the measured
    # curve. On the real M-Lab data the Klang Valley mobile trough is
    # roughly 19:00-00:00 and mornings show no trough at all, so the
    # configured window can be correct-in-timezone yet still a poor match
    # for observed behaviour. Surfaced so the feature is redefined
    # deliberately rather than left as an untested assumption.
    if "is_peak_period" in panel_df.columns:
        peak_mean = panel_df.loc[panel_df["is_peak_period"] == 1, "dl_median"].mean()
        off_mean = panel_df.loc[panel_df["is_peak_period"] == 0, "dl_median"].mean()
        if pd.notna(peak_mean) and pd.notna(off_mean) and peak_mean >= off_mean:
            report.warn(
                "is_peak_period does not match the measured curve: mean dl_median during the "
                "configured peak window is %.1f Mbps vs %.1f off-peak, i.e. 'peak' hours look no "
                "worse than the rest of the day. The window (local 08-10, 17-20) is a commuter-hours "
                "assumption; the measured M-Lab trough for this city is ~19:00-00:00 with no morning "
                "dip. Consider deriving the peak window from the observed curve instead. This is NOT "
                "the timezone bug (validate_diurnal_alignment above covers that) — it is a separate, "
                "milder definitional mismatch.",
                peak_mean, off_mean,
            )


def validate_targets(targets_df: pd.DataFrame, report: ValidationReport) -> None:
    if "target_is_degraded" in targets_df.columns:
        rate = targets_df["target_is_degraded"].mean()
        report.warn(
            "target_is_degraded positive rate is %.2f%% — severe class imbalance; the modelling team "
            "should use class weighting / PR-AUC, not plain accuracy.", 100 * rate,
        ) if rate < 0.02 or rate > 0.5 else None

    for col in ("target_qoe_index",):
        if col in targets_df.columns:
            out_of_range = ~targets_df[col].between(0, 100)
            if out_of_range.any():
                report.error("%d row(s) with %s outside [0, 100].", out_of_range.sum(), col)

    # Regression guard for the weight-sum bug: with correctly-summing
    # QOE_INDEX_WEIGHTS, at least some rows with excellent dl/ul/rtt should
    # be able to approach the top of the range. If the observed maximum is
    # suspiciously far below 100, the weights have silently drifted again.
    if "target_qoe_index" in targets_df.columns and len(targets_df):
        observed_max = targets_df["target_qoe_index"].max()
        if observed_max < 90.0:
            report.warn(
                "target_qoe_index's observed maximum is only %.1f (expected values approaching 100 to "
                "be reachable) — check that config.QOE_INDEX_WEIGHTS['download'] + ['upload'] still sum "
                "to 1.0; a value well below 100 here previously indicated a silent weight-sum bug.",
                observed_max,
            )

    # Regression guard for the is_degraded / degradation_class inconsistency
    # bug: both are now derived from the same thresholds by construction, so
    # this should always pass — kept as a hard error specifically because a
    # future edit to qoe_target.py re-introducing separate thresholds is the
    # most likely way this regresses silently.
    if {"target_is_degraded", "target_degradation_class"} <= set(targets_df.columns):
        mismatch = (targets_df["target_is_degraded"] == 1) != (targets_df["target_degradation_class"] >= 1)
        if mismatch.any():
            report.error(
                "%d row(s) where target_is_degraded disagrees with target_degradation_class — these "
                "must be derived from the same threshold source (see qoe_target.compute_qoe_targets).",
                mismatch.sum(),
            )

    next_hour_valid_cols = [c for c in targets_df.columns if c.endswith("_valid") and c.startswith("target_next_")]
    for col in next_hour_valid_cols:
        invalid_frac = 1.0 - targets_df[col].mean()
        report.warn(
            "%.3f%% of rows have %s == 0 (end-of-series warm-up, no future observation) — these must be "
            "dropped before training on the corresponding next-hour target, not treated as label 0.",
            100 * invalid_frac, col,
        )

    validate_target_anchor_correlation(targets_df, report)


def validate_derived_layer_passthrough(static_df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Guard against the single most repeated bug in this project: a layer is
    extracted correctly, then a merge that names the columns to KEEP
    silently drops the rest.

    It has happened three times:
      - ESA WorldCover: 11 classes extracted, 7 forwarded. Mangroves (up
        to 100% of a cell, >10% in 305 coastal cells) vanished, and 480
        cells ended up with land-cover percentages summing to under 99 —
        reading as "no data" where the truth was "entirely mangrove".
      - VIIRS: 6 monthly composites extracted, only their MEAN forwarded.
        No trace at all downstream; the loss was undetectable from the
        outputs.
      - The original VIIRS/WorldCover/Sunway merge: 24 real columns
        reached cell_static_features and nothing else.

    None of these raised an error. Sums still looked plausible, row counts
    matched, and the modelling team simply never learned the data existed.
    This check compares every registered derived layer against the static
    feature table and fails the build on any column that did not make it,
    so the failure mode becomes loud instead of silent.

    A column deliberately not forwarded belongs in DERIVED_LAYER_DROP_OK
    with a reason — an explicit, reviewable decision rather than an
    accident of a stale keep-list.
    """
    from config import DERIVED_LAYERS

    # Columns intentionally not carried into the static table, with why.
    DERIVED_LAYER_DROP_OK = {
        # Per-month VIIRS composites: superseded by viirs_avg_rad_mean plus
        # the viirs_monthly_std / _cv stability columns, which is the whole
        # of their information at this resolution.
        "viirs_avg_rad_2024306", "viirs_avg_rad_2024336", "viirs_avg_rad_2025001",
        "viirs_avg_rad_2025032", "viirs_avg_rad_2025060", "viirs_avg_rad_2025091",
    }

    present = set(static_df.columns)
    for key, path in DERIVED_LAYERS.items():
        if not path.exists():
            continue
        try:
            layer_cols = set(pd.read_parquet(path).columns) - {"h3_index"}
        except Exception as exc:
            report.warn("Could not read derived layer %s for the passthrough check (%s).", path.name, exc)
            continue

        lost = sorted(layer_cols - present - DERIVED_LAYER_DROP_OK)
        if lost:
            report.error(
                "Derived layer '%s' (%s) produces %d column(s) that never reach "
                "cell_static_features: %s. This is the extract-then-silently-drop bug that has "
                "already cost this project the mangrove land-cover class and the monthly VIIRS "
                "composites. Either forward them (merges should exclude, not include) or add them "
                "to DERIVED_LAYER_DROP_OK with a reason.",
                key, path.name, len(lost), lost,
            )
        else:
            logger.info(
                "Derived layer '%s': all %d produced column(s) reach the static features.",
                key, len(layer_cols),
            )


def validate_derived_layer_freshness(report: ValidationReport) -> None:
    """
    Fail the build when a derived layer is OLDER than the extraction it was
    built from.

    This caught a real bug. After the VIIRS backfill grew the extraction
    from 6 monthly composites to 43, `merge_layers.py` was never re-run, so
    static_new_layers_by_h3.parquet still carried 6-month averages and
    stability columns while viirs_by_h3.parquet held 43 months. Nothing
    else would have noticed: the column names were identical, the
    passthrough guard passed, row counts matched, and the values were
    plausible — just computed from stale input. Only the file timestamps
    and a value comparison revealed it.

    Timestamps are a blunt instrument, but the failure mode here is
    exactly "input changed, derivative did not", which mtime detects
    reliably enough to be worth a hard stop.
    """
    from config import DERIVED_LAYERS, PROCESSED_DIR

    # derived layer -> the extraction outputs it is built from
    UPSTREAM = {
        "static_new_layers": [PROCESSED_DIR / "viirs_by_h3.parquet",
                               PROCESSED_DIR / "worldcover_by_h3.parquet"],
        "sunway_by_h3": [PROCESSED_DIR / "sunway_real_points.parquet"],
    }
    for key, sources in UPSTREAM.items():
        layer = DERIVED_LAYERS.get(key)
        if layer is None or not layer.exists():
            continue
        layer_mtime = layer.stat().st_mtime
        stale = [s.name for s in sources if s.exists() and s.stat().st_mtime > layer_mtime]
        if stale:
            report.error(
                "Derived layer '%s' (%s) is OLDER than the extraction(s) it is built from: %s. "
                "It is carrying values computed from superseded input — the column names and row "
                "counts will look correct, so nothing else catches this. Re-run the merge step "
                "(scripts/extract/merge_layers.py for static_new_layers, "
                "src.data_preprocessing.ingest_sunway_real for sunway) before trusting the outputs.",
                key, layer.name, stale,
            )
        else:
            logger.info("Derived layer '%s' is newer than its inputs — not stale.", key)


def validate_relative_targets(panel_df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Measure — rather than assert — how much the relative target actually
    moves over time.

    qoe_target.compute_relative_targets documents that while ONE shared
    diurnal curve drives the panel, the cross-cell ranking cannot change
    through the week (a common positive factor does not reorder cells), so
    target_is_below_area_norm is time-invariant by construction. That claim
    should be checked against the shipped data every run, not taken on
    trust, because it stops being true the moment real per-district curves
    are wired in — and at that point this warning disappearing is the
    signal that the panel finally carries spatio-temporal information.
    """
    col = "target_is_below_area_norm"
    anchor_col = "qoe_level_is_ookla_anchored"
    if col not in panel_df.columns or "h3_index" not in panel_df.columns:
        return

    per_cell_unique = panel_df.groupby("h3_index")[col].nunique()
    n_cells = len(per_cell_unique)
    if n_cells == 0:
        return
    n_varying = int((per_cell_unique > 1).sum())
    frac_varying = n_varying / n_cells

    # Aggregate time-variation is NOT evidence of real signal, and reporting
    # it alone was actively misleading in an earlier version of this check.
    # Unanchored cells carry a np.random.normal term in their level, which
    # reshuffles the cross-cell ranking every hour; anchored cells are
    # deterministic and cannot move. So a healthy-looking aggregate figure
    # can be almost entirely injected noise. Always split by anchor status.
    anchored_frac = unanchored_frac = None
    if anchor_col in panel_df.columns:
        by_anchor = (
            panel_df.groupby(["h3_index", anchor_col])[col].nunique().reset_index()
        )
        for is_anchored, group in by_anchor.groupby(anchor_col):
            frac = float((group[col] > 1).mean())
            if bool(is_anchored):
                anchored_frac = frac
            else:
                unanchored_frac = frac

    if anchored_frac is not None and unanchored_frac is not None and unanchored_frac > 0:
        ratio = unanchored_frac / max(anchored_frac, 1e-9)
        if anchored_frac < 0.05 and ratio > 5.0:
            report.warn(
                "Relative target '%s' appears to vary over time for %.1f%% of cells overall, but "
                "that movement is SIMULATION NOISE, not signal: only %.1f%% of Ookla-ANCHORED "
                "cells ever change state, versus %.1f%% of UNANCHORED cells. Anchored cells are "
                "deterministic (level x one shared curve), so a common positive factor cannot "
                "reorder them; unanchored cells carry a np.random.normal term in their level that "
                "reshuffles the ranking every hour. Do not read the aggregate figure as evidence "
                "that the panel carries spatio-temporal information — it does not. This resolves "
                "only when the shape itself varies spatially (real per-district curves) or when a "
                "genuinely time-varying real source replaces the panel (e.g. multi-quarter Ookla, "
                "which gives real per-cell change without needing an hourly source).",
                col, 100 * frac_varying, 100 * anchored_frac, 100 * unanchored_frac,
            )
            return

    if frac_varying < 0.01:
        report.warn(
            "Relative target '%s' is TIME-INVARIANT for %.1f%% of cells (%d of %d cells never "
            "change state across the whole week). This is the expected, documented consequence of "
            "driving every cell with ONE shared diurnal curve: a common positive factor cannot "
            "reorder cells, so 'below the area norm' collapses to a static property of the cell's "
            "level. The column is still the correct definition and is exported so the target "
            "contract is stable — but on THIS panel it carries spatial information only, no "
            "temporal information.",
            col, 100 * (1 - frac_varying), n_cells - n_varying, n_cells,
        )
    else:
        logger.info(
            "Relative target '%s' varies over time for %d of %d cells (%.1f%%), and that variation "
            "is NOT concentrated in noise-driven unanchored cells (anchored %.1f%% vs unanchored "
            "%.1f%%) — the panel carries genuine spatio-temporal information.",
            col, n_varying, n_cells, 100 * frac_varying,
            100 * (anchored_frac or 0), 100 * (unanchored_frac or 0),
        )


def validate_target_anchor_correlation(targets_df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Class-distribution audit against Ookla-anchor status.

    On the SIMULATED hourly panel, cells with no real Ookla quarterly
    anchor (qoe_level_is_ookla_anchored == False) generate their hourly
    dl/rtt purely from a population/POI formula (see
    data_downloader.build_dynamic_panel) that, at the currently configured
    QOE_THRESHOLDS, may never cross the degradation threshold. If nearly
    all (or literally all) positive target_is_degraded_next_1h labels come
    from anchored cells, then qoe_level_is_ookla_anchored — or any feature
    correlated with it — is a near-trivial predictor of the target: a
    model could score a high PR-AUC by learning "is this cell anchored"
    rather than learning a genuine degradation pattern. This is a
    structural property of how the simulation is built, not something a
    training-time threshold tweak fixes — the two real options are (a)
    train/evaluate on anchored cells only, or (b) report metrics split by
    anchor status so the imbalance isn't hidden inside one aggregate
    number.
    """
    target_col = "target_is_degraded_next_1h"
    anchor_col = "qoe_level_is_ookla_anchored"
    valid_col = "target_next_1h_valid"
    if not {target_col, anchor_col} <= set(targets_df.columns):
        return

    df = targets_df
    if valid_col in df.columns:
        df = df[df[valid_col] == 1]
    if df.empty:
        return

    rates = df.groupby(anchor_col)[target_col].mean()
    counts = df.groupby(anchor_col)[target_col].sum()
    anchored_rate = float(rates.get(True, 0.0))
    unanchored_rate = float(rates.get(False, 0.0))
    anchored_positives = int(counts.get(True, 0))
    unanchored_positives = int(counts.get(False, 0))
    total_positives = anchored_positives + unanchored_positives

    if total_positives == 0:
        return

    if unanchored_positives == 0 and anchored_positives > 0:
        report.warn(
            "%s has a 0%% positive rate for cells with NO real Ookla anchor (%s == False) vs %.2f%% for "
            "anchored cells — ALL %d positive labels in the panel come from the %.1f%% of cells that are "
            "Ookla-anchored. %s (or anything correlated with it, e.g. ookla_quarter_test_count) is "
            "therefore a near-perfect trivial predictor of this target on the SIMULATED hourly panel. "
            "Before training on cell_time_dataset.parquet: either restrict training/evaluation to "
            "anchored cells only, or report metrics split by %s so a high aggregate PR-AUC can't be "
            "mistaken for genuine degradation prediction. This is a structural property of the "
            "simulation formula, not a threshold-tuning bug.",
            target_col, anchor_col, anchored_rate * 100, total_positives,
            100 * df[anchor_col].mean(), anchor_col, anchor_col,
        )
    elif unanchored_rate > 0 and anchored_rate / max(unanchored_rate, 1e-9) > 5.0:
        report.warn(
            "%s positive rate is %.2fx higher for Ookla-anchored cells (%.2f%%) than for unanchored "
            "cells (%.2f%%) — a large enough gap that anchor status carries real predictive signal "
            "about the label independent of genuine demand/QoE patterns. Consider reporting metrics "
            "split by %s.",
            target_col, anchored_rate / max(unanchored_rate, 1e-9), anchored_rate * 100,
            unanchored_rate * 100, anchor_col,
        )


def run_full_validation(grid_gdf, static_df, panel_df, targets_df=None) -> ValidationReport:
    report = ValidationReport()
    validate_grid_geometry(grid_gdf, report)
    validate_static_features(static_df, report)
    validate_derived_layer_passthrough(static_df, report)
    validate_derived_layer_freshness(report)
    validate_panel(panel_df, expected_cells=len(grid_gdf), report=report)
    validate_diurnal_alignment(panel_df, report)
    validate_relative_targets(panel_df, report)
    if targets_df is not None:
        validate_targets(targets_df, report)

    logger.info(
        "Validation complete: %d error(s), %d warning(s).", len(report.errors), len(report.warnings)
    )
    report.raise_if_errors()
    return report
