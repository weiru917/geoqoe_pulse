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
    validate_panel(panel_df, expected_cells=len(grid_gdf), report=report)
    if targets_df is not None:
        validate_targets(targets_df, report)

    logger.info(
        "Validation complete: %d error(s), %d warning(s).", len(report.errors), len(report.warnings)
    )
    report.raise_if_errors()
    return report
