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

    for col in ("target_qoe_score",):
        if col in targets_df.columns:
            out_of_range = ~targets_df[col].between(0, 100)
            if out_of_range.any():
                report.error("%d row(s) with %s outside [0, 100].", out_of_range.sum(), col)


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
