"""
GeoQoE Pulse Feature Engineering & Temporal Windowing Module.

Engineers the demand-pressure index from REAL static GIS features (no more
`pop_count`/`road_length_primary_m` placeholders — see
data_downloader.build_static_features for population_density_per_km2,
poi_*, built_up_ratio, road_density_km_per_km2), and temporal lag/rolling
features with genuinely zero leakage.
"""
import logging

import numpy as np
import pandas as pd

from config import DEMAND_WEIGHTS, LAG_WARMUP_HOURS, HIGH_LEAKAGE_RISK_FEATURES

logger = logging.getLogger(__name__)


# Scaler parameters actually used, recorded so metadata.json can state
# exactly what was fitted and on which rows — a scaler is part of the data
# contract, not an implementation detail.
FITTED_SCALERS = {}

# Fraction of the panel's timestamps used to FIT min/max scalers. Fitting
# on all rows would let the scaler see the test period, which is a real
# (if mild) leakage path under the temporal holdout the README recommends.
SCALER_FIT_FRACTION = 0.80

# How many hours the MEASURED peak window spans. 6 matches the length of
# the configured commuter window (08-10 + 17-20 = 7h) closely enough for
# the two to be compared like for like.
PEAK_WINDOW_HOURS = 6


def _minmax(series: pd.Series, fit_mask: pd.Series = None, name: str = None) -> pd.Series:
    """
    Min-max scale to [0, 1].

    `fit_mask` restricts which rows the min/max are LEARNED from, while
    still transforming every row. Under a temporal holdout, fitting on all
    rows means the scaler has already seen the test period — a genuine
    leakage path, and the last one remaining in this pipeline after the
    lag/rolling features were made strictly causal. Values outside the
    fitted range are clipped to [0, 1] rather than allowed to exceed it,
    so a test-period extreme cannot silently produce an out-of-range
    feature.

    Passing no mask fits on everything, which is correct for genuinely
    static per-cell inputs (there is no temporal split to leak across) and
    is what compute_demand_pressure_base does.
    """
    source = series if fit_mask is None else series[fit_mask]
    min_v, max_v = source.min(), source.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(0.5, index=series.index)
    if name:
        FITTED_SCALERS[name] = {
            "min": float(min_v), "max": float(max_v),
            "fitted_on_rows": int(len(source)), "total_rows": int(len(series)),
            "fitted_on_all_rows": fit_mask is None,
        }
    return ((series - min_v) / (max_v - min_v)).clip(0.0, 1.0)


def training_window_mask(panel_df: pd.DataFrame,
                          fraction: float = SCALER_FIT_FRACTION) -> pd.Series:
    """
    Boolean mask selecting the earliest `fraction` of the panel's DISTINCT
    timestamps — the rows a temporal-holdout model would be allowed to
    train on. Used to fit scalers without touching the held-out tail.
    """
    if "timestamp" not in panel_df.columns:
        return pd.Series(True, index=panel_df.index)
    stamps = np.sort(panel_df["timestamp"].unique())
    if len(stamps) == 0:
        return pd.Series(True, index=panel_df.index)
    cutoff = stamps[min(int(len(stamps) * fraction), len(stamps) - 1)]
    return panel_df["timestamp"] <= cutoff


def compute_temporal_shifted_features(panel_df: pd.DataFrame,
                                       metrics=("dl_median", "rtt_median")) -> pd.DataFrame:
    """
    Historical temporal lags and rolling statistics, causal by construction
    (`shift(1)` applied before every rolling window) AND leakage-free at the
    series boundary (no bfill/current-value fallback — see module docstring).
    """
    panel_df = panel_df.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)

    panel_df["hour_of_day"] = panel_df["timestamp"].dt.hour
    panel_df["day_of_week"] = panel_df["timestamp"].dt.dayofweek
    panel_df["is_weekend"] = (panel_df["day_of_week"] >= 5).astype(int)
    # The commuter-hours ASSUMPTION, always exported under its own name so
    # the original definition stays available and comparable.
    panel_df["is_commuter_period"] = (
        (panel_df["is_weekend"] == 0)
        & (panel_df["hour_of_day"].between(8, 10) | panel_df["hour_of_day"].between(17, 20))
    ).astype(int)

    # is_peak_period is now DERIVED FROM THE DATA by default. The commuter
    # window was never measured — it was assumed, and the real M-Lab curve
    # contradicts it: mean throughput during 08-10 and 17-20 is no worse
    # than the rest of the day, because this city's mobile trough is
    # 19:00-00:00. A feature that names the wrong hours as "peak" quietly
    # corrupts every peak-vs-offpeak comparison built on it.
    from config import PEAK_PERIOD_DEFINITION
    if PEAK_PERIOD_DEFINITION == "measured" and "dl_median" in panel_df.columns:
        hourly_mean = panel_df.groupby("hour_of_day")["dl_median"].mean()
        peak_hours = sorted(hourly_mean.nsmallest(PEAK_WINDOW_HOURS).index.tolist())
        panel_df["is_peak_period"] = panel_df["hour_of_day"].isin(peak_hours).astype(int)
        panel_df["is_measured_peak_period"] = panel_df["is_peak_period"]
        logger.info(
            "is_peak_period DERIVED from the realised curve: hours %s local are the %d "
            "lowest-throughput hours. The previous hardcoded commuter window (08-10, 17-20) "
            "is retained separately as is_commuter_period — the measurement does not support "
            "it as a peak.", peak_hours, PEAK_WINDOW_HOURS,
        )
    else:
        panel_df["is_peak_period"] = panel_df["is_commuter_period"]
        panel_df["is_measured_peak_period"] = panel_df["is_peak_period"]
        logger.warning(
            "is_peak_period is using the COMMUTER ASSUMPTION (config."
            "PEAK_PERIOD_DEFINITION=%r). The real diurnal curve does not support that "
            "window as a peak for this city.", PEAK_PERIOD_DEFINITION,
        )

    # Position of each row within its own cell's time series — used to build
    # validity flags without any fill-based workaround.
    panel_df["_cell_row_idx"] = panel_df.groupby("h3_index").cumcount()

    for metric in metrics:
        grp = panel_df.groupby("h3_index")[metric]
        panel_df[f"lag_1h_{metric}"] = grp.shift(1)
        panel_df[f"lag_2h_{metric}"] = grp.shift(2)
        panel_df[f"lag_24h_{metric}"] = grp.shift(24)

        panel_df[f"rolling_mean_24h_{metric}"] = (
            panel_df.groupby("h3_index")[metric]
            .transform(lambda x: x.shift(1).rolling(window=24, min_periods=1).mean())
        )
        panel_df[f"rolling_std_24h_{metric}"] = (
            panel_df.groupby("h3_index")[metric]
            .transform(lambda x: x.shift(1).rolling(window=24, min_periods=1).std())
        )
        # std of a single point is NaN by definition (not a leakage issue,
        # just an undefined statistic) — 0.0 is the correct fill here, not a
        # leak, since it asserts "no observed variance yet" rather than
        # borrowing a real value from elsewhere.
        panel_df[f"rolling_std_24h_{metric}"] = panel_df[f"rolling_std_24h_{metric}"].fillna(0.0)

        # Validity flags: 1 once a row has enough real history behind it for
        # that specific lag depth, 0 during warm-up (where the corresponding
        # lag/rolling column is a real, honest NaN).
        panel_df[f"lag_1h_{metric}_valid"] = (panel_df["_cell_row_idx"] >= 1).astype(int)
        panel_df[f"lag_2h_{metric}_valid"] = (panel_df["_cell_row_idx"] >= 2).astype(int)
        panel_df[f"lag_24h_{metric}_valid"] = (panel_df["_cell_row_idx"] >= 24).astype(int)
        panel_df[f"rolling_mean_24h_{metric}_valid"] = (panel_df["_cell_row_idx"] >= 1).astype(int)

    panel_df["qoe_trend_1h"] = panel_df["lag_1h_dl_median"] - panel_df["rolling_mean_24h_dl_median"]
    panel_df["in_warmup_window"] = (panel_df["_cell_row_idx"] < LAG_WARMUP_HOURS).astype(int)
    panel_df = panel_df.drop(columns=["_cell_row_idx"])

    n_warmup = int(panel_df["in_warmup_window"].sum())
    logger.info(
        "Temporal features: %d/%d rows (%.1f%%) are within each cell's %dh warm-up window — "
        "their *_valid flags are 0 and the corresponding lag columns are real NaN, not a filled value.",
        n_warmup, len(panel_df), 100 * n_warmup / len(panel_df), LAG_WARMUP_HOURS,
    )
    return panel_df


def compute_demand_pressure_base(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Static (non-time-varying) demand pressure base index, built from REAL
    GIS features only: population density, POI density (retail+office+
    transport), built-up ratio, and road density. 0-100 min-max scale.

    Weights are documented in config.DEMAND_WEIGHTS. The `historical_qoe`
    and `nearby_demand` weighted components are intentionally NOT part of
    this static base — they require, respectively, temporal lag features
    and a spatial-lag pass that don't exist yet at this point in the
    pipeline. This function's weights are renormalized to sum to 1 across
    only the four inputs it actually has; see
    `apply_dynamic_demand_adjustments` for where the other two are folded
    in downstream, using their own configured weights.
    """
    static_df = static_df.copy()
    w = DEMAND_WEIGHTS
    static_component_weights = {
        k: w[k] for k in ("population", "poi", "built_area", "transport")
    }
    total = sum(static_component_weights.values())
    static_component_weights = {k: v / total for k, v in static_component_weights.items()}

    norm_pop = _minmax(static_df["population_density_per_km2"])
    norm_poi = _minmax(
        static_df["poi_retail_food_count"] + static_df["poi_office_count"] + static_df["poi_transport_stop_count"]
    )
    norm_built = _minmax(static_df["built_up_ratio"])
    norm_road = _minmax(static_df["road_density_km_per_km2"])

    static_df["demand_pressure_base"] = 100.0 * (
        static_component_weights["population"] * norm_pop
        + static_component_weights["poi"] * norm_poi
        + static_component_weights["built_area"] * norm_built
        + static_component_weights["transport"] * norm_road
    )
    logger.info("Demand pressure base computed from real static features (weights=%s).", static_component_weights)
    return static_df[["h3_index", "demand_pressure_base"]]


def apply_dynamic_demand_adjustments(panel_df: pd.DataFrame, demand_base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Folds the static base into the panel, then adds the `historical_qoe`
    component (config.DEMAND_WEIGHTS["historical_qoe"]) using the already
    causal `lag_24h_rtt_median` column — degraded recent history raises
    demand pressure. `nearby_demand` (config.DEMAND_WEIGHTS["nearby_demand"])
    is intentionally left for the caller to fold in via
    `finalize_demand_pressure_index` AFTER spatial_processor computes the
    spatial lag of `demand_pressure_index_pre_spatial` — this function alone
    cannot see neighbor values.

    LEAKAGE-RISK CAVEAT (see config.HIGH_LEAKAGE_RISK_FEATURES): while the
    hourly panel's rtt_median is a SIMULATED value generated from
    population/POI (data_downloader.build_dynamic_panel), the
    historical_qoe component below is mechanically correlated with the same
    population/POI inputs that already drive demand_pressure_base — a
    circular path, not an independently discovered relationship. This is
    harmless once real (non-simulated) QoE data replaces the panel, but
    until then, prefer `demand_pressure_base` (returned by
    compute_demand_pressure_base, real inputs only) over the full
    `demand_pressure_index` for anything presented as a validated finding.
    """
    logger.warning(
        "demand_pressure_index's historical_qoe component is circular with the SIMULATED QoE panel — "
        "%s Use demand_pressure_base for a non-circular real-data-only feature.",
        HIGH_LEAKAGE_RISK_FEATURES.get("demand_pressure_index", ""),
    )
    panel_df = panel_df.merge(demand_base_df, on="h3_index", how="left")
    w = DEMAND_WEIGHTS

    peak_multiplier = 1.0 + 0.5 * panel_df["is_peak_period"]

    fit_mask = training_window_mask(panel_df)
    if "lag_24h_rtt_median" in panel_df.columns:
        # Warm-up fill uses the TRAINING window's median of the lag column
        # itself — not the current-hour rtt_median across the whole panel,
        # which both leaked the test period and filled a lag feature from a
        # same-hour column.
        warmup_fill = panel_df.loc[fit_mask, "lag_24h_rtt_median"].median()
        historical_qoe_pressure = _minmax(
            panel_df["lag_24h_rtt_median"].fillna(warmup_fill),
            fit_mask=fit_mask, name="historical_qoe_pressure",
        )
    else:
        historical_qoe_pressure = pd.Series(0.0, index=panel_df.index)

    non_spatial_weight = 1.0 - w["nearby_demand"]
    base_share = (w["population"] + w["poi"] + w["built_area"] + w["transport"]) / non_spatial_weight
    hist_share = w["historical_qoe"] / non_spatial_weight

    panel_df["demand_pressure_index_pre_spatial"] = np.clip(
        (base_share * (panel_df["demand_pressure_base"] / 100.0) + hist_share * historical_qoe_pressure)
        * peak_multiplier
        * 100.0,
        0, 200,  # peak_multiplier can push above 100; capped downstream in finalize step
    )
    return panel_df


def finalize_demand_pressure_index(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Folds in the `nearby_demand` weighted component using
    `spatial_lag_demand_pressure_index_pre_spatial` (must already be present
    — computed by spatial_processor.compute_vectorized_spatial_lags on
    `demand_pressure_index_pre_spatial`), producing the final 0-100
    `demand_pressure_index`.
    """
    col = "spatial_lag_demand_pressure_index_pre_spatial"
    if col not in panel_df.columns:
        raise KeyError(f"{col} not found — run the spatial lag pass on demand_pressure_index_pre_spatial first.")

    w = DEMAND_WEIGHTS
    fit_mask = training_window_mask(panel_df)
    combined = (
        (1.0 - w["nearby_demand"]) * panel_df["demand_pressure_index_pre_spatial"]
        + w["nearby_demand"] * _minmax(panel_df[col], fit_mask=fit_mask,
                                        name="nearby_demand_spatial_lag") * 100.0
    )
    panel_df["demand_pressure_index"] = _minmax(
        combined, fit_mask=fit_mask, name="demand_pressure_index"
    ) * 100.0
    return panel_df
