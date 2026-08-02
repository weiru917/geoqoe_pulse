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


def _minmax(series: pd.Series) -> pd.Series:
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return pd.Series(0.5, index=series.index)
    return (series - min_v) / (max_v - min_v)


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
    panel_df["is_peak_period"] = (
        (panel_df["is_weekend"] == 0)
        & (panel_df["hour_of_day"].between(8, 10) | panel_df["hour_of_day"].between(17, 20))
    ).astype(int)

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

    if "lag_24h_rtt_median" in panel_df.columns:
        historical_qoe_pressure = _minmax(panel_df["lag_24h_rtt_median"].fillna(panel_df["rtt_median"].median()))
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
    combined = (
        (1.0 - w["nearby_demand"]) * panel_df["demand_pressure_index_pre_spatial"]
        + w["nearby_demand"] * _minmax(panel_df[col]) * 100.0
    )
    panel_df["demand_pressure_index"] = _minmax(combined) * 100.0
    return panel_df
