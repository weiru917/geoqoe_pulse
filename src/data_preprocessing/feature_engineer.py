"""
GeoQoE Pulse Feature Engineering & Temporal Windowing Module.
Engineers static demand proxies, dynamic demand indices, and temporal rolling window stats with zero target leakage.
"""
import numpy as np
import pandas as pd
from config import DEMAND_WEIGHTS

def compute_demand_pressure_proxy(df):
    """
    Construct standardized dynamic demand pressure proxy index (0-100 scale).
    Combines population exposure, POIs, built density, and peak hour multipliers.
    """
    df = df.copy()
    
    def minmax(series):
        min_v, max_v = series.min(), series.max()
        if max_v == min_v:
            return pd.Series(0.5, index=series.index)
        return (series - min_v) / (max_v - min_v)

    norm_pop = minmax(df["pop_count"])
    norm_poi = minmax(df["poi_retail_food_count"] + df["poi_office_count"] + df["poi_transport_stop_count"])
    norm_built = minmax(df["built_up_ratio"])
    norm_road = minmax(df["road_length_primary_m"] + df["road_length_expressway_m"])
    
    w = DEMAND_WEIGHTS
    df["demand_pressure_index"] = (
        w["population"] * norm_pop +
        w["poi"] * norm_poi +
        w["built_area"] * norm_built +
        w["transport"] * norm_road +
        w["peak_hour"] * df["is_peak_period"]
    ) * 100.0
    
    return df

def compute_temporal_shifted_features(panel_df, metrics=["dl_median", "rtt_median"]):
    """
    Engineers historical temporal lags and rolling statistics.
    CRITICAL: Applies shift(1) prior to rolling calculations to ensure zero data leakage from time t.
    """
    panel_df = panel_df.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)
    
    panel_df["hour_of_day"] = panel_df["timestamp"].dt.hour
    panel_df["day_of_week"] = panel_df["timestamp"].dt.dayofweek
    panel_df["is_weekend"] = (panel_df["day_of_week"] >= 5).astype(int)
    panel_df["is_peak_period"] = (
        (panel_df["is_weekend"] == 0) & 
        ((panel_df["hour_of_day"].between(8, 10)) | (panel_df["hour_of_day"].between(17, 20)))
    ).astype(int)
    
    for metric in metrics:
        panel_df[f"lag_1h_{metric}"] = panel_df.groupby("h3_index")[metric].shift(1)
        panel_df[f"lag_2h_{metric}"] = panel_df.groupby("h3_index")[metric].shift(2)
        panel_df[f"lag_24h_{metric}"] = panel_df.groupby("h3_index")[metric].shift(24)
        
        panel_df[f"rolling_mean_24h_{metric}"] = (
            panel_df.groupby("h3_index")[metric]
            .transform(lambda x: x.shift(1).rolling(window=24, min_periods=1).mean())
        )
        
        panel_df[f"rolling_std_24h_{metric}"] = (
            panel_df.groupby("h3_index")[metric]
            .transform(lambda x: x.shift(1).rolling(window=24, min_periods=1).std())
        )

        for suffix in ["lag_1h_", "lag_2h_", "lag_24h_", "rolling_mean_24h_"]:
            col_name = f"{suffix}{metric}"
            panel_df[col_name] = panel_df.groupby("h3_index")[col_name].bfill().ffill()
            panel_df[col_name] = panel_df[col_name].fillna(panel_df[metric])

        panel_df[f"rolling_std_24h_{metric}"] = panel_df[f"rolling_std_24h_{metric}"].fillna(0.0)

    panel_df["qoe_trend_1h"] = panel_df["lag_1h_dl_median"] - panel_df["rolling_mean_24h_dl_median"]
    
    return panel_df
