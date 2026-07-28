"""
GeoQoE Pulse Master Execution Pipeline.
Executes data preprocessing, H3 spatial feature engineering, saves intermediate checkpoints to data/processed/ and data/features/,
and exports final modeling Parquet outputs to data/outputs/.
"""
import sys
from pathlib import Path

# Add src to sys.path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

import pandas as pd
from config import OUTPUT_DIR, PROCESSED_DIR, FEATURES_DIR, H3_PRIMARY_RESOLUTION
from src.data_preprocessing.data_downloader import generate_h3_study_grid, load_real_or_baseline_datasets
from src.data_preprocessing.spatial_processor import build_h3_spatial_weights, compute_vectorized_spatial_lags, compute_getis_ord_hotspots
from src.data_preprocessing.feature_engineer import compute_temporal_shifted_features, compute_demand_pressure_proxy
from src.data_preprocessing.qoe_target import compute_composite_qoe_targets

def run_master_pipeline():
    print("=" * 70)
    print("GeoQoE Pulse — Master Data Engineering & GIS Pipeline Execution")
    print("=" * 70)
    
    # 1. Generate H3 Spatial Framework & Save Processed Grid Checkpoint
    print("\n[Step 1/6] Building H3 Spatial Hexagon Grid (Resolution 8)...")
    grid_df = generate_h3_study_grid(res=H3_PRIMARY_RESOLUTION)
    print(f"-> Generated {len(grid_df)} H3 hexagonal cells for Klang Valley study area.")
    
    # Save intermediate processed grid checkpoint
    grid_checkpoint = PROCESSED_DIR / "klang_valley_h3_grid_res8.parquet"
    grid_df.to_parquet(grid_checkpoint, index=False)
    print(f"-> Saved intermediate spatial grid checkpoint to: {grid_checkpoint.name}")
    
    # 2. Ingest Spatial Datasets & Dynamic QoE Observations
    print("\n[Step 2/6] Ingesting Spatial Datasets & Dynamic QoE Observations...")
    static_df, dynamic_df = load_real_or_baseline_datasets(grid_df, days=7)
    print(f"-> Static features loaded: {len(static_df)} cells.")
    print(f"-> Dynamic panel observations loaded: {len(dynamic_df)} rows ({dynamic_df['timestamp'].nunique()} hourly time steps).")
    
    # Save intermediate static feature checkpoint
    static_checkpoint = FEATURES_DIR / "static_spatial_features.parquet"
    static_df.to_parquet(static_checkpoint, index=False)
    print(f"-> Saved static spatial features checkpoint to: {static_checkpoint.name}")
    
    # 3. Spatial Weights & Spatial Lag Feature Engineering
    print("\n[Step 3/6] Building H3 Spatial Topology & Computing Spatial Lag Features (W @ y)...")
    h3_list = grid_df["h3_index"].tolist()
    W_sparse, h3_order = build_h3_spatial_weights(h3_list)
    
    panel = dynamic_df.merge(static_df, on="h3_index", how="left")
    spatial_target_cols = ["dl_median", "rtt_median", "pop_count"]
    panel = compute_vectorized_spatial_lags(panel, W_sparse, h3_order, spatial_target_cols)
    print(f"-> Generated spatial lag features: {[f'spatial_lag_{c}' for c in spatial_target_cols]}")
    
    # 4. Temporal Features & Dynamic Demand Proxies
    print("\n[Step 4/6] Computing Leakage-Free Temporal Lags & Dynamic Demand Pressure Proxies...")
    panel = compute_temporal_shifted_features(panel, metrics=["dl_median", "rtt_median"])
    panel = compute_demand_pressure_proxy(panel)
    panel = compute_vectorized_spatial_lags(panel, W_sparse, h3_order, ["demand_pressure_index"])
    
    # Getis-Ord Gi* Hotspot Z-scores
    gi_dfs = []
    for ts, group in panel.groupby("timestamp"):
        group_scored = compute_getis_ord_hotspots(group, W_sparse, col="rtt_median")
        group_scored["timestamp"] = ts
        gi_dfs.append(group_scored)
    panel = pd.concat(gi_dfs, ignore_index=True)
    
    # Save intermediate feature matrix checkpoint
    features_checkpoint = FEATURES_DIR / "engineered_panel_features.parquet"
    panel.to_parquet(features_checkpoint, index=False)
    print(f"-> Saved engineered panel features checkpoint to: {features_checkpoint.name}")
    
    # 5. QoE Composite Target Calculation
    print("\n[Step 5/6] Formulating Composite ITU QoE Scores, MOS, and Categorical Degradation Targets...")
    panel = compute_composite_qoe_targets(panel)
    
    # 6. Quality Control & Dataset Export
    print("\n[Step 6/6] Executing Quality Control Audits & Exporting Model-Ready Datasets...")
    feature_cols = [
        "pop_count", "built_up_ratio", "poi_retail_food_count", "poi_office_count",
        "poi_transport_stop_count", "road_length_primary_m", "dist_to_nearest_tower_m",
        "hour_of_day", "day_of_week", "is_weekend", "is_peak_period", "demand_pressure_index",
        "spatial_lag_dl_median", "spatial_lag_rtt_median", "spatial_lag_demand_pressure_index",
        "lag_1h_dl_median", "lag_24h_dl_median", "rolling_mean_24h_dl_median", "rolling_std_24h_dl_median",
        "getis_ord_gi_zscore_rtt_median", "qoe_confidence_score"
    ]
    
    null_counts = panel[feature_cols].isnull().sum().sum()
    print(f"-> Quality Assertion: Total null values across engineered ML features = {null_counts}")
    assert null_counts == 0, "Error: Unexpected NaN values detected in final feature matrix!"
    
    # Export Final Outputs
    output_parquet_path = OUTPUT_DIR / "cell_time_dataset.parquet"
    static_parquet_path = OUTPUT_DIR / "cell_static_features.parquet"
    targets_parquet_path = OUTPUT_DIR / "qoe_targets.parquet"
    
    panel.to_parquet(output_parquet_path, engine="pyarrow", compression="zstd", index=False)
    static_df.to_parquet(static_parquet_path, engine="pyarrow", compression="zstd", index=False)
    
    target_cols = ["h3_index", "timestamp", "target_qoe_score", "target_mos", "target_is_degraded", "target_degradation_class"]
    panel[target_cols].to_parquet(targets_parquet_path, engine="pyarrow", compression="zstd", index=False)
    
    # Generate Feature Dictionary Excel Table
    dict_records = []
    for col in panel.columns:
        is_target = "target" in col
        dict_records.append({
            "Feature Name": col,
            "Data Type": str(panel[col].dtype),
            "Sample Value": str(panel[col].iloc[0]),
            "Feature Category": "Target Variable" if is_target else ("Spatial/Demographic" if col in static_df.columns else "Dynamic/Lag"),
            "Suitable for ML?": "No (Target)" if is_target else ("Key ID" if col in ["h3_index", "timestamp"] else "Yes")
        })
    dict_df = pd.DataFrame(dict_records)
    dict_excel_path = OUTPUT_DIR / "feature_dictionary.xlsx"
    dict_df.to_excel(dict_excel_path, index=False)
    
    print("\n" + "=" * 70)
    print("SUCCESS: GeoQoE Pulse Master Pipeline Execution Complete!")
    print(f"-> Saved master panel dataset to: {output_parquet_path}")
    print(f"-> Saved static features to:      {static_parquet_path}")
    print(f"-> Saved QoE target variables to: {targets_parquet_path}")
    print(f"-> Saved Feature Dictionary to:   {dict_excel_path}")
    print(f"-> Final Table Shape: {panel.shape[0]} rows x {panel.shape[1]} columns.")
    print("=" * 70)

if __name__ == "__main__":
    run_master_pipeline()
