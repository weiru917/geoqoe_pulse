"""
GeoQoE Pulse Spatial Processing & Spatial Lags Module.
Handles H3 topology graph generation, spatial weights matrices, and spatial lag operations.
"""
import numpy as np
import pandas as pd
import h3
from scipy.sparse import csr_matrix

def build_h3_spatial_weights(h3_index_list):
    """
    Construct a row-standardized spatial weights adjacency matrix directly from H3 topology graph.
    Returns: scipy.sparse.csr_matrix and ordered h3 list.
    """
    h3_index_list = list(h3_index_list)
    h3_set = set(h3_index_list)
    h3_to_idx = {cell: i for i, cell in enumerate(h3_index_list)}
    n = len(h3_index_list)
    
    rows = []
    cols = []
    data = []
    
    for i, cell in enumerate(h3_index_list):
        try:
            neighbors = h3.grid_ring(cell, 1)
        except AttributeError:
            neighbors = set(h3.k_ring(cell, 1)) - {cell}
        
        valid_neighbors = [n for n in neighbors if n in h3_set]
        num_neighbors = len(valid_neighbors)
        
        if num_neighbors > 0:
            weight = 1.0 / float(num_neighbors)
            for nbr in valid_neighbors:
                rows.append(i)
                cols.append(h3_to_idx[nbr])
                data.append(weight)
                
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.array(data, dtype=np.float64)
    
    W_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W_sparse, h3_index_list

def compute_vectorized_spatial_lags(panel_df, W_sparse, h3_order, target_cols):
    """
    Calculate spatial lag features (W @ y) across panel data per time step using sparse matrix operations.
    """
    processed_dfs = []
    for ts, group in panel_df.groupby("timestamp"):
        group_indexed = group.set_index("h3_index").reindex(h3_order)
        group_indexed["timestamp"] = ts
        
        for col in target_cols:
            lag_col_name = f"spatial_lag_{col}"
            values = group_indexed[col].fillna(0).values
            lag_values = W_sparse @ values
            group_indexed[lag_col_name] = lag_values
            
        processed_dfs.append(group_indexed.reset_index())

    return pd.concat(processed_dfs, ignore_index=True)

def compute_getis_ord_hotspots(df, W_sparse, col="rtt_median"):
    """
    Calculate Getis-Ord Gi* Z-scores for identifying statistical hotspot clusters.
    """
    df = df.copy()
    values = df[col].values
    mean_val = np.mean(values)
    std_val = np.std(values) + 1e-6
    
    spatial_sum = W_sparse @ values
    z_scores = (spatial_sum - mean_val) / std_val
    
    df[f"getis_ord_gi_zscore_{col}"] = z_scores
    return df
