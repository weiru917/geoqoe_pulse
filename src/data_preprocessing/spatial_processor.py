"""
GeoQoE Pulse Spatial Processing Module.

Builds H3 grid-ring adjacency as a proper libpysal spatial-weights object,
computes vectorized spatial lag / rolling / variance features across the
full panel, and runs statistically validated hotspot detection via
`esda.G_Local` (Getis-Ord Gi*) and `esda.Moran_Local` (Local Moran's I)
"""
import logging

import h3
import libpysal
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran_Local
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# Local Moran's I quadrant codes (esda convention)
MORAN_QUADRANT_LABELS = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def build_h3_spatial_weights(h3_index_list):
    """
    Construct H3 grid-ring (1-ring) adjacency as a real `libpysal.weights.W`
    object, so downstream hotspot statistics (esda) get a properly-typed
    spatial weights matrix rather than a bare sparse array.

    Returns (w, h3_index_list) where `w` is row-standardized
    (`w.transform = "R"`) and `w.id_order == h3_index_list`.
    """
    h3_index_list = list(h3_index_list)
    h3_set = set(h3_index_list)

    neighbors = {}
    for cell in h3_index_list:
        ring = h3.grid_ring(cell, 1)
        valid = [n for n in ring if n in h3_set]
        # libpysal requires every unit to have at least one neighbor entry;
        # an isolated cell (edge of the study grid with no in-grid ring
        # neighbor) is recorded as an explicit island rather than dropped.
        neighbors[cell] = valid

    w = libpysal.weights.W(neighbors, ids=h3_index_list, silence_warnings=True)
    w.transform = "R"

    n_islands = len(w.islands)
    if n_islands:
        logger.warning(
            "%d H3 cell(s) have no in-grid neighbor (edge-of-study-area islands): %s",
            n_islands, w.islands[:5],
        )

    return w, h3_index_list


def _w_to_sparse(w: "libpysal.weights.W") -> csr_matrix:
    """Row-standardized sparse adjacency matrix in `w.id_order`, for fast
    vectorized (W @ y) spatial-lag computation across an entire panel."""
    sparse, id_order = w.sparse, w.id_order
    assert list(id_order) == list(w.id_order)
    return sparse


def compute_vectorized_spatial_lags(panel_df: pd.DataFrame, w: "libpysal.weights.W",
                                     h3_order, target_cols) -> pd.DataFrame:
    """
    Spatial lag (W @ y) for each column in `target_cols`, computed
    per-timestamp across the full panel via a single sparse matrix multiply
    per timestamp (vectorized — no per-row Python loop).
    """
    W_sparse = _w_to_sparse(w)
    processed = []
    for ts, group in panel_df.groupby("timestamp"):
        group_indexed = group.set_index("h3_index").reindex(h3_order)
        group_indexed["timestamp"] = ts

        for col in target_cols:
            values = group_indexed[col].fillna(0).values
            group_indexed[f"spatial_lag_{col}"] = W_sparse @ values
            # Spatial variance: how much do a cell's neighbors disagree with
            # each other, not just their mean — a cell whose neighbors are
            # all similar has low spatial variance even if the lag is high.
            mean_nbr = W_sparse @ values
            sq_lag = W_sparse @ (values ** 2)
            group_indexed[f"spatial_var_{col}"] = np.maximum(0.0, sq_lag - mean_nbr ** 2)

        processed.append(group_indexed.reset_index())

    return pd.concat(processed, ignore_index=True)


def compute_spatial_rolling_mean(panel_df: pd.DataFrame, w: "libpysal.weights.W",
                                  h3_order, col: str, window: int = 24) -> pd.DataFrame:
    """
    "What have this cell's neighbors looked like recently" — the spatial
    lag of the CAUSAL (shift(1)-then-rolling) temporal mean, not of the
    current-timestep value. This is deliberately built on top of an
    already-lagged column so it inherits the same no-leakage guarantee;
    call this only after feature_engineer.compute_temporal_shifted_features
    has produced `rolling_mean_{window}h_{col}`.
    """
    source_col = f"rolling_mean_{window}h_{col}"
    if source_col not in panel_df.columns:
        raise KeyError(
            f"{source_col} not found — run compute_temporal_shifted_features first; "
            "spatial rolling mean must be built on a causal (already-lagged) column."
        )
    W_sparse = _w_to_sparse(w)
    out = []
    for ts, group in panel_df.groupby("timestamp"):
        group_indexed = group.set_index("h3_index").reindex(h3_order)
        group_indexed["timestamp"] = ts
        values = group_indexed[source_col].fillna(0).values
        group_indexed[f"spatial_rolling_mean_{window}h_{col}"] = W_sparse @ values
        out.append(group_indexed.reset_index())
    return pd.concat(out, ignore_index=True)


def compute_hotspots(panel_df: pd.DataFrame, w: "libpysal.weights.W", h3_order,
                      col: str = "rtt_median", permutations: int = 99,
                      seed: int = 42) -> pd.DataFrame:
    """
    Statistically validated Getis-Ord Gi* hotspot detection via
    `esda.G_Local` (star=True includes the focal cell itself, the correct
    Gi* — as opposed to Gi — definition). Computed per-timestamp.

    Adds three columns: `gi_star_zscore_{col}`, `gi_star_pvalue_{col}`, and
    `hotspot_category_{col}` classified into the standard ArcGIS-style
    confidence bands (99%/95%/90% hot, 99%/95%/90% cold, not significant).
    """
    out = []
    for ts, group in panel_df.groupby("timestamp"):
        group_indexed = group.set_index("h3_index").reindex(h3_order)
        values = group_indexed[col].fillna(group_indexed[col].mean()).values

        g = G_Local(values, w, transform="B", star=True, permutations=permutations, seed=seed)
        z = g.Zs
        p = g.p_sim if permutations > 0 else g.p_norm

        category = np.full(len(z), "not_significant", dtype=object)
        category[(z > 0) & (p <= 0.01)] = "hot_99"
        category[(z > 0) & (p <= 0.05) & (p > 0.01)] = "hot_95"
        category[(z > 0) & (p <= 0.10) & (p > 0.05)] = "hot_90"
        category[(z < 0) & (p <= 0.01)] = "cold_99"
        category[(z < 0) & (p <= 0.05) & (p > 0.01)] = "cold_95"
        category[(z < 0) & (p <= 0.10) & (p > 0.05)] = "cold_90"

        group_indexed["timestamp"] = ts
        group_indexed[f"gi_star_zscore_{col}"] = z
        group_indexed[f"gi_star_pvalue_{col}"] = p
        group_indexed[f"hotspot_category_{col}"] = category
        out.append(group_indexed.reset_index())

    result = pd.concat(out, ignore_index=True)
    logger.info(
        "Getis-Ord Gi* (real esda implementation, %d permutations/timestamp): %d/%d cell-hours "
        "flagged hot at >=90%% confidence.",
        permutations, (result[f"hotspot_category_{col}"].str.startswith("hot")).sum(), len(result),
    )
    return result


def compute_local_moran(panel_df: pd.DataFrame, w: "libpysal.weights.W", h3_order,
                         col: str = "rtt_median", permutations: int = 99,
                         seed: int = 42) -> pd.DataFrame:
    """
    Local Moran's I via `esda.Moran_Local` — identifies spatial clusters
    (High-High, Low-Low) and spatial outliers (High-Low, Low-High), which
    Getis-Ord Gi* alone cannot distinguish (Gi* finds hot/cold clusters but
    not outliers surrounded by dissimilar neighbors).
    """
    out = []
    for ts, group in panel_df.groupby("timestamp"):
        group_indexed = group.set_index("h3_index").reindex(h3_order)
        values = group_indexed[col].fillna(group_indexed[col].mean()).values

        m = Moran_Local(values, w, permutations=permutations, seed=seed)
        quadrant = pd.Series(m.q).map(MORAN_QUADRANT_LABELS).values
        significant = m.p_sim <= 0.05

        group_indexed["timestamp"] = ts
        group_indexed[f"local_moran_i_{col}"] = m.Is
        group_indexed[f"local_moran_pvalue_{col}"] = m.p_sim
        group_indexed[f"local_moran_quadrant_{col}"] = np.where(significant, quadrant, "not_significant")
        out.append(group_indexed.reset_index())

    return pd.concat(out, ignore_index=True)


def compute_historical_hotspot_frequency(panel_df: pd.DataFrame, col: str = "rtt_median",
                                          window: int = 168) -> pd.DataFrame:
    """
    Rolling frequency of past hot-spot flags per cell — "how often has this
    cell been a real statistical hotspot recently". Uses `shift(1)` before
    the rolling window so the current timestamp's own hotspot flag is never
    counted (would otherwise be a direct leak: today's label predicting
    today's label).
    """
    panel_df = panel_df.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)
    hot_flag = panel_df[f"hotspot_category_{col}"].str.startswith("hot").astype(int)
    panel_df[f"historical_hotspot_freq_{col}"] = (
        hot_flag.groupby(panel_df["h3_index"])
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )
    panel_df[f"historical_hotspot_freq_{col}"] = panel_df[f"historical_hotspot_freq_{col}"].fillna(0.0)
    return panel_df
