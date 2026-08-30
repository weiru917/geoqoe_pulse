"""
GeoQoE Pulse Data & GIS Pipeline — Orchestrator.

    Raw Data
        v
    H3 Study Grid (real, trimmed Klang-Valley-core boundary — see config.py)
        v
    Real Spatial Joins (deduplicated towers, buildings, Ookla, OSM, WorldPop, DOSM, ridership)
        v
    Real Raster/Inset Merges (VIIRS nighttime lights, ESA WorldCover land cover,
                              Sunway field measurements) -- BEFORE the baseline is built
        v
    Temporal Feature Engineering (leakage-free lags/rolling stats)
        v
    Spatial Feature Engineering (real libpysal weights, spatial lags/variance)
        v
    Demand Pressure Index (real static features; historical-QoE component
                            flagged as leakage-risk — see config.HIGH_LEAKAGE_RISK_FEATURES)
        v
    Hotspot Detection (real esda Getis-Ord Gi* + Local Moran's I)
        v
    QoE Target Generation (unified thresholds; SAME-HOUR + 1-HOUR-AHEAD forecast targets)
        v
    Validation (incl. target-consistency and score-range regression guards)
        v
    Export ML-ready datasets + honest static baseline + grid geometry + metadata

IMPORTANT: cell_time_dataset.parquet is a HIERARCHICAL panel, not a
measurement — a real citywide hourly shape (M-Lab NDT, converted from UTC
to Asia/Kuala_Lumpur at load) applied to a real per-cell level (Ookla
quarterly) where one exists, and a formula-derived level where it does
not. Jitter, loss and upload remain simulated. Because ONE shape is shared
by every cell, the anchored panel is rank-one and its forecast target is
analytically reconstructable — see config.PANEL_STRUCTURE_DISCLOSURE
before reporting any forecast metric from it. For a modelling claim that
needs no caveat, use cell_static_qoe_baseline.parquet (real Ookla labels,
real GIS features, zero synthetic QoE).

This is the SINGLE entry point. run_full_data_pipeline.py is a deprecated
shim that forwards here; the raster/inset merges it used to perform AFTER
export now run at stage 2a, so those real columns reach every output.

Exported files:
    data/processed/grid_geometry.parquet / .geojson
    data/outputs/cell_time_dataset.parquet          (hourly panel — see disclosure above)
    data/outputs/cell_static_features.parquet
    data/outputs/cell_static_qoe_baseline.parquet   (REAL — primary modelling deliverable)
    data/outputs/qoe_targets.parquet
    data/outputs/feature_dictionary.xlsx            (completeness asserted)
    data/outputs/data_quality_report.json           (validator output + ground-truth coverage)
    data/outputs/metadata.json
"""
import json
import logging
import time

import numpy as np
import pandas as pd

from config import (
    GLOBAL_RANDOM_SEED, PROCESSED_DIR, OUTPUT_DIR, BASE_DIR, H3_PRIMARY_RESOLUTION,
    FORECAST_HORIZON_HOURS, HIGH_LEAKAGE_RISK_FEATURES, PIPELINE_VERSION,
    SIMULATION_DISCLAIMER, DERIVED_LAYERS, LAG_WARMUP_HOURS,
)
from src.data_preprocessing.geometry_utils import generate_h3_study_grid, export_grid_geometry
from src.data_preprocessing.data_downloader import (
    build_static_features, load_ookla, build_dynamic_panel, build_static_qoe_baseline,
)
from src.data_preprocessing.feature_engineer import (
    compute_temporal_shifted_features, compute_demand_pressure_base,
    apply_dynamic_demand_adjustments, finalize_demand_pressure_index,
    FITTED_SCALERS,
)
from src.data_preprocessing.spatial_processor import (
    build_h3_spatial_weights, compute_vectorized_spatial_lags, compute_spatial_rolling_mean,
    compute_hotspots, compute_local_moran, compute_historical_hotspot_frequency,
)
from src.data_preprocessing.qoe_target import (
    compute_qoe_targets, compute_relative_targets, compute_next_hour_targets,
)
from src.data_preprocessing.application_profiles import (
    compute_application_profiles, profile_metadata,
)
from src.data_preprocessing.coverage_bias import compute_coverage_bias_columns
from src.data_preprocessing.viirs_quarterly import (
    load_viirs_quarterly, build_viirs_growth_features,
    attach_to_quarterly_panel as attach_viirs_quarterly,
)
from src.data_preprocessing.district_population import (
    assign_cells_to_districts, load_district_population,
    build_district_population_features, attach_to_quarterly_panel,
)
from src.data_preprocessing.ookla_quarterly import (
    build_quarterly_panel, summarize_quarterly_features, build_quarterly_ml_dataset,
    QUARTERLY_PANEL_PATH,
)
from src.data_preprocessing.data_validator import run_full_validation
from src.data_preprocessing.metadata_utils import generate_metadata, summarize_synthetic_fraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

# --------------------------------------------------------------------------
# Feature-exclusion sets — TWO modes, because "what's safe to use as a
# feature" depends on which target you're predicting. This replaces the
# single NON_FEATURE_COLS set from the previous version, which only ever
# supported same-hour prediction and didn't exclude spatial_lag_dl_median/
# spatial_lag_rtt_median (both direct functions of the same-hour target
# inputs) at all.
# --------------------------------------------------------------------------
STRUCTURAL_COLS = {"h3_index", "timestamp", "row_id", "h3_parent_res6", "h3_parent_res5"}

# Constant / zero-information columns — never useful as a feature regardless
# of target mode.
CONSTANT_COLS = {
    # Provenance markers, constant across every panel row — zero information
    # as features, but kept in the exported data so the disclosure travels
    # with the file rather than living only in metadata.json.
    "qoe_hourly_shape_source", "qoe_jitter_loss_is_simulated", "qoe_hourly_shape_scope",
    # Same, for the static layers merged in from the raster extractions.
    "viirs_source", "worldcover_source", "sunway_source", "viirs_months_covered",
    "building_height_source", "building_height_is_usable",
    "building_height_tagged_fraction", "building_height_coverage_is_biased",
    "sunway_ping_fields_sparse", "sunway_test_event_unit_verified",
    "worldcover_tile_covered",
}

RAW_QOE_OBSERVATION_COLS = {
    "dl_median", "dl_mean", "dl_std", "dl_p95",
    "ul_median", "ul_mean", "ul_std",
    "rtt_median", "rtt_mean", "rtt_std",
    "jitter_median", "loss_median",
}
# Same-hour derivatives of the raw QoE columns that a naive user would NOT
# think to exclude just by reading "exclude dl_median/rtt_median" — flagged
# explicitly here because they were the review's leakage finding.
SAME_HOUR_QOE_DERIVED_COLS = {"spatial_lag_dl_median", "spatial_lag_rtt_median"}

# Ordinal encodings for categorical string columns that would otherwise
# crash tree-based learners (XGBoost, LightGBM) at fit time. The string
# originals are kept in the exported panel for dashboard human-readable
# labels; the _encoded integer versions are what the ML feature lists
# include. Ordering is deliberately semantic (cold → not_significant → hot;
# LL → LH → HL → HH) so ordinal encoding preserves meaningful ordering.
CATEGORICAL_ENCODINGS = {
    "hotspot_category_rtt_median": {
        "cold_99": 0, "cold_95": 1, "cold_90": 2, "not_significant": 3,
        "hot_90": 4, "hot_95": 5, "hot_99": 6,
    },
    "local_moran_quadrant_rtt_median": {
        "not_significant": 0, "LL": 1, "LH": 2, "HL": 3, "HH": 4,
    },
}
# Raw string columns excluded from ML features — use the _encoded versions.
STRING_CATEGORICAL_COLS = set(CATEGORICAL_ENCODINGS.keys())

# Priority-score components are post-processing for the dashboard / ranking
# pipeline, not independent predictors — including them would just add
# multicollinearity with their real-feature inputs (population_day,
# historical_hotspot_freq, qoe_confidence_score).
PRIORITY_SCORE_COLS = {"priority_exposure_score", "priority_base_score"}

# NEW: columns that determine (rather than merely correlate with) whether
# the SIMULATED hourly panel's degradation target can be positive at all —
# see data_validator.validate_target_anchor_correlation, which found that
# on this panel 100% of positive target_is_degraded_next_1h labels came
# from qoe_level_is_ookla_anchored == True cells. That's not a soft
# correlation to flag and leave in (the way demand_pressure_index's
# historical_qoe circularity is handled) — a model with direct access to
# this column, or the quarterly counts that are only ever non-null when it
# is True, could trivially learn "is this cell anchored" instead of any
# genuine demand/QoE pattern. Hard-excluded from BOTH target modes for the
# panel/targets tables (a modelling team that deliberately wants to
# stratify by anchor status can still do so explicitly — this only changes
# the SAFE DEFAULT feature list).
ANCHOR_STATUS_LEAKAGE_COLS = {
    "qoe_level_is_ookla_anchored", "qoe_baseline_source",
    "ookla_quarter_test_count", "ookla_quarter_device_count",
}

# NEW: on cell_static_qoe_baseline.parquet specifically, these ARE the
# target (ookla_dl_mbps / ookla_rtt_ms / ookla_tests, per the README's
# "Option A" recommended workflow) or a direct derivative/split-indicator
# of it (has_real_ookla_label, qoe_risk_label — qoe_risk_label is literally
# thresholded from ookla_dl_mbps; ookla_is_synthetic is has_real_ookla_label
# inverted under a different name). None of these belong in a feature list
# for predicting Ookla QoE from GIS features — the previous version of
# build_feature_dictionary never actually applied this or any other
# exclusion to the two static tables at all (see the "timestamp in
# df.columns" branch fix below), so it flagged h3_index, h3_parent_res6,
# and all of the columns below as approved ML features.
BASELINE_TARGET_AND_LABEL_COLS = {
    "ookla_dl_mbps", "ookla_rtt_ms", "ookla_tests", "ookla_devices",
    "has_real_ookla_label", "qoe_risk_label", "ookla_is_synthetic",
    # Coverage-bias columns are WEIGHTS and DIAGNOSTICS, not predictors.
    # coverage_propensity is fitted to predict which rows have a label, so
    # using it as a feature would teach a model to recognise labelled rows
    # rather than to predict QoE.
    "coverage_propensity", "in_common_support", "ookla_label_confidence",
}


# --------------------------------------------------------------------------
# Quarterly dataset feature rules — deliberately SEPARATE from
# get_feature_columns(), because the same column name means different
# things in the two tables and one shared rule would be wrong in both.
#
# On cell_static_qoe_baseline.parquet, `ookla_dl_mbps` IS the Option A
# target and must never be a feature. On cell_quarter_dataset.parquet the
# same column is quarter t's REAL observation being used to predict
# quarter t+1 — genuinely available beforehand, and the single most
# informative feature available. Overloading one exclusion set across both
# tables would either leak the static target or cripple the quarterly one.
# --------------------------------------------------------------------------
QUARTERLY_STRUCTURAL_COLS = {
    "h3_index", "h3_parent_res6", "h3_parent_res5",
    "year", "quarter", "quarter_label", "quarter_index",
}
# Same-quarter QoE observations. Legitimate features for the NEXT-quarter
# target (observed before the quarter being predicted); leakage for any
# same-quarter target.
# Split/provenance indicators on the quarterly table — never predictors.
# district_population_is_carried_forward is 1 exactly when the quarter is
# past the end of the DOSM series (2026), which under the recommended
# temporal holdout is EXACTLY the test set. As a feature it is a perfect
# train/test indicator: constant 0 in train, constant 1 in test. Same
# exclusion reason as has_real_ookla_label on the static table.
QUARTERLY_SPLIT_INDICATOR_COLS = {
    "district_population_is_carried_forward",
}

QUARTERLY_SAME_PERIOD_QOE_COLS = {
    "ookla_dl_mbps", "ookla_ul_mbps", "ookla_rtt_ms",
    "is_degraded", "is_below_area_norm", "dl_percentile_in_quarter",
}


def get_quarterly_feature_columns(df: pd.DataFrame, target_mode: str = "next_quarter") -> list:
    """Approved feature list for cell_quarter_dataset.parquet."""
    import pandas.api.types as ptypes

    exclude = (QUARTERLY_STRUCTURAL_COLS | _target_cols(df) | PRIORITY_SCORE_COLS
               | QUARTERLY_SPLIT_INDICATOR_COLS)
    exclude |= {c for c in df.columns if c.endswith("_valid")}
    if target_mode == "current_quarter":
        exclude |= QUARTERLY_SAME_PERIOD_QOE_COLS
    exclude |= {
        c for c in df.columns
        if not (ptypes.is_numeric_dtype(df[c]) or ptypes.is_bool_dtype(df[c]))
        and c not in exclude
    }
    return [c for c in df.columns if c not in exclude]


def _target_cols(df: pd.DataFrame) -> set:
    return {c for c in df.columns if c.startswith("target_")}


def get_feature_columns(df: pd.DataFrame, target_mode: str) -> list:
    """
    target_mode = "current"   -> predicting target_is_degraded/_class/_index
                                  for THIS hour. Must exclude every raw QoE
                                  column AND their same-hour derivatives
                                  (spatial lags), or the model trivially
                                  reconstructs the label from a same-hour
                                  proxy (e.g. dl_mean = dl_median * 1.05).
    target_mode = "next_hour" -> predicting target_is_degraded_next_1h (the
                                  recommended MVP target). Current-hour QoE
                                  values and their spatial lags ARE legitimate
                                  features here, because they are genuinely
                                  observed before the hour being predicted.

    Works on any of the four output tables, not just the hourly panel —
    static per-cell tables (no "timestamp" column) go through exactly the
    same exclusion logic rather than a separate "everything is a feature"
    shortcut (see build_feature_dictionary's previous "timestamp in
    df.columns" branch, now removed).
    """
    if target_mode not in ("current", "next_hour"):
        raise ValueError('target_mode must be "current" or "next_hour"')

    exclude = (
        set(STRUCTURAL_COLS) | set(CONSTANT_COLS) | _target_cols(df) | STRING_CATEGORICAL_COLS
        | PRIORITY_SCORE_COLS | ANCHOR_STATUS_LEAKAGE_COLS | BASELINE_TARGET_AND_LABEL_COLS
    )
    if target_mode == "current":
        exclude |= RAW_QOE_OBSERVATION_COLS | SAME_HOUR_QOE_DERIVED_COLS

    # Defensive dtype catch-all: any remaining free-text/string column (e.g.
    # a future provenance/label string nobody remembered to add to a named
    # exclusion set above) is dropped from the approved feature list rather
    # than silently handed to a tree-based learner that would choke on it,
    # or — worse — encoded by surprise into something that looks like a
    # legitimate numeric feature. Checks numeric/bool-ness rather than
    # `dtype == object` specifically, because pandas' newer Arrow-backed
    # StringDtype columns (dtype shows as "str", not "object" — e.g.
    # tower_source in this pipeline) would otherwise slip through a
    # numpy-object-only check.
    import pandas.api.types as ptypes
    exclude |= {
        c for c in df.columns
        if not (ptypes.is_numeric_dtype(df[c]) or ptypes.is_bool_dtype(df[c]))
        and c not in exclude
    }

    return [c for c in df.columns if c not in exclude]


COLUMN_DESCRIPTIONS = {
    "h3_index": "Uber H3 res-8 cell identifier (join key).",
    "timestamp": "Hourly timestamp (UTC-naive, local Klang Valley convention).",
    "row_id": "Surrogate unique row identifier (h3_index + timestamp composite, as int).",
    "h3_parent_res6": "Coarse H3 resolution-6 parent of h3_index (~130 groups over the current grid). Use for SPATIAL HOLDOUT splits — hold out whole groups, don't randomly sample individual h3_index values (spatially autocorrelated neighbours would leak across a random split). Structural/grouping key, not an ML feature.",
    "dl_median": "Median simulated download throughput (Mbps) for this cell-hour. NOT a measurement — see config.SIMULATION_DISCLAIMER.",
    "dl_mean": "dl_median x 1.05 (formula-derived, not an independent statistic). Excluded from features when predicting the SAME-HOUR target (leakage); allowed for next-hour prediction.",
    "dl_std": "dl_median x 0.15 (formula-derived). Same leakage caveat as dl_mean.",
    "dl_p95": "dl_median x 1.30 (formula-derived). Same leakage caveat as dl_mean.",
    "ul_median": "Median simulated upload throughput (Mbps) for this cell-hour. Simulated, see above.",
    "ul_mean": "ul_median x 1.05 (formula-derived).",
    "ul_std": "ul_median x 0.15 (formula-derived).",
    "rtt_median": "Median simulated round-trip latency (ms) for this cell-hour. Simulated, see above.",
    "rtt_mean": "rtt_median x 1.05 (formula-derived).",
    "rtt_std": "rtt_median x 0.20 (formula-derived).",
    "jitter_median": "Simulated jitter (ms), derived from rtt_median plus noise.",
    "loss_median": "Simulated packet loss percentage, derived from the same load_factor as dl/rtt.",
    "test_sample_count": "Simulated HOURLY-equivalent test count. For Ookla-anchored cells this is Poisson-sampled around an hourly rate derived from the REAL quarterly total (ookla_quarter_test_count / ~2160 hours, scaled by demand), NOT the quarterly total itself repeated per hour (that was a bug — see CHANGELOG). Represents measurement VOLUME, not the outcome — safe as a feature in either target mode.",
    "unique_device_count": "~70% of test_sample_count (rounded). Same status as test_sample_count.",
    "ookla_quarter_test_count": "REAL Ookla quarterly test count for this cell (constant across all 168 simulated hours by definition — it is a QUARTERLY total, never a per-hour figure). NaN if the cell has no Ookla tile coverage.",
    "ookla_quarter_device_count": "REAL Ookla quarterly unique-device count for this cell (constant across all 168 simulated hours; quarterly, not hourly). NaN if unavailable in the source tile or no coverage exists.",
    "qoe_baseline_source": "Human-readable provenance string: 'ookla_q4_2024_static_anchor' or 'no_real_anchor_simulated_only'. Complements the qoe_level_is_ookla_anchored boolean.",
    "ridership_daily_surge_factor": "Real daily ridership-derived demand multiplier (normalized to the selected week's mean), summed only over Klang-Valley-relevant transit modes (national/inter-city services like ETS, Intercity, Komuter Utara, and Tebrau are excluded). Previously folded silently into the hourly formula with no standalone column; now exported directly so it can be inspected or used as its own feature.",
    "qoe_level_is_ookla_anchored": "True if a real Ookla quarterly average existed to re-centre this cell's simulated hourly values on (~41% of cells). False does not mean 'less real' shape — the hourly SHAPE is simulated either way.",
    "qoe_hourly_shape_source": "Provenance of the hourly SHAPE applied to this row: the real M-Lab NDT citywide diurnal curve, keyed in UTC in the source CSV and converted to Asia/Kuala_Lumpur at load time. ONE curve is shared by every cell — a real shape, not per-cell resolution. Constant, excluded from ML features.",
    "qoe_jitter_loss_is_simulated": "Always True. jitter_median and loss_median have no real source at this resolution and are formula-derived. Sparse real ping/jitter/loss values exist only inside the Sunway inset and are kept separately (sunway_median_*), never merged into these columns. Constant, excluded from ML features.",
    # --- VIIRS nighttime lights (real, 100% coverage) ---
    "viirs_avg_rad_mean": "REAL mean nighttime-lights radiance for the cell, averaged over 6 monthly VIIRS VNP46A3 AllAngle_Composite_Snow_Free composites (Nov 2024 - Apr 2025), via zonal statistics. A strong proxy for economic activity and built-up intensity that is independent of OSM completeness.",
    "viirs_source": "Provenance string for the VIIRS layer. Constant, excluded from ML features.",
    "viirs_monthly_std": "Standard deviation of this cell's 6 monthly VIIRS composites (Nov 2024 - Apr 2025). A real STABILITY measure — lights that fluctuate month to month behave differently from steady ones. NaN where any month lacks a valid retrieval (cloud), so the figure is never computed over a different set of months than a neighbouring cell's.",
    "viirs_monthly_cv": "viirs_monthly_std divided by the cell's mean radiance — variability normalised for brightness, so a dim fluctuating cell is comparable to a bright one.",
    "viirs_n_months_observed": "How many of the 6 monthly VIIRS composites produced a valid retrieval for this cell (Nov 2024 has 1,037 cloud-affected cells). Below 6, the stability columns are NaN.",
    "viirs_months_covered": "Which monthly composites back the VIIRS columns. Constant; excluded from features. NOTE these 6 months span only 3 of the panel's 14 quarters — VIIRS is a STATIC feature here, not a time-varying one.",
    # --- OSM building heights (real, sparse, coverage-biased) ---
    "building_height_mean_m": "Mean height (m) of height-tagged buildings in the cell, from real OSM `height` tags or `building:levels` x 3.2 m. NaN for the 79% of cells with no tagged building — never imputed. SPARSE AND BIASED, see building_height_coverage_is_biased.",
    "building_height_max_m": "Tallest height-tagged building in the cell (m). Sanity check: the study-area maximum is 678.9 m, which is Merdeka 118 — the extraction is finding real structures.",
    "building_height_p90_m": "90th-percentile height of tagged buildings in the cell — less sensitive to a single landmark than the max.",
    "building_n_with_height": "How many buildings in the cell carry usable height information. 0 means no tagged buildings, not zero-height buildings. Use it to judge whether the height columns mean anything for a given cell.",
    "building_height_is_usable": "True when study-area tag coverage cleared the 2% threshold (it is 6.51%). Constant; excluded from features.",
    "building_height_tagged_fraction": "Fraction of buildings in the study area carrying a usable height/levels tag (0.0651). Constant; excluded from features.",
    "building_height_coverage_is_biased": "Always True. OSM height tags cluster where mappers are active — city centre, malls, landmarks — so this column partly encodes 'is this a well-mapped area' rather than 'how tall is it'. Same non-random-coverage trap as the Ookla labels; pair with coverage_propensity when interpreting.",
    "building_height_source": "Provenance string for the building-height layer. Constant; excluded from features.",
    # --- Copernicus DEM terrain (real, 100% coverage) ---
    "elevation_mean_m": "REAL mean surface elevation (m) of the cell, Copernicus DEM GLO-30 zonal statistics. SURFACE model (DSM): includes buildings and canopy, so this is NOT bare-earth ground height and it correlates with the building layers by construction. Median across the grid is 28.9 m.",
    "elevation_min_m": "REAL lowest surface elevation (m) within the cell.",
    "elevation_max_m": "REAL highest surface elevation (m) within the cell. The grid maximum is 1,282 m in the Titiwangsa foothills on the eastern edge.",
    "elevation_range_m": "elevation_max_m minus elevation_min_m — the vertical span inside one cell.",
    "terrain_ruggedness_m": "Standard deviation of surface height within the cell (m), median 6.4. The obstruction measure: high values mean broken terrain or tall structures, both of which impede propagation. This is what separates cells that are hard to serve for PHYSICAL reasons from cells that are merely busy — the persistent-limitation vs demand-pressure distinction.",
    "dem_pixel_count": "Number of 30 m DEM pixels contributing to this cell (~900 for a full res-8 cell). A coverage check, not a feature.",
    "dem_is_surface_model": "Always True. Records that this is a DSM (surface, includes buildings/canopy) rather than a DTM (bare earth), so elevation_mean_m is never mistaken for ground height. Constant; excluded from features.",
    "dem_source": "Provenance string for the terrain layer. Constant; excluded from features.",
    # --- ESA WorldCover land cover (real, 100% coverage, all 11 classes) ---
    "landcover_built_up_pct": "REAL percentage of the cell classified built-up by ESA WorldCover 2021 (10 m). Independent of building footprints — useful where Google Open Buildings is incomplete.",
    "landcover_tree_cover_pct": "REAL percentage of the cell under tree cover (ESA WorldCover 2021).",
    "landcover_water_pct": "REAL percentage of the cell that is permanent water (ESA WorldCover 2021). High values mark cells where population/demand are structurally low.",
    "landcover_cropland_pct": "REAL percentage of the cell under cropland (ESA WorldCover 2021).",
    "landcover_shrubland_pct": "REAL percentage of the cell under shrubland (ESA WorldCover 2021).",
    "landcover_grassland_pct": "REAL percentage of the cell under grassland (ESA WorldCover 2021).",
    "landcover_bare_sparse_veg_pct": "REAL percentage of the cell that is bare or sparsely vegetated (ESA WorldCover 2021).",
    "landcover_snow_ice_pct": "REAL percentage snow/ice (ESA WorldCover 2021). Zero everywhere in this study area; retained so the 11 class percentages sum to exactly 100 and a zero can be distinguished from a dropped class.",
    "landcover_wetland_herbaceous_pct": "REAL percentage of the cell under herbaceous wetland (ESA WorldCover 2021). Previously dropped in the merge step.",
    "landcover_mangroves_pct": "REAL percentage of the cell under mangroves (ESA WorldCover 2021) — mean 3.9% of the grid, above 10% in 305 coastal Klang/Kuala Langat cells, 100% in some. Previously dropped in the merge step, which made those cells read as 'no land-cover data'. Genuine signal: mangrove coast is low-demand, hard-to-serve terrain.",
    "landcover_moss_lichen_pct": "REAL percentage moss/lichen (ESA WorldCover 2021). Zero everywhere in this study area; retained for sum-to-100 completeness.",
    "worldcover_tile_covered": "True if any WorldCover pixels fell inside the cell. True for all 4,918 cells. Constant, excluded from ML features.",
    "worldcover_pixel_count": "Number of 10 m WorldCover pixels contributing to this cell's percentages, summed across BOTH source tiles. For the 100 cells straddling the tile boundary at latitude 3.0 this is now the full-cell count; a previous version described those cells using only the sliver inside whichever tile was processed last.",
    "worldcover_n_tiles": "How many source tiles contributed pixels to this cell (1 for interior cells, 2 for the 100 cells on the latitude-3.0 seam). Diagnostic for the tile-accumulation fix.",
    "worldcover_source": "Provenance string for the land-cover layer. Constant, excluded from ML features.",
    # --- Sunway real field-measurement inset (10 of 4,918 cells) ---
    "sunway_n_records": "Count of REAL GNetTrack field measurements (Mendeley dx5xyyfz2y) falling in this cell. Non-null for only 10 of 4,918 cells — NaN elsewhere is honest sparsity, never imputed.",
    "sunway_median_dl_mbps": "Median of the REAL passive DL_bitrate log (converted kbps -> Mbps) for this cell. Near-zero for most cells and NOT a bug: a continuous passive log is dominated by idle moments, so this is 'typical instant', not 'achievable speed'. NOT comparable to an Ookla speed-test median.",
    "sunway_p90_dl_mbps": "90th percentile of the REAL passive throughput log — closer to achievable speed than the median, but still a passive observation, not a speed test.",
    "sunway_median_ul_mbps": "Median REAL passive upload throughput (kbps -> Mbps). Same idle-dominated caveat as sunway_median_dl_mbps.",
    "sunway_median_rsrp_dbm": "Median REAL RSRP (dBm) measured in the cell — genuine radio signal strength, the only direct signal-quality measurement anywhere in this dataset.",
    "sunway_median_snr_db": "Median REAL signal-to-noise ratio (dB) measured in the cell.",
    "sunway_n_real_ping_samples": "How many REAL ping samples back sunway_median_ping_ms for this cell (0.2% of records carry them). Single digits in most cells — check this before trusting the ping columns.",
    "sunway_median_ping_ms": "Median REAL measured ping (ms). Sparse (see sunway_n_real_ping_samples), never imputed.",
    "sunway_median_jitter_ms": "Median REAL PINGSTDEV (ms) — a ping standard deviation used as a jitter proxy, not an RFC 3550 interarrival jitter. Sparse.",
    "sunway_median_packet_loss_pct": "Median REAL measured packet loss (%). All observed values are 0.0 across the 66 real samples — real, but degenerate; do not read variation into it.",
    "sunway_n_real_test_events": "Count of REAL deliberate speed-test events recorded in this cell (69 across the whole dataset). Use this to judge whether sunway_test_event_dl_raw means anything for a given cell.",
    "sunway_test_event_dl_mbps": "REAL measured download throughput (Mbps) from a deliberate GNetTrack speed test in this cell. UNIT RESOLVED EMPIRICALLY as kbps and converted: the file's Test_Status column marks each row's phase, and the reported value matches the throughput actually measured during that phase of the same session (ratio 1.03 for download, 1.04 for upload). DO NOT COMPARE TO ookla_dl_mbps: this test averages ~0.5 Mbps in cells where Ookla measures ~150 Mbps, so it is a short or constrained transfer, not a saturating speed test. Use it to compare Sunway cells with each other only. Non-null for 8 cells.",
    "sunway_test_event_ul_mbps": "REAL measured upload throughput (Mbps) from the same GNetTrack test events, same kbps conversion and same verification. Averages ~5.6 Mbps — genuinely faster than the download test in this campaign, which is unusual but internally consistent and independently confirmed against concurrent UL_bitrate.",
    "sunway_test_event_unit_verified": "Always False. Machine-readable disclaimer so sunway_test_event_dl_raw cannot be treated as a calibrated speed without tripping over this column. Constant, excluded from ML features.",
    "sunway_source": "Provenance string for the Sunway inset. Constant, excluded from ML features.",
    "sunway_ping_fields_sparse": "Always True — flags that the sunway ping/jitter/loss columns have ~0.2% coverage. Constant, excluded from ML features.",
    "target_qoe_index": "Custom composite 0-100 QoE index (log-utility download/upload minus exponential latency/loss penalty). NOT a validated ITU-T G.107 MOS — see qoe_target.py docstring.",
    "target_service_quality_1to5": "1-5 linear rescaling of target_qoe_index. NOT a validated MOS/ACR score.",
    "target_is_degraded": "Binary SAME-HOUR degradation flag, derived from target_degradation_class >= 1 (single shared threshold source — config.QOE_THRESHOLDS — so this can no longer disagree with target_degradation_class).",
    "target_degradation_class": "0=normal, 1=degraded, 2=severe. Same threshold source as target_is_degraded.",
    f"target_is_degraded_next_{FORECAST_HORIZON_HOURS}h": "DEMONSTRATION TARGET ONLY — NOT the primary ML target (that is target_is_below_area_norm_next_q on cell_quarter_dataset.parquet). This label is analytically reconstructable from (cell, hour-of-week) because the panel is rank-one, so a model scores near-perfectly on it by memorisation; reported skill here is not evidence of predictive ability. See PANEL_STRUCTURE_DISCLOSURE in config.py. Mechanically: target_is_degraded shifted -1 per cell (groupby h3_index).shift(-1) — genuinely predicting the future, not classifying the present. NaN/0-valid for each cell's final row (no future observation exists).",
    f"target_degradation_class_next_{FORECAST_HORIZON_HOURS}h": "3-class version of the next-hour forecast target.",
    f"target_qoe_index_next_{FORECAST_HORIZON_HOURS}h": "Continuous version of the next-hour forecast target.",
    f"target_next_{FORECAST_HORIZON_HOURS}h_valid": "1 if a real future observation existed to build the next-hour target from; 0 for each cell's final row. Drop rows where this is 0 before training — do not treat them as a 'not degraded' case.",
    "target_qoe_percentile_in_area": "RELATIVE target: this cell's dl_median percentile (0-100) among ALL cells at the SAME timestamp. 50 = exactly typical for the study area at that hour. Complements the absolute threshold, which fires on only ~2.3% of cells against a real Ookla median of 175 Mbps. CAVEAT: while the panel uses one shared diurnal curve, this is time-invariant by construction (a common positive factor cannot change a cross-cell ordering) — it becomes genuinely spatio-temporal once per-district curves exist.",
    "target_is_below_area_norm": "RELATIVE degradation label: 1 if target_qoe_percentile_in_area is below config.RELATIVE_DEGRADATION_PERCENTILE (default 25). Expresses 'this neighbourhood is worse than the city right now', which the absolute 10 Mbps cutoff cannot. Same time-invariance caveat as the percentile column.",
    "target_is_below_area_norm_next_1h": "target_is_below_area_norm shifted -1 per cell — the relative counterpart of target_is_degraded_next_1h.",
    "target_qoe_percentile_in_area_next_1h": "target_qoe_percentile_in_area shifted -1 per cell (continuous relative forecast target).",
    "qoe_hourly_shape_scope": "Which diurnal curve produced this row's shape factor: 'district_mlab_curve' if a real per-district M-Lab curve covered this cell and hour, 'citywide_mlab_curve_fallback' otherwise. Currently 100% fallback — no per-district M-Lab export exists yet — which is why the anchored panel remains rank-one. Constant while that holds; excluded from ML features.",
    "h3_parent_res5": "Coarse H3 resolution-5 parent (~20 km, 26 groups over this grid). The join handle for PER-DISTRICT diurnal curves (see mlab_diurnal_shape). Structural/grouping key, not an ML feature.",
    "coverage_tier": "Three-way honesty label for spatial predictions. 'measured' (2,950 cells) = a real Ookla label exists, show the number. 'estimable' (280) = no label but the cell sits inside the labelled cells' feature range, so a prediction is INTERPOLATION — show it with a confidence caveat. 'unknown' (1,688) = no label AND outside common support, so a prediction is EXTRAPOLATION into territory the model never saw — present as 'insufficient data', not as a low estimate. The honest reach of the static model is therefore ~280 cells beyond what is measured, not 1,968. Structural label, not a feature.",
    "sunway_rsrp_p10_dbm": "10th-percentile RSRP (dBm) from real field measurements — the WEAK TAIL. A cell can have an acceptable median while its p10 sits at -110 dBm, which is where calls actually drop. Medians hide coverage holes; this exposes them. Non-null for 10 cells only.",
    "sunway_rsrp_p90_dbm": "90th-percentile RSRP (dBm) — best-case signal in the cell. Paired with p10 it gives the spread, which matters more for coverage than the centre.",
    "sunway_snr_p10_db": "10th-percentile signal-to-noise ratio (dB). Low values mean interference or cell-edge conditions even where raw signal strength looks adequate.",
    "sunway_pct_poor_rsrp": "Percentage of real measurements in the cell below -100 dBm, the conventional weak-coverage threshold. The clearest single indicator of a coverage hole: one cell in the inset reads 91.4%, which no median would have revealed.",
    "sunway_pct_5g_samples": "Percentage of real measurements that connected via 5G rather than 4G. A direct observation of technology availability on the ground, not an inference from a tower database.",
    "sunway_median_cqi": "Median Channel Quality Indicator reported by the handset (0-15). A device-side view of link quality that combines signal, interference and noise.",
    "sunway_n_operators": "How many of the three measured operators were observed in this cell. Where it is 1, the per-operator columns are not comparable — only one network was sampled there.",
    "sunway_rsrp_operator_a_dbm": "Median RSRP (dBm) for Operator A in this cell. The three operators were measured SIMULTANEOUSLY on the same routes, so differences between these columns are a genuine like-for-like comparison — rare in open data, and the only operator-specific statement this project can make. One cell shows a 19 dB spread between operators. NaN where that operator was not sampled.",
    "sunway_rsrp_operator_b_dbm": "Median RSRP (dBm) for Operator B — see sunway_rsrp_operator_a_dbm for the comparison caveat.",
    "sunway_rsrp_operator_c_dbm": "Median RSRP (dBm) for Operator C — see sunway_rsrp_operator_a_dbm for the comparison caveat.",
    "coverage_propensity": "Out-of-fold predicted probability that this cell has a real Ookla label, fitted from GIS features only with a spatial (h3_parent_res6) grouping. Ookla coverage is predictable at ROC-AUC ~0.98, so the labelled cells are NOT a random sample. USE AS AN INVERSE-PROPENSITY WEIGHT when training on labelled cells. NOT a predictor — excluded from feature lists, since training on it teaches a model to recognise labelled rows.",
    "in_common_support": "True where this cell's coverage_propensity falls inside the range the LABELLED cells actually span (their p5-p95). Predictions for unlabelled cells outside common support are extrapolation, not interpolation, and should be flagged in the dashboard rather than shown at equal confidence. 85.4% of unlabelled cells currently fall outside it.",
    "ookla_label_confidence": "0-1 confidence in the LABEL itself, log-scaled from the real test count behind it and capped at ~100 tests. NaN where the cell has no label (a missing label has no confidence; 0 would wrongly read as 'confidently bad'). The median labelled cell rests on ~34 real tests and the p25 on ~8, so labels are not equally reliable.",
    "is_measured_peak_period": "1 during the hours the REAL M-Lab curve identifies as this city's congestion window (19:00-00:00 local, worst at 22:00), derived from 2.05M real tests rather than assumed. Contrast is_peak_period, which hardcodes commuter hours (08-10, 17-20) that the measurement does not support. Both are exported; prefer this one for any claim about when the network is actually busy.",
    "district_name": "Administrative district containing this cell's centroid (point-in-polygon against the study boundary). Location hierarchy for dashboard roll-up and filtering — use instead of showing a raw H3 hash. Structural label, not an ML feature.",
    "dosm_district": "The DOSM spelling of district_name, used to join the official population series (GADM writes 'HuluLangat', DOSM writes 'Ulu Langat'). Join key, not a feature.",
    "district_population": "REAL official DOSM population of this cell's district, in persons, for the YEAR of this quarter. The project's first demand feature that varies over TIME as well as space — every other demand proxy is constant per cell across all quarters. Resolution is DISTRICT, not cell: 8 districts cover 4,918 cells, so all cells in Petaling share one trajectory. Supports 'degradation concentrates in districts whose population is growing'; does NOT support a per-hexagon causal claim.",
    "district_population_yoy_growth_pct": "Year-on-year percentage change in the district's official population. NaN for the first year of the series (no prior year to compare — 0 would wrongly read as 'flat'). Districts genuinely diverge: Klang +8.6% and Ulu Langat +8.3% over 2020-2025 against Petaling's +3.2%.",
    "district_population_growth_since_2020_pct": "Percentage change in district population against the 2020 baseline — cumulative growth rather than the year-on-year rate.",
    "district_population_is_carried_forward": "1 where the Ookla panel runs past the end of the DOSM series (which ends 2025) and the last observed year's value is applied. ~14.6% of rows, all in 2026. These are real values applied to a later period, not estimates — filter them out if you need strictly contemporaneous population. NOT A FEATURE: under the recommended temporal holdout this flag is 1 exactly on the test set and 0 exactly on train, making it a perfect split indicator. Excluded from the feature lists for the same reason as has_real_ookla_label.",
    "is_commuter_period": "1 during the ASSUMED commuter window (weekday 08-10 and 17-20). Retained for comparison and reproducibility only — the measured curve does not support these hours as a peak for this city. Prefer is_peak_period.",
    "is_measured_peak_period": "Identical to is_peak_period while config.PEAK_PERIOD_DEFINITION is 'measured'. Kept as an explicit name so the provenance of the peak window is unambiguous.",
    "app_voice_call_ok": "1 if the cell meets the MEASURABLE requirements for online voice calls (>=0.1 Mbps, <=150 ms latency per ITU-T G.114). NaN where inputs are missing — never 0 by default, since 'unknown' and 'fails' justify different actions. PARTIAL: jitter and packet loss are not in Ookla, so a cell can pass here and still sound broken. Voice is the highest-importance activity for Malaysian mobile users (4.31/5) and carries the largest satisfaction gap (-0.34).",
    "app_web_browsing_ok": "1 if the cell meets the requirements for web browsing (>=5 Mbps, <=100 ms latency). Latency-dominated: a page load is many round trips. Second-highest importance (4.24/5), gap -0.31.",
    "app_video_call_ok": "1 if the cell meets the measurable requirements for online video calls (>=2 Mbps down, >=1.5 Mbps UP, <=150 ms). Upload is usually the binding constraint. PARTIAL — jitter and loss unmeasurable. Importance 4.09/5, gap -0.19.",
    "app_video_streaming_ok": "1 if the cell supports HD streaming (>=5 Mbps, <=400 ms — buffering absorbs latency). Importance 3.88/5, and the only mobile scope where expectations were already met (gap +0.04).",
    "app_online_gaming_ok": "1 if the cell meets the measurable gaming requirements (>=1 Mbps, <=100 ms). Latency dominates; bandwidth needs are tiny. Lowest importance (3.47/5) and expectations already exceeded (gap +0.33), so a gaming failure is weighted least.",
    "app_user_impact_score": "0-1. The share of measurable application requirements this cell FAILS, weighted by how much Malaysian users say each application matters (MCMC BQoES 2024 importance scores). 0 = everything measurable works; 1 = every measurable application fails. Weights come from the survey; thresholds come from engineering guidance — the two are deliberately not conflated.",
    "app_n_evaluable": "How many of the five application profiles could actually be evaluated for this row, given which QoE inputs are present. Use it to judge how much app_user_impact_score is worth.",
    "app_n_failing": "How many application profiles this cell fails outright.",
    "app_worst_failing": "The highest-importance application this cell fails, by BQoES weighting — the dashboard's answer to 'what is actually broken here'. Blank where nothing measurable fails.",
    "viirs_rad_quarter": "REAL mean VIIRS nighttime radiance for this cell in THIS quarter, averaged over whichever monthly composites in the quarter produced a valid retrieval. The project's finest-resolution time-varying demand proxy — per cell, per quarter, from 43 real monthly granules spanning 2022-12 to 2026-06. A proxy for economic and human activity, NOT a measurement of mobile demand: say 'associated with rising activity', never 'caused by rising demand'.",
    "viirs_n_months_in_quarter": "How many valid monthly VIIRS retrievals back this cell-quarter (1-3). Cloud cover is heavy at this latitude, so a quarter resting on one month is weaker evidence than one resting on three — weight or filter by this.",
    "viirs_rad_yoy_growth_pct": "Year-on-year PERCENTAGE change in this cell's quarterly radiance. HEAVY-TAILED: percentage change explodes on a dim baseline (1.37% of valid rows exceed +100%, median baseline 1.26 against 11.92 overall; the maximum is +1,152%). Intuitive for a dashboard, but prefer viirs_rad_yoy_log_ratio for modelling. YoY rather than quarter-on-quarter deliberately: the monsoon makes consecutive quarters differ by season as much as by activity, while the same quarter a year earlier is seasonally comparable. The 4-quarter lag is validated against the real quarter gap, so a missing quarter cannot produce a mislabelled comparison. NaN where no comparable quarter exists or either side rests on fewer than 2 monthly retrievals.",
    "viirs_rad_yoy_change_abs": "Year-on-year change in quarterly radiance, in radiance UNITS rather than percent. Unaffected by a dim baseline, so it is the right comparison across cells of very different brightness.",
    "viirs_rad_yoy_log_ratio": "log(this quarter's radiance / the same quarter a year earlier). The natural scale for multiplicative growth: symmetric (a halving and a doubling are equal and opposite) and free of the heavy tail that percentage change carries. RECOMMENDED over viirs_rad_yoy_growth_pct for modelling.",
    "viirs_yoy_is_valid": "1 where viirs_rad_yoy_growth_pct is backed by a genuine seasonally-comparable year-earlier quarter with enough monthly retrievals on both sides. Filter on this before using the growth column.",
    # --- cell_quarter_dataset.parquet: the REAL temporal axis -------------
    "quarter_label": "Quarter this row measures, e.g. '2024Q4'. Structural key, not a feature.",
    "quarter_index": "Sortable integer quarter key (year*4 + quarter-1). Used for contiguity checks and temporal splits. Structural, not a feature.",
    "year": "Calendar year of this quarterly observation. Structural.",
    "quarter": "Calendar quarter 1-4. Structural.",
    "ookla_ul_mbps": "REAL Ookla tests-weighted mean upload speed (Mbps) for this cell-quarter.",
    "ookla_devices": "REAL Ookla unique-device count backing this cell-quarter.",
    "ookla_tile_count": "How many distinct Ookla ~610 m tiles fell inside this H3 cell in this quarter. A coverage/reliability indicator.",
    "is_degraded": "Same-quarter absolute degradation label for this cell-quarter (dl < 10 Mbps or rtt > 100 ms). LEAKAGE for a same-quarter target; a legitimate feature for the next-quarter forecast, since it is observed before the quarter being predicted.",
    "dl_percentile_in_quarter": "This cell's download percentile (0-100) among all cells observed in the SAME quarter. Unlike the hourly panel's equivalent, this genuinely moves between periods because each quarter is independently measured.",
    "is_below_area_norm": "1 if dl_percentile_in_quarter < 25 — 'this cell is in the worst quarter of the city this quarter'. Same leakage semantics as is_degraded.",
    "recurrence_to_date": "Fraction of this cell's PREVIOUS observed quarters that were degraded. Expanding mean after shift(1), so the current quarter never counts toward its own feature. This is the proposal's persistent-vs-transient distinction, measured rather than assumed.",
    "quarters_observed_to_date": "How many real quarters of history this cell has BEFORE the current row. Use it to weight or filter recurrence_to_date, which is noisy at small counts.",
    "below_area_norm_rate_to_date": "Fraction of this cell's previous observed quarters spent in the city's bottom 25%. Leakage-free (shift(1) then expanding).",
    "target_is_degraded_next_q": "REAL next-quarter absolute degradation label — measured, not simulated, on both sides. NaN where the following quarter was not observed for this cell; filter on target_next_q_valid.",
    "target_is_below_area_norm_next_q": "REAL next-quarter RELATIVE degradation label (bottom 25% of the city). Recommended primary target for the quarterly forecast: ~22% positive rate versus ~2% for the absolute label.",
    "target_dl_mbps_next_q": "REAL next-quarter download speed (Mbps) — continuous regression target.",
    "target_next_q_valid": "1 when the NEXT observed quarter for this cell is exactly one quarter later, so the target is a true one-step-ahead label. 0 where coverage is missing or this is the cell's final observation. Ookla coverage is uneven, so a naive shift(-1) would silently create 2-quarter-ahead labels — filter on this column before training.",
    "ookla_n_quarters_observed": "How many of the 14 ingested quarters this cell was actually covered by Ookla. 0 means never observed; NaN in the quarterly features rather than an imputed value.",
    "ookla_dl_trend_per_quarter": "Least-squares slope of real quarterly download speed (Mbps per quarter). Emitted only where at least 4 real quarters exist — below that a slope is a line through noise, not an estimate.",
    "ookla_rtt_trend_per_quarter": "Least-squares slope of real quarterly latency (ms per quarter). Same 4-quarter minimum.",
    "ookla_recurrence_rate": "Fraction of this cell's observed quarters that were degraded, across the whole window.",
    "ookla_quarters_degraded": "Count of observed quarters in which this cell was degraded — the real, measurable version of 'this area degrades repeatedly'.",
    "ookla_quarters_below_area_norm": "Count of observed quarters this cell spent in the city's bottom 25%.",
    "ookla_has_multiquarter_history": "True when the cell has at least 4 real quarters, i.e. enough for trend and recurrence to mean something.",
    "qoe_confidence_score": "min(1.0, test_sample_count / 10) — higher sample count = more confidence in the cell-hour's QoE estimate.",
    "hour_of_day": "Local (Asia/Kuala_Lumpur) hour, 0-23, from the row's timestamp. Safe in either target mode. NOTE: the real M-Lab curve puts this city's mobile trough at 19:00-00:00, worst at 22:00 — see is_peak_period.",
    "day_of_week": "Day of week from the row's timestamp, pandas convention (0=Monday ... 6=Sunday). Safe in either target mode.",
    "is_weekend": "1 if day_of_week >= 5 (Saturday/Sunday), else 0. Safe in either target mode.",
    "is_peak_period": "1 during the CONFIGURED commuter peak window (local 08-10 and 17-20 on weekdays), else 0. This is an ASSUMPTION, not a measurement: the real M-Lab curve for Klang Valley troughs at 19:00-00:00 with no morning dip, so this window does not align with observed degradation (data_validator.validate_diurnal_alignment warns about exactly this). Usable as a feature, but do not present it as the measured peak.",
    "in_warmup_window": f"1 for the first {LAG_WARMUP_HOURS} hours of each cell's series, where lag/rolling columns are not yet fully backed by real history (their values are honest NaN and their *_valid flags are 0). Filter these rows out before training rather than treating their NaNs as zeros. Structural flag, not a predictor.",
    "qoe_trend_1h": "lag_1h_dl_median minus rolling_mean_24h_dl_median — short-term deviation from the cell's own recent baseline. Built entirely from already-shifted columns, so it is causal and leakage-free in both target modes.",
    "spatial_rolling_mean_24h_dl_median": "Row-standardized H3 1-ring neighbour average of rolling_mean_24h_dl_median — 'what have this cell's neighbours looked like recently'. Built on an already-lagged column, so it inherits that column's no-leakage guarantee (see spatial_processor.compute_spatial_rolling_mean).",
    "tower_density_per_km2": "Deduplicated OpenCellID site count divided by the cell's real area in km^2 (projected CRS). Complements tower_count by normalising for the small cell-area variation across the grid.",
    "demand_pressure_base": "0-100 composite of ONLY real static features (population, POI, built-up, roads/transport). Zero QoE dependency — safe to use with any target, any mode.",
    "demand_pressure_index_pre_spatial": f"demand_pressure_base plus a historical-QoE component. {HIGH_LEAKAGE_RISK_FEATURES.get('demand_pressure_index_pre_spatial', '')}",
    "demand_pressure_index": f"Final spatially-smoothed demand pressure index. {HIGH_LEAKAGE_RISK_FEATURES.get('demand_pressure_index', '')}",
    "population_night": "Real WorldPop 2020 zonal-statistics residential population estimate for the cell.",
    "population_day": "Modelled daytime population (residential population reweighted by real OSM office/retail density) — explicitly an assumption, not a raster.",
    "population_density_per_km2": "population_night / real cell area (km^2, projected CRS).",
    "built_up_ratio": "Real building footprint area / real H3 cell area (both from GIS geometry, not a hardcoded constant).",
    "tower_count": "Deduplicated approximate PHYSICAL SITE count (OpenCellID records collapsed by ~11m coordinate rounding) — see opencellid_record_count for the raw, non-deduplicated record count.",
    "opencellid_record_count": "Raw OpenCellID record count per cell BEFORE deduplication — includes multiple sectors/technologies/operators at the same physical site. Use tower_count for a physical-site estimate.",
    "tower_source": "Provenance string: 'opencellid_deduplicated' (real) or 'synthetic_fallback_no_file'.",
    "dist_to_nearest_tower_m": "Real geometric nearest-neighbor distance (sjoin_nearest, projected CRS) from the cell centroid to the nearest deduplicated OpenCellID site.",
    "building_count": "Real Google Open Buildings V3 footprint count per cell (confidence >= 0.65 only).",
    "total_building_area_m2": "Real summed footprint area (m^2) per cell.",
    "building_density_per_km2": "building_count / real cell area (km^2).",
    "avg_building_footprint_m2": "total_building_area_m2 / building_count.",
    "poi_education_count": "Real OSM POI count: schools, universities, colleges, kindergartens.",
    "poi_healthcare_count": "Real OSM POI count: hospitals, clinics, pharmacies, doctors.",
    "poi_retail_food_count": "Real OSM POI count: restaurants, cafes, fast food, shops.",
    "poi_office_count": "Real OSM POI count: tagged office=*.",
    "poi_recreation_count": "Real OSM POI count: leisure=* or tourism=*.",
    "poi_transport_stop_count": "Real OSM POI count of public-transport/rail stops joined into the cell.",
    "road_length_m": "Real road length (m) per cell via polygon overlay in a projected CRS (not a heuristic).",
    "road_density_km_per_km2": "road_length_m / 1000 / real cell area (km^2).",
    "intersection_count": "Real OSM road-network node count per cell.",
    "intersection_density_per_km2": "intersection_count / real cell area (km^2).",
    "dist_to_nearest_transit_m": "Real geometric nearest-neighbor distance to the nearest OSM transit/rail POI (sjoin_nearest, projected CRS).",
    "ookla_dl_mbps": "REAL Ookla tests-weighted mean download speed (Mbps) for the cell, static/quarterly. NaN where no tile coverage exists (not imputed).",
    "ookla_rtt_ms": "REAL Ookla tests-weighted mean latency (ms) for the cell, static/quarterly. NaN where no tile coverage exists.",
    "ookla_tests": "REAL summed Ookla test count backing ookla_dl_mbps/ookla_rtt_ms for the cell.",
    "ookla_is_synthetic": "True if the cell fell outside all Ookla tile coverage (no real anchor at all for this cell).",
    "has_real_ookla_label": "True if ookla_dl_mbps is non-null — i.e. this row has a REAL label usable for training in cell_static_qoe_baseline.parquet.",
    "qoe_risk_label": "high_risk/moderate_risk/low_risk from REAL ookla_dl_mbps thresholds. None where has_real_ookla_label is False (deliberately not guessed).",
    "gi_star_zscore_rtt_median": "Real Getis-Ord Gi* z-score for latency (esda.G_Local, star=True) — the full Gi* expectation/variance formulation, not a plain standardized neighbor average.",
    "hotspot_category_rtt_median": "Hot/cold spot classification at 90/95/99% confidence, from real esda.G_Local p-values.",
    "local_moran_quadrant_rtt_median": "HH/LH/LL/HL spatial cluster/outlier classification (real esda.Moran_Local).",
    "historical_hotspot_freq_rtt_median": "Rolling (shift(1)-based, leakage-free) fraction of past hours this cell was a real (esda-confirmed) hotspot.",
    "hotspot_category_rtt_median_encoded": "Ordinal-encoded hotspot confidence band (0=cold_99, 1=cold_95, 2=cold_90, 3=not_significant, 4=hot_90, 5=hot_95, 6=hot_99). ML-compatible integer version of hotspot_category_rtt_median (string, kept for dashboard).",
    "local_moran_quadrant_rtt_median_encoded": "Ordinal-encoded Local Moran's I spatial cluster/outlier class (0=not_significant, 1=LL, 2=LH, 3=HL, 4=HH). ML-compatible integer version of local_moran_quadrant_rtt_median (string, kept for dashboard).",
    "priority_exposure_score": "MinMax-scaled population_day floored at 0.01 — the population-exposure component of the Priority Score. Not an ML feature (derived from population_day).",
    "priority_base_score": "priority_exposure_score × historical_hotspot_freq × max(0.1, qoe_confidence_score) — pre-computed Priority Score base without model risk probability. Dashboard: final_priority = model_predicted_risk × priority_base_score. Not an ML feature.",
}


def _describe_column(col: str) -> str:
    if col in COLUMN_DESCRIPTIONS:
        return COLUMN_DESCRIPTIONS[col]
    if col.endswith("_is_synthetic"):
        return f"True if {col.replace('_is_synthetic', '')} used a synthetic fallback (raw source file missing)."
    if col.startswith("lag_1q_"):
        return f"Value of {col[7:]} in the PREVIOUS quarter, validated for contiguity (NaN where that quarter was not observed). Leakage-free."
    if col.startswith("lag_4q_"):
        return f"Value of {col[7:]} exactly FOUR quarters (one year) earlier, validated for contiguity. Captures year-on-year change. Leakage-free."
    if col.startswith("rolling_mean_4q_"):
        return f"Trailing mean of {col[len('rolling_mean_4q_'):]} over the previous 4 observed quarters, computed after shift(1). Leakage-free."
    if col.startswith("rolling_std_4q_"):
        return f"Trailing standard deviation of {col[len('rolling_std_4q_'):]} over the previous 4 observed quarters (needs >=2 points), after shift(1). A real volatility measure per cell."
    if col.startswith("lag_1h_"):
        return f"Value of {col[7:]} exactly 1 hour prior (shift(1), leakage-free)."
    if col.startswith("lag_2h_"):
        return f"Value of {col[7:]} exactly 2 hours prior (shift(2), leakage-free)."
    if col.startswith("lag_24h_"):
        return f"Value of {col[8:]} exactly 24 hours prior (shift(24), leakage-free)."
    if col.startswith("rolling_mean_"):
        return f"Trailing rolling mean of {col[len('rolling_mean_'):]}, computed after shift(1) (leakage-free)."
    if col.startswith("rolling_std_"):
        return f"Trailing rolling standard deviation of {col[len('rolling_std_'):]}, computed after shift(1) (leakage-free)."
    if col.startswith("spatial_lag_"):
        return f"Row-standardized H3 1-ring neighbor average of {col[12:]} at the same timestamp (real libpysal weights)."
    if col.startswith("spatial_var_"):
        return f"Row-standardized H3 1-ring neighbor variance of {col[12:]} at the same timestamp (real libpysal weights)."
    if col.startswith("gi_star_pvalue_"):
        return f"Permutation-based p-value (99 permutations) for the Getis-Ord Gi* statistic on {col[len('gi_star_pvalue_'):]}."
    if col.startswith("local_moran_i_"):
        return f"Local Moran's I statistic value for {col[len('local_moran_i_'):]} (real esda.Moran_Local)."
    if col.startswith("local_moran_pvalue_"):
        return f"Permutation-based p-value (99 permutations) for Local Moran's I on {col[len('local_moran_pvalue_'):]}."
    if col.endswith("_valid"):
        return "1 if the corresponding lag/rolling/forecast column is backed by real history/future, 0 otherwise (warm-up or end-of-series). Filter to 1 before training on the corresponding column."
    return "Undocumented — flag for the Data/GIS lead rather than guessing; do not assume this column is safe to use as-is."


def build_feature_dictionary(panel_df: pd.DataFrame, static_df: pd.DataFrame,
                              baseline_df: pd.DataFrame, targets_df: pd.DataFrame,
                              quarterly_df: pd.DataFrame = None) -> pd.DataFrame:
    rows = []
    tables = [
        ("cell_time_dataset", panel_df),
        ("cell_static_features", static_df),
        ("cell_static_qoe_baseline", baseline_df),
        ("qoe_targets", targets_df),
    ]
    if quarterly_df is not None:
        tables.append(("cell_quarter_dataset", quarterly_df))
    for name, df in tables:
        # Bug fixed here: this used to fall back to `set(df.columns)` — i.e.
        # "every column is an approved feature" — for any table without a
        # "timestamp" column (cell_static_features.parquet and
        # cell_static_qoe_baseline.parquet, both one-row-per-cell). That
        # meant h3_index, h3_parent_res6, and — on the baseline table — the
        # target itself (ookla_dl_mbps/ookla_rtt_ms/ookla_tests) and its
        # direct derivatives (has_real_ookla_label, qoe_risk_label) were all
        # flagged is_feature_for_current_target=True / is_feature_for_
        # next_hour_target=True in feature_dictionary.xlsx. get_feature_
        # columns() doesn't actually need a "timestamp" column to work
        # correctly — it only ever inspects column names/dtypes — so it's
        # now applied uniformly to every table.
        if name == "cell_quarter_dataset":
            # Quarterly table uses its own rules — see
            # get_quarterly_feature_columns. "current" maps to the
            # same-quarter target, "next" to the next-quarter forecast.
            current_features = set(get_quarterly_feature_columns(df, "current_quarter"))
            next_hour_features = set(get_quarterly_feature_columns(df, "next_quarter"))
        else:
            current_features = set(get_feature_columns(df, "current"))
            next_hour_features = set(get_feature_columns(df, "next_hour"))
        for col in df.columns:
            rows.append({
                "file": name,
                "column": col,
                "dtype": str(df[col].dtype),
                "pct_non_null": round(100 * df[col].notna().mean(), 2),
                "is_feature_for_current_target": col in current_features,
                "is_feature_for_next_hour_target": col in next_hour_features,
                "leakage_risk_note": HIGH_LEAKAGE_RISK_FEATURES.get(col, ""),
                "description": _describe_column(col),
            })
    return pd.DataFrame(rows)


def merge_static_new_layers(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold the real VIIRS nighttime-lights and ESA WorldCover land-cover
    layers into the static feature table.

    WHY THIS LIVES HERE NOW: this merge used to run in
    run_full_data_pipeline.py, AFTER run_pipeline.main() had already built
    and exported cell_static_qoe_baseline.parquet, the hourly panel and
    the feature dictionary. The 24 real columns therefore reached
    cell_static_features.parquet and nothing else — not the modelling
    dataset, not the panel, not the dictionary the README tells the
    modelling team to query. A repo-wide search for these column names
    outside their own extraction scripts returned zero consumers.

    Running it here, before build_static_qoe_baseline(), puts them into
    every downstream table at once.
    """
    path = DERIVED_LAYERS["static_new_layers"]
    if not path.exists():
        logger.warning(
            "%s not found — skipping the VIIRS/WorldCover merge. Run "
            "scripts/extract/viirs.py, scripts/extract/worldcover.py and scripts/extract/merge_layers.py "
            "to generate it. The pipeline will still complete, but without "
            "nighttime lights or land cover in ANY output.",
            path,
        )
        return static_df

    new_layers = pd.read_parquet(path)
    missing_cells = set(static_df["h3_index"]) - set(new_layers["h3_index"])
    if missing_cells:
        raise ValueError(
            f"{len(missing_cells)} grid cells have no row in {path.name} — the "
            "H3 grid has changed since that extraction ran. Re-run "
            "scripts/extract/viirs.py / scripts/extract/worldcover.py / scripts/extract/merge_layers.py "
            "against the current grid rather than merging a stale layer."
        )

    before = len(static_df)
    merged = static_df.merge(new_layers, on="h3_index", how="left")
    if len(merged) != before:
        raise ValueError(
            f"Row count changed during the new-layer merge ({before} -> {len(merged)}) — "
            "duplicate h3_index values in the extraction output."
        )
    for required in ("viirs_avg_rad_mean", "landcover_built_up_pct"):
        if merged[required].isna().any():
            raise ValueError(f"{required} has nulls after the merge — extraction coverage is incomplete.")

    logger.info(
        "Merged real VIIRS + WorldCover layers: %d new columns, 100%% cell coverage. "
        "These now flow into the static QoE baseline, the hourly panel and the feature dictionary.",
        len(new_layers.columns) - 1,
    )
    return merged


def merge_derived_layer(static_df: pd.DataFrame, key: str, label: str,
                         coverage_col: str = None) -> pd.DataFrame:
    """
    Merge one registered derived layer, forwarding EVERY column it
    produces except the join key.

    THIS IS DELIBERATELY AN EXCLUSION MERGE, NOT AN INCLUSION MERGE.
    Three separate columns-lost bugs in this project came from a merge
    that named the columns to keep: the mangrove land-cover class, the
    six monthly VIIRS composites, and the original 24 VIIRS/WorldCover/
    Sunway columns. Each was real signal, correctly extracted, then
    silently dropped by a hand-maintained keep-list that nobody updated
    when the extractor grew a column. Forwarding everything and dropping
    nothing makes that class of bug impossible here, and
    data_validator.validate_derived_layer_passthrough checks it held.
    """
    path = DERIVED_LAYERS.get(key)
    if path is None or not path.exists():
        logger.warning("%s not found — skipping the %s layer.", path, label)
        return static_df

    layer = pd.read_parquet(path)
    missing_cells = set(static_df["h3_index"]) - set(layer["h3_index"])
    if missing_cells:
        raise ValueError(
            f"{len(missing_cells)} grid cells have no row in {path.name} — the H3 grid "
            f"changed since that extraction ran. Re-run it against the current grid."
        )

    before = len(static_df)
    merged = static_df.merge(layer, on="h3_index", how="left")
    if len(merged) != before:
        raise ValueError(f"Row count changed merging {label} ({before} -> {len(merged)}).")

    n_new = len(layer.columns) - 1
    if coverage_col and coverage_col in merged.columns:
        n_cov = int(merged[coverage_col].notna().sum())
        logger.info("%s: %d new columns; %d of %d cells (%.1f%%) have a value, rest NaN (never imputed).",
                    label, n_new, n_cov, len(merged), 100 * n_cov / len(merged))
    else:
        logger.info("%s: %d new columns merged.", label, n_new)
    return merged


def merge_building_heights(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join per-cell OSM building-height statistics. Vertical density is a
    better indoor-demand proxy than footprint area: two cells with the same
    built-up ratio differ enormously if one is warehouses and the other is
    towers.

    SPARSE AND BIASED — both flags travel with the data. Only 6.5% of
    buildings carry a height or storey tag, so 79% of cells get NaN, and
    the tagged ones cluster where OSM mappers are active (city centre,
    malls, landmarks). That is the same non-random-coverage trap as the
    Ookla labels: a model can read this as "is this a well-mapped area"
    rather than "how tall is it". Never impute the NaNs.
    """
    path = DERIVED_LAYERS.get("building_heights")
    if path is None or not path.exists():
        logger.warning(
            "%s not found — skipping building heights. Run "
            "`python scripts/extract/building_heights.py` (no download needed, it reads "
            "the OSM PBF already in data/raw/).", path,
        )
        return static_df

    heights = pd.read_parquet(path)
    before = len(static_df)
    merged = static_df.merge(heights, on="h3_index", how="left")
    if len(merged) != before:
        raise ValueError(f"Row count changed during the building-height merge ({before} -> {len(merged)}).")

    n_cov = int((merged["building_n_with_height"] > 0).sum())
    logger.info(
        "Building heights: %d of %d cells (%.1f%%) have at least one height-tagged building; "
        "the rest are NaN, never imputed. Tag coverage is %.2f%% of buildings and is biased "
        "toward well-mapped areas — flagged by building_height_coverage_is_biased.",
        n_cov, len(merged), 100 * n_cov / len(merged),
        100 * float(merged["building_height_tagged_fraction"].iloc[0]),
    )
    return merged


def merge_sunway_inset(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join the real Sunway/Subang Jaya field-measurement aggregation as a
    high-resolution inset. NaN outside the ~10 cells it covers — that is
    honest sparsity, never imputed.
    """
    path = DERIVED_LAYERS["sunway_by_h3"]
    if not path.exists():
        logger.warning(
            "%s not found — skipping the Sunway inset merge. Run "
            "`python -m src.data_preprocessing.ingest_sunway_real` to generate it.",
            path,
        )
        return static_df

    sunway_agg = pd.read_parquet(path)
    before = len(static_df)
    merged = static_df.merge(sunway_agg, on="h3_index", how="left")
    if len(merged) != before:
        raise ValueError(f"Row count changed during the Sunway merge ({before} -> {len(merged)}).")

    n_covered = int(merged["sunway_n_records"].notna().sum())
    logger.info(
        "Joined real Sunway inset: %d of %d cells (%.2f%%) have real field ground truth; "
        "the rest correctly show NaN for sunway_* columns.",
        n_covered, len(merged), 100 * n_covered / len(merged),
    )
    return merged


def summarize_sunway_ground_truth(static_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    """
    Report WHERE independent real ground truth exists, without asserting a
    comparison the data cannot currently support.

    WITHDRAWN CLAIM: a previous version published a Pearson correlation
    (-0.599) and a "mean absolute % error vs Sunway" (66.7%) between
    ookla_dl_mbps and what was then called `sunway_test_event_dl_mbps`,
    described in the output JSON as "real evidence about Ookla's
    reliability as a wide-area proxy". That comparison was invalid on two
    independent grounds:

      1. UNIT. The Sunway column is the raw GNetTrack TESTDOWNLINK value
         with no conversion, and its unit is unverified — TESTUPLINK reads
         ~10x higher than TESTDOWNLINK, and read as Mbps the values exceed
         the same dataset's own passive throughput peak in the same cells.
         See ingest_sunway_real.aggregate_sunway_to_h3's docstring.
      2. SAMPLE SIZE. n = 8 overlapping cells, median 8 real test events
         each. No correlation computed on that is reportable in either
         direction, unit question aside.

    Publishing a wrong number about your own primary data source is worse
    than publishing none, so the correlation is gone. What remains is the
    true and still-valuable statement: independent real ground truth
    exists here, and here is exactly how much.
    """
    n_cells = len(static_df)
    has_sunway = static_df["sunway_n_records"].notna() if "sunway_n_records" in static_df.columns else None

    summary = {
        "status": "CORRELATION_WITHDRAWN_PENDING_UNIT_VERIFICATION",
        "why_withdrawn": (
            "sunway_test_event_dl_raw carries the raw GNetTrack TESTDOWNLINK value with an "
            "unverified unit (TESTUPLINK reads ~10x higher than TESTDOWNLINK; read as Mbps the "
            "values exceed this dataset's own passive throughput peak of 103.9 Mbps in the same "
            "cells). Separately, only 8 cells overlap with real Ookla coverage, at a median of 8 "
            "test events each. Neither the unit nor the sample size supports a correlation claim."
        ),
        "to_reinstate": (
            "Resolve TESTDOWNLINK against the GNetTrack Pro field specification, apply the "
            "conversion in ingest_sunway_real.load_real_sunway_points, then report the comparison "
            "as DESCRIPTIVE at n=8 — never as validation of Ookla."
        ),
    }

    if has_sunway is not None:
        covered = static_df.loc[has_sunway]
        summary["independent_ground_truth_coverage"] = {
            "cells_with_real_field_measurements": int(has_sunway.sum()),
            "cells_total": n_cells,
            "total_real_point_measurements": int(covered["sunway_n_records"].sum()),
            "cells_with_real_ping_samples": int((covered["sunway_n_real_ping_samples"] > 0).sum()),
            "cells_with_real_test_events": int((covered["sunway_n_real_test_events"] > 0).sum()),
            "statement": (
                "Independent real multi-operator field measurements exist for these cells and "
                "nowhere else in the study area. This is a genuine, disclosable coverage fact and "
                "does not depend on the unresolved unit question."
            ),
        }
        if "ookla_dl_mbps" in baseline_df.columns:
            overlap = covered[["h3_index"]].merge(
                baseline_df.loc[baseline_df["ookla_dl_mbps"].notna(), ["h3_index"]], on="h3_index"
            )
            summary["independent_ground_truth_coverage"]["cells_overlapping_real_ookla"] = len(overlap)

    return summary


def assert_feature_dictionary_complete(feature_dict_df: pd.DataFrame, tables: dict) -> None:
    """
    Hard guard against the dictionary drifting out of sync with the data.

    The README instructs the modelling team to query feature_dictionary.xlsx
    rather than maintain a manual exclusion list. That contract broke
    silently once already: the dictionary described cell_static_features as
    31 columns while the exported file had 55, so following the documented
    workflow discarded 24 real features without any error. This makes that
    class of drift impossible to ship.
    """
    problems = []
    for name, df in tables.items():
        documented = set(feature_dict_df.loc[feature_dict_df["file"] == name, "column"])
        actual = set(df.columns)
        undocumented = actual - documented
        phantom = documented - actual
        if undocumented:
            problems.append(f"{name}: {len(undocumented)} column(s) in the data but NOT in the "
                            f"dictionary: {sorted(undocumented)}")
        if phantom:
            problems.append(f"{name}: {len(phantom)} column(s) in the dictionary but NOT in the "
                            f"data: {sorted(phantom)}")
    if problems:
        raise ValueError(
            "feature_dictionary.xlsx is out of sync with the exported tables:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    undescribed = feature_dict_df["description"].str.startswith("Undocumented").sum()
    if undescribed:
        cols = feature_dict_df.loc[
            feature_dict_df["description"].str.startswith("Undocumented"), "column"
        ].unique()
        logger.warning(
            "%d dictionary row(s) have no written description (columns: %s). Add them to "
            "COLUMN_DESCRIPTIONS — a modelling team reading the dictionary will otherwise see "
            "'do not assume this column is safe to use as-is'.",
            undescribed, sorted(cols),
        )
    logger.info(
        "Feature dictionary completeness check OK: %d columns documented across %d tables.",
        len(feature_dict_df), len(tables),
    )


def main(days: int = 7):
    t_start = time.time()
    np.random.seed(GLOBAL_RANDOM_SEED)  # single global seed, set once, before ANY randomness
    logger.info("=== GeoQoE Pulse Data & GIS Pipeline v%s ===", PIPELINE_VERSION)
    logger.info(SIMULATION_DISCLAIMER)

    # --- 1. Real H3 study grid -------------------------------------------------
    logger.info("[1/10] Generating H3 study grid from real, trimmed study boundary...")
    grid_gdf = generate_h3_study_grid(res=H3_PRIMARY_RESOLUTION)
    export_grid_geometry(grid_gdf, PROCESSED_DIR)

    # --- 2. Real spatial joins ---------------------------------------------------
    logger.info("[2/10] Real spatial joins: deduplicated towers, buildings, OSM, WorldPop, DOSM...")
    static_df = build_static_features(grid_gdf)

    logger.info("[2a/10] Assigning cells to districts (location hierarchy + demand join key)...")
    cell_districts = assign_cells_to_districts(grid_gdf)
    static_df = static_df.merge(cell_districts, on="h3_index", how="left")

    logger.info("[2a/10] Merging real VIIRS + WorldCover static layers (BEFORE the baseline is built)...")
    static_df = merge_static_new_layers(static_df)

    logger.info("[2a/10] Merging real Sunway field-measurement inset...")
    static_df = merge_sunway_inset(static_df)

    logger.info("[2a/10] Merging real OSM building heights (from the PBF already on disk)...")
    static_df = merge_building_heights(static_df)

    logger.info("[2a/10] Merging real Copernicus DEM terrain statistics...")
    static_df = merge_derived_layer(static_df, "dem", "Copernicus DEM terrain",
                                     coverage_col="elevation_mean_m")

    logger.info("[2b/10] Real Ookla spatial join (static per-cell baseline)...")
    ookla_df = load_ookla(grid_gdf)

    logger.info("[2c/10] Building the HONEST static QoE baseline (real Ookla labels, no fabrication)...")
    baseline_df = build_static_qoe_baseline(static_df, ookla_df)
    # Coverage-bias tooling: Ookla coverage is highly predictable from the
    # GIS features, so training on labelled cells and predicting the rest
    # is extrapolation. Ship the columns to weight and flag it, not just a
    # README warning.
    baseline_df = compute_coverage_bias_columns(baseline_df)
    # Per-application QoE profiles — computed on REAL tables only. The
    # hourly panel's jitter/loss are generated, so scoring voice quality
    # against them would measure the formula, not the network.
    baseline_df = compute_application_profiles(baseline_df, ul_col="__none__")

    logger.info("[2e/10] Building the REAL multi-quarter Ookla panel (the project's genuine temporal axis)...")
    quarterly_df = None
    if QUARTERLY_PANEL_PATH.exists():
        quarterly_panel = pd.read_parquet(QUARTERLY_PANEL_PATH)
        quarterly_df = build_quarterly_ml_dataset(quarterly_panel, static_df)
        # First demand feature in the project that varies over TIME as well
        # as space. District resolution (8 districts over 4,918 cells), real
        # DOSM annual series 2020-2025, joined by year of quarter.
        try:
            _pop_features = build_district_population_features(load_district_population())
            quarterly_df = attach_to_quarterly_panel(quarterly_df, cell_districts, _pop_features)
        except FileNotFoundError as exc:
            logger.warning(
                "District population unavailable (%s) — the quarterly panel loses its "
                "district-level time-varying demand feature.", exc,
            )
        # Per-CELL per-quarter nighttime lights: the finest-resolution
        # time-varying demand proxy in the project. Requires the full 43-month
        # VIIRS backfill; without it the layer is simply absent, never faked.
        try:
            _viirs_q = build_viirs_growth_features(load_viirs_quarterly())
            quarterly_df = attach_viirs_quarterly(quarterly_df, _viirs_q)
        except FileNotFoundError as exc:
            logger.warning(
                "VIIRS quarterly unavailable (%s) — demand-over-time remains district-level "
                "only. Run scripts/extract/viirs.py with the full monthly backfill to enable it.", exc,
            )

        # Per-application QoE profiles, after every QoE column is in place.
        quarterly_df = compute_application_profiles(quarterly_df)
    else:
        logger.warning(
            "%s not found — skipping the real quarterly dataset. Run "
            "`python -m src.data_preprocessing.ookla_quarterly` style ingestion first "
            "(see scripts). Without it the project has NO real temporal axis at all.",
            QUARTERLY_PANEL_PATH,
        )

    logger.info("[2d/10] Assembling hourly panel (%d days) for dashboard/demo use...", days)
    panel_df = build_dynamic_panel(grid_gdf, static_df, ookla_df, days=days)

    # --- 3. Temporal feature engineering (leakage-free) --------------------------
    logger.info("[3/10] Temporal features (leakage-free lags/rolling stats)...")
    panel_df = compute_temporal_shifted_features(panel_df)

    # Bring static features into the panel so per-timestamp spatial lags of
    # e.g. population are possible, and so the exported panel is self-contained.
    # Merge static context into the hourly panel — but NOT the full static
    # table. Static features are constant per cell, so every column merged
    # here is duplicated 168 times. At 86 static columns that is ~570 MB of
    # redundant float64 for a table that is explicitly a DEMO FIXTURE, and
    # it exhausted memory outright once the DEM, building-height and
    # 43-month VIIRS layers landed.
    #
    # The heavy spatial-modelling layers are therefore excluded HERE and
    # only here. They are not lost: cell_static_features.parquet carries
    # every one of them, joinable on h3_index in a single line, and the
    # derived-layer passthrough guard still verifies they all arrive there.
    # This is a deliberate, documented subset for one fixture table — not
    # the silent keep-list drop that cost this project the mangrove class.
    PANEL_STATIC_EXCLUDE_PREFIXES = (
        "landcover_", "worldcover_", "sunway_", "viirs_", "dem_",
        "elevation_", "terrain_", "building_height_",
    )
    static_cols_for_panel = [
        c for c in static_df.columns
        if c != "h3_index" and not c.startswith(PANEL_STATIC_EXCLUDE_PREFIXES)
    ]
    n_excluded = len(static_df.columns) - 1 - len(static_cols_for_panel)
    logger.info(
        "Panel static merge: %d of %d static columns carried into the hourly fixture; %d "
        "spatial-modelling columns excluded to keep it in memory (they remain in "
        "cell_static_features.parquet, joinable on h3_index).",
        len(static_cols_for_panel), len(static_df.columns) - 1, n_excluded,
    )
    # The exclusion above is structurally the same shape as the bug that
    # cost this project the mangrove class: a merge that decides which
    # columns travel. The difference must be verifiable, not asserted — so
    # every column withheld from the panel is checked to exist in the
    # static table that IS exported. If that ever fails, a column has gone
    # missing from both places and the build stops.
    _withheld = [c for c in static_df.columns
                 if c != "h3_index" and c not in static_cols_for_panel]
    _lost = [c for c in _withheld if c not in static_df.columns]
    if _lost:
        raise ValueError(
            f"{len(_lost)} column(s) withheld from the hourly panel are also absent from "
            f"cell_static_features: {_lost}. Withholding is only acceptable while the column "
            "remains available elsewhere."
        )
    logger.info(
        "  %d withheld column(s) verified present in cell_static_features.parquet — "
        "available via a one-line join on h3_index, not lost.", len(_withheld),
    )
    panel_df = panel_df.merge(static_df[["h3_index"] + static_cols_for_panel], on="h3_index", how="left")

    # --- 4. Demand pressure index (real features; historical-QoE term flagged) ---
    logger.info("[4/10] Demand pressure index (real static features + flagged historical-QoE term)...")
    demand_base_df = compute_demand_pressure_base(static_df)
    panel_df = apply_dynamic_demand_adjustments(panel_df, demand_base_df)
    static_df = static_df.merge(demand_base_df, on="h3_index", how="left")  # also export in static features

    # --- 5. Spatial weights + spatial features ------------------------------------
    logger.info("[5/10] Building real libpysal spatial weights (H3 1-ring adjacency)...")
    w, h3_order = build_h3_spatial_weights(grid_gdf["h3_index"])

    logger.info("[5b/10] Spatial lags, spatial variance, spatial rolling mean...")
    panel_df = compute_vectorized_spatial_lags(
        panel_df, w, h3_order,
        target_cols=["dl_median", "rtt_median", "population_night", "demand_pressure_index_pre_spatial"],
    )
    panel_df = finalize_demand_pressure_index(panel_df)
    panel_df = compute_spatial_rolling_mean(panel_df, w, h3_order, col="dl_median", window=24)

    # --- 6. Hotspot detection (real esda implementations) -------------------------
    logger.info("[6/10] Getis-Ord Gi* + Local Moran's I (esda, real statistics with p-values)...")
    panel_df = compute_hotspots(panel_df, w, h3_order, col="rtt_median", permutations=99)
    panel_df = compute_local_moran(panel_df, w, h3_order, col="rtt_median", permutations=99)
    panel_df = compute_historical_hotspot_frequency(panel_df, col="rtt_median", window=168)

    # --- 6b. Encode categorical string columns for ML compatibility -----------
    logger.info("[6b/10] Encoding categorical columns (ordinal integers for ML, strings kept for dashboard)...")
    for _cat_col, _cat_mapping in CATEGORICAL_ENCODINGS.items():
        if _cat_col in panel_df.columns:
            panel_df[f"{_cat_col}_encoded"] = panel_df[_cat_col].map(_cat_mapping).fillna(3).astype(int)

    # --- 7. QoE targets: same-hour AND next-hour forecast ---------------------------
    logger.info("[7/10] QoE targets (unified thresholds; same-hour + %dh-ahead forecast)...", FORECAST_HORIZON_HOURS)
    panel_df = compute_qoe_targets(panel_df)
    # Relative (cross-cell, per-timestamp) degradation, alongside the
    # absolute threshold — see qoe_target.compute_relative_targets for why
    # the absolute 10 Mbps cutoff cannot express localised degradation in a
    # city whose real Ookla median is 175 Mbps.
    panel_df = compute_relative_targets(panel_df)
    panel_df = compute_next_hour_targets(panel_df, horizon_hours=FORECAST_HORIZON_HOURS)

    # --- 7b. Pre-compute priority score components ----------------------------
    logger.info("[7b/10] Pre-computing priority score components (exposure × recurrence × confidence)...")
    _pop = panel_df["population_day"].copy()
    _pmin, _pmax = _pop.min(), _pop.max()
    panel_df["priority_exposure_score"] = np.clip(
        (_pop - _pmin) / max(_pmax - _pmin, 1e-9), 0.01, 1.0
    )
    panel_df["priority_base_score"] = (
        panel_df["priority_exposure_score"]
        * panel_df["historical_hotspot_freq_rtt_median"].fillna(0.0)
        * panel_df["qoe_confidence_score"].clip(lower=0.1)
    )

    panel_df = panel_df.sort_values(["h3_index", "timestamp"]).reset_index(drop=True)
    panel_df.insert(0, "row_id", panel_df.index)

    suffix = f"next_{FORECAST_HORIZON_HOURS}h"
    targets_df = panel_df[[
        "row_id", "h3_index", "timestamp",
        "target_qoe_index", "target_service_quality_1to5", "target_is_degraded", "target_degradation_class",
        f"target_is_degraded_{suffix}", f"target_degradation_class_{suffix}", f"target_qoe_index_{suffix}",
        f"target_{suffix}_valid",
        "target_qoe_percentile_in_area", "target_is_below_area_norm",
        f"target_is_below_area_norm_{suffix}", f"target_qoe_percentile_in_area_{suffix}",
        "qoe_confidence_score",
        # Carried through so run_full_validation can audit whether the
        # degradation target is trivially predictable from anchor status
        # alone (see data_validator.validate_target_anchor_correlation) —
        # without these, that check would need a separate merge back to
        # panel_df every time targets_df is used on its own.
        "qoe_level_is_ookla_anchored", "qoe_baseline_source",
    ]].copy()

    # --- 8. Validation ----------------------------------------------------------------
    logger.info("[8/10] Running automated data validation (incl. target-consistency guards)...")
    report = run_full_validation(grid_gdf, static_df, panel_df, targets_df)
    if report.warnings:
        logger.warning("Validation passed with %d warning(s) — see log above for details.", len(report.warnings))

    # --- 9. Feature dictionary --------------------------------------------------------
    logger.info("[9/10] Building feature dictionary (per-target-mode feature flags + leakage notes)...")
    feature_dict_df = build_feature_dictionary(panel_df, static_df, baseline_df, targets_df, quarterly_df)
    _dict_tables = {
        "cell_time_dataset": panel_df,
        "cell_static_features": static_df,
        "cell_static_qoe_baseline": baseline_df,
        "qoe_targets": targets_df,
    }
    if quarterly_df is not None:
        _dict_tables["cell_quarter_dataset"] = quarterly_df
    assert_feature_dictionary_complete(feature_dict_df, _dict_tables)

    # --- 10. Export ---------------------------------------------------------------------
    logger.info("[10/10] Exporting ML-ready datasets, feature dictionary, and metadata...")
    panel_path = OUTPUT_DIR / "cell_time_dataset.parquet"
    static_path = OUTPUT_DIR / "cell_static_features.parquet"
    baseline_path = OUTPUT_DIR / "cell_static_qoe_baseline.parquet"
    targets_path = OUTPUT_DIR / "qoe_targets.parquet"
    quarterly_path = OUTPUT_DIR / "cell_quarter_dataset.parquet"

    panel_df.to_parquet(panel_path, index=False, compression="zstd")
    static_df.to_parquet(static_path, index=False, compression="zstd")
    baseline_df.to_parquet(baseline_path, index=False, compression="zstd")
    targets_df.to_parquet(targets_path, index=False, compression="zstd")
    if quarterly_df is not None:
        quarterly_df.to_parquet(quarterly_path, index=False, compression="zstd")
    feature_dict_df.to_excel(OUTPUT_DIR / "feature_dictionary.xlsx", index=False)

    # Persist the validation report instead of discarding it, and record
    # where independent real ground truth exists (replacing the withdrawn
    # Ookla-vs-Sunway correlation — see summarize_sunway_ground_truth).
    quality_report = {
        "pipeline_version": PIPELINE_VERSION,
        "validation": report.summary(),
        "sunway_ground_truth": summarize_sunway_ground_truth(static_df, baseline_df),
    }
    with open(OUTPUT_DIR / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2, default=str)
    logger.info(
        "Wrote data_quality_report.json (%d error(s), %d warning(s)) — previously the validator "
        "built this object and discarded it.",
        quality_report["validation"]["n_errors"], quality_report["validation"]["n_warnings"],
    )

    synthetic_summary = summarize_synthetic_fraction(static_df, panel_df)
    generate_metadata(
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        feature_columns_current=get_feature_columns(panel_df, "current"),
        feature_columns_next_hour=get_feature_columns(panel_df, "next_hour"),
        row_counts={"grid_cells": len(grid_gdf), "static_rows": len(static_df),
                    "panel_rows": len(panel_df), "target_rows": len(targets_df),
                    "static_qoe_baseline_rows": len(baseline_df),
                    "static_qoe_baseline_real_labels": int(baseline_df["has_real_ookla_label"].sum()),
                    "quarterly_rows": int(len(quarterly_df)) if quarterly_df is not None else 0,
                    "quarterly_cells": int(quarterly_df["h3_index"].nunique()) if quarterly_df is not None else 0,
                    "quarterly_rows_with_real_next_q_target": int(quarterly_df["target_next_q_valid"].sum()) if quarterly_df is not None else 0},
        synthetic_summary=synthetic_summary,
        fitted_scalers=dict(FITTED_SCALERS),
        coverage_bias=baseline_df.attrs.get('coverage_bias_diagnostics'),
        application_profiles=profile_metadata(),
    )

    elapsed = time.time() - t_start
    logger.info(
        "=== Pipeline complete in %.1fs — %d cells x %d hours = %d panel rows (SIMULATED), "
        "%d cells in the static QoE baseline (%d with a REAL Ookla label). "
        "Synthetic-fallback fractions: %s ===",
        elapsed, len(grid_gdf), len(panel_df) // len(grid_gdf), len(panel_df),
        len(baseline_df), int(baseline_df["has_real_ookla_label"].sum()), synthetic_summary,
    )
    return grid_gdf, static_df, panel_df, baseline_df, targets_df


if __name__ == "__main__":
    main()
