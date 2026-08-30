"""
GeoQoE Pulse Configuration File
Central configuration for CRS, H3 resolution, spatial bounds, study boundary,
reproducibility, and feature weights.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Base Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BOUNDARY_DIR = DATA_DIR / "boundary"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Ensure directories exist
for folder in [RAW_DIR, BOUNDARY_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Spatial Reference Systems
# --------------------------------------------------------------------------
# WGS84 — used for all storage/export (lat/lon, web-map compatible)
CRS_WGS84 = "EPSG:4326"
# GDM2000 / Peninsular Malaysia RSO — projected, metres. Used for any
# distance/area calculation (H3 cell area, nearest-tower distance, road
# length, building density) since WGS84 degrees are not equal-distance.
CRS_PENINSULAR_MYS = "EPSG:3375"
CRS_EAST_MYS = "EPSG:3376"  # GDM2000 / East Malaysia BRSO (not used — Klang Valley only)

# Rule of thumb used throughout src/data_preprocessing/:
#   - Store & export geometry in CRS_WGS84.
#   - Reproject to CRS_PENINSULAR_MYS ("to_crs") immediately before any
#     .area / .length / .distance call, then reproject results (not
#     geometries) back if needed. Never compute area/length/distance in
#     raw WGS84 degrees.

# --------------------------------------------------------------------------
# Study Boundary
# --------------------------------------------------------------------------
# Real GADM level-2 administrative polygon (Selangor districts + Putrajaya),
# dissolved with one additional real state-level polygon merged in for KL
# (GADM has no level-2 subdivision for a federal territory). As of this
# version it covers the Klang Valley CORE districts:
#   Gombak, Hulu Langat, Klang, Petaling, Sepang, Kuala Langat, Putrajaya,
#   Kuala Lumpur
# Two districts that are NOT part of any common "Klang Valley" definition —
# Hulu Selangor (rural/interior) and Sabak Bernam (rural, north coast) — were
# removed from the original full-Selangor GADM extract (see
# scripts/trim_boundary.py) because they diluted hotspot detection with very
# low-density, low-POI cells and were never part of the intended study area.
#
# FIXED (previously a known gap): Kuala Lumpur (W.P. Kuala Lumpur) was not
# included in earlier versions — no KL boundary file was available when the
# original boundary was built. A real WP Kuala Lumpur polygon (source:
# nullifye/malaysia.geojson, state code KUL; state-level administrative
# boundary, not GADM level-2, since KL has no further subdivision in GADM)
# has since been merged into data/boundary/study_boundary.geojson. Sanity
# checked with point-in-polygon: KLCC (3.1579, 101.7123) and KL Sentral
# (3.1339, 101.6865) fall inside; Batu Caves (3.2379, 101.6839, in Gombak)
# and Petaling Jaya (3.1073, 101.6067, in Petaling) fall outside — i.e. it's
# KL proper, not an oversized box, and it doesn't overlap the existing
# Selangor/Putrajaya polygons. This added 259 real H3 cells (~222 km^2,
# 4,659 -> 4,918 at res 8) that were previously invisible to every layer in
# the pipeline. KL is the country's single highest mobile-demand area, so
# every hotspot/demand result from before this fix should be treated as
# incomplete, not just "missing a bit of coverage."
STUDY_BOUNDARY_PATH = BOUNDARY_DIR / "study_boundary.geojson"

# Defensive filter: if a raw, unfiltered Selangor-wide GADM extract is ever
# reloaded (e.g. someone re-downloads NAME_1 == "Selangor" from scratch),
# geometry_utils.load_study_boundary() uses this list to automatically drop
# out-of-scope districts rather than silently including all of Selangor
# again. Matched against the GADM "NAME_2" column (falls back to "NAME_1"
# for Putrajaya and Kuala Lumpur, neither of which has a NAME_2).
KLANG_VALLEY_DISTRICT_NAMES = {
    # Matched against GADM's actual NAME_2 spelling (no spaces), plus the
    # NAME_1/NAME_2 values used by the merged KL polygon (see
    # STUDY_BOUNDARY_PATH comment above) and a couple of formatting variants
    # so the filter doesn't silently drop a district over a spelling
    # difference if the boundary file is ever regenerated from a fresh
    # source.
    "Gombak", "HuluLangat", "Klang", "Petaling", "Sepang", "KualaLangat",
    "Putrajaya", "KualaLumpur", "Kuala Lumpur",
}
STUDY_AREA_LABEL = (
    "Klang Valley core: Kuala Lumpur federal territory, Putrajaya federal "
    "territory, and 6 urbanised Selangor districts (Gombak, Hulu Langat, "
    "Klang, Kuala Langat, Petaling, Sepang). Deliberately excludes rural/"
    "exurban Selangor (Sabak Bernam, Hulu Selangor, etc.)."
)

# Fallback bounding box — only used if STUDY_BOUNDARY_PATH is missing, and a
# clear warning is logged whenever this fallback fires. Widened to match the
# real trimmed Klang Valley core boundary's actual extent (previously this
# was a much smaller box that had also — incorrectly — been reused as the
# raw-file extraction filter in data_downloader.py; that usage is now fixed
# to derive extraction bounds from the real grid via extraction_bounds(),
# so this box is ONLY the last-resort grid-generation fallback).
STUDY_BBOX = {
    "min_lon": 101.15,
    "min_lat": 2.55,
    "max_lon": 102.02,
    "max_lat": 3.45,
}

# --------------------------------------------------------------------------
# H3 Hexagonal Grid
# --------------------------------------------------------------------------
H3_PRIMARY_RESOLUTION = 8    # ~0.74 km^2 cell area, ~461m edge
H3_SECONDARY_RESOLUTION = 9  # ~0.11 km^2 cell area, ~174m edge

# NEW: coarse H3 parent resolution used purely to stamp a spatial-holdout
# grouping key (`h3_parent_res6`) onto every output row. Nothing in the
# pipeline previously exposed a grouping column, so a real spatial
# train/test split (hold out whole groups, not individual res-8 cells)
# required deriving h3.cell_to_parent() by hand downstream. Res 6 gives
# ~130 groups over the current (post-KL) grid, each several km across —
# coarse enough that res-8 cells inside a held-out group are genuinely
# unseen, not just nominally different h3_index values that a random split
# could still leak across via spatial autocorrelation.
SPATIAL_HOLDOUT_RESOLUTION = 6

# --------------------------------------------------------------------------
# Temporal Configuration
# --------------------------------------------------------------------------
TIME_FREQ = "1h"
ROLLING_WINDOW_24H = 24
ROLLING_WINDOW_7D = 168
# Rows within this many hours of each cell's series start are a "warm-up"
# window: rolling/lag features are not yet fully backed by real history.
LAG_WARMUP_HOURS = 24

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# Single global seed applied once, at the very start of the pipeline, before
# any random sampling (real-data fallback generation included). Do not seed
# anywhere else — re-seeding mid-pipeline silently breaks the intended
# reproducibility guarantee.
GLOBAL_RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Raw Dataset Registry
# --------------------------------------------------------------------------
# Central place documenting expected raw filenames/schemas, so ingestion
# code and this config never drift apart.
RAW_DATASETS = {
    "ookla": RAW_DIR / "2024-10-01_performance_mobile_tiles.parquet",
    "buildings": RAW_DIR / "31d_buildings.csv.gz",
    "towers": RAW_DIR / "502.csv.gz",
    "osm": RAW_DIR / "malaysia-singapore-brunei-260727.osm.pbf",
    "worldpop": RAW_DIR / "mys_ppp_2020_UNadj.tif",
    "dosm_population": RAW_DIR / "population_malaysia.csv",
    "ridership": RAW_DIR / "ridership_headline.csv",
    # Registered so they are checksummed and fingerprinted in metadata.json
    # like the original seven. Previously these two real sources were read
    # by hardcoded absolute paths inside their own modules and appeared
    # nowhere in the provenance record.
    "mlab_diurnal_curve": RAW_DIR / "mlab.csv",
    "sunway_qoe": RAW_DIR / "sunway_qoe.csv",
    # Official DOSM population by district, ANNUAL 2020-2025 — the only
    # time-varying demand source in the project.
    "dosm_population_district": RAW_DIR / "population_district.parquet",
}

# Raw sources that are DIRECTORIES of files (satellite tiles/granules)
# rather than a single file. Fingerprinted by file count + total bytes.
RAW_DATASET_DIRS = {
    "viirs_vnp46a3": RAW_DIR / "VNP46A3_2-20260826_125341",
    "esa_worldcover": RAW_DIR / "worldcover",
}

# Intermediate per-H3 extractions produced by the standalone raster
# scripts (scripts/extract/viirs.py, scripts/extract/worldcover.py, scripts/extract/merge_layers.py).
# run_pipeline merges `static_new_layers` into cell_static_features BEFORE
# the static QoE baseline is built, so VIIRS/WorldCover reach the
# modelling dataset, the hourly panel and the feature dictionary — not
# just the static features file, which is where they previously
# dead-ended.
DERIVED_LAYERS = {
    "static_new_layers": PROCESSED_DIR / "static_new_layers_by_h3.parquet",
    "sunway_by_h3": PROCESSED_DIR / "sunway_real_by_h3.parquet",
    # OSM building heights, extracted from the PBF already in data/raw/.
    # Sparse (6.5% of buildings tagged, 21% of cells covered) and
    # coverage-biased toward well-mapped areas — see scripts/extract/building_heights.py.
    "building_heights": PROCESSED_DIR / "building_heights_by_h3.parquet",
    # Copernicus DEM GLO-30 terrain statistics. Public S3, no account.
    "dem": PROCESSED_DIR / "dem_by_h3.parquet",
}

# Acquisition vintage per source, recorded in metadata.json. The hourly
# panel's timestamps are a synthetic calendar window chosen to overlap the
# ridership file's real coverage — they are NOT the dates the underlying
# measurements were taken, and these vintages are the honest answer to
# "how fresh is this data".
SOURCE_VINTAGES = {
    "ookla": "14 quarterly tile releases, 2023Q1-2026Q2. The single 2024Q4 snapshot is still used as the hourly panel's anchor level; the quarterly panel uses all 14",
    "buildings": "Google Open Buildings v3 (2023 release)",
    "towers": "OpenCellID MCC 502 export, downloaded 2026-07",
    "osm": "Geofabrik extract dated 2026-07-27",
    "worldpop": "WorldPop 2020 UN-adjusted (2020 reference year)",
    "dosm_population": "data.gov.my national series, latest row used",
    "ridership": "data.gov.my daily ridership, coverage through 2026-06-30",
    "mlab_diurnal_curve": "M-Lab NDT, aggregated over Dec 2024 onward (UTC-keyed; converted to Asia/Kuala_Lumpur at load)",
    "sunway_qoe": "Published third-party dataset, NOT self-collected — Kabeer, Nordin & Behjati (2025), 'An Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments', Mendeley Data v1, DOI 10.17632/dx5xyyfz2y.1, published 2025-06-17. Measurements span 2024-12-03 to 2025-04-02",
    "dosm_population_district": "DOSM district population, annual series 2020-2025 (published values, in thousands)",
    "viirs_vnp46a3": "43 monthly composites, 2022-12 to 2026-06",
    "esa_worldcover": "ESA WorldCover v200, 2021 reference year",
}

# --------------------------------------------------------------------------
# Demand Proxy Weights
# --------------------------------------------------------------------------
# demand_pressure_base uses only the first four (population/poi/built_area/
# transport) — all real static GIS inputs, zero QoE dependency, safe to use
# as a feature against ANY target, current or simulated.
#
# The full demand_pressure_index additionally folds in "historical_qoe"
# (lag_24h_rtt_median) and "nearby_demand" (spatial lag of itself). Because
# the simulated hourly panel's rtt_median is itself generated as a function
# of population/POI (see data_downloader.build_dynamic_panel), including
# historical_qoe creates a circular path back to the same real static inputs
# that already drive demand_pressure_base — i.e. part of any observed
# demand_pressure_index -> target correlation in this dataset is guaranteed
# by construction, not discovered. HIGH_LEAKAGE_RISK_FEATURES below flags
# this for the modelling team; prefer demand_pressure_base over the full
# demand_pressure_index until real (non-simulated) QoE data is available.
DEMAND_WEIGHTS = {
    "population": 0.30,
    "poi": 0.20,
    "built_area": 0.15,
    "transport": 0.15,
    "historical_qoe": 0.10,
    "nearby_demand": 0.10,
}

# Columns that are mechanically linked to the simulated QoE generator and
# should not be presented as independently-validated predictors while the
# panel remains simulated. Exported into feature_dictionary.xlsx and
# metadata.json so the modelling team sees this without reading source code.
HIGH_LEAKAGE_RISK_FEATURES = {
    "demand_pressure_index": (
        "Circular with the simulated QoE generator via its historical_qoe "
        "component (lag_24h_rtt_median). Use demand_pressure_base instead "
        "for a non-circular, real-data-only feature."
    ),
    "demand_pressure_index_pre_spatial": (
        "Same circularity as demand_pressure_index, pre-spatial-smoothing."
    ),
    # NEW — see data_validator.validate_target_anchor_correlation. On the
    # simulated hourly panel, cells with no real Ookla anchor generate their
    # hourly dl/rtt from a formula that (at current QOE_THRESHOLDS) may
    # never cross the degradation threshold, so almost all positive
    # target_is_degraded_next_1h labels come from anchored cells. These four
    # columns determine or are only non-null when qoe_level_is_ookla_anchored
    # is True, so a model with access to them can trivially learn "is this
    # cell anchored" instead of a genuine demand/QoE pattern. Hard-excluded
    # from run_pipeline.get_feature_columns()'s default feature list for
    # BOTH target modes (not just flagged) — see ANCHOR_STATUS_LEAKAGE_COLS.
    "qoe_level_is_ookla_anchored": (
        "Near-perfectly determines whether target_is_degraded_next_1h can be "
        "positive at all on the simulated panel — see run_full_validation's "
        "validate_target_anchor_correlation warning for the current split. "
        "Excluded from the default feature list, not just flagged."
    ),
    "qoe_baseline_source": (
        "Human-readable restatement of qoe_level_is_ookla_anchored — same "
        "exclusion reason."
    ),
    "ookla_quarter_test_count": (
        "Only non-null when qoe_level_is_ookla_anchored is True — a direct "
        "proxy for anchor status, same exclusion reason."
    ),
    "ookla_quarter_device_count": (
        "Only non-null when qoe_level_is_ookla_anchored is True — a direct "
        "proxy for anchor status, same exclusion reason."
    ),
    # NEW — on cell_static_qoe_baseline.parquet, these ARE the target
    # (README's "Option A" workflow trains on ookla_dl_mbps / classifies
    # qoe_risk_label) or a direct derivative/split-indicator of it.
    # qoe_risk_label in particular is literally thresholded from
    # ookla_dl_mbps — including it as a "feature" would let a model read
    # the answer off a relabelled copy of itself.
    "ookla_dl_mbps": "This IS the Option A modelling target — never a feature.",
    "ookla_rtt_ms": "This IS the Option A modelling target — never a feature.",
    "ookla_tests": "Real quarterly Ookla sample count — target-adjacent, never a feature.",
    "ookla_devices": "Real quarterly Ookla device count — target-adjacent, never a feature.",
    "has_real_ookla_label": (
        "Split indicator (labelled vs. prediction-target cells), not a "
        "predictor — training on it would leak which rows have a real label."
    ),
    "qoe_risk_label": (
        "Directly thresholded from ookla_dl_mbps (the target) — using it as "
        "a feature reads the answer off a relabelled copy of itself."
    ),
    "ookla_is_synthetic": "has_real_ookla_label inverted under a different name — same exclusion reason.",
}

# --------------------------------------------------------------------------
# QoE Thresholds for Classification
# --------------------------------------------------------------------------
# SINGLE source of truth for every degradation-related label in the project.
# target_is_degraded (binary) and target_degradation_class (3-class) are both
# derived from these same two tiers in src/data_preprocessing/qoe_target.py,
# so the two labels can no longer disagree with each other on the same row
# (previously: is_degraded used dl<10/rtt>100 while the class label used a
# separately hardcoded dl<15/rtt>80 — fixed).
# RELATIVE degradation cutoff, as a cross-cell percentile evaluated per
# timestamp. Complements (does not replace) the absolute QOE_THRESHOLDS
# below. The absolute 10 Mbps cutoff fires on only ~2.3% of labelled cells
# against a real Ookla median of 175 Mbps, so it cannot express "this
# neighbourhood is worse than the city" — the question the proposal is
# actually about. See qoe_target.compute_relative_targets.
# How is_peak_period is defined.
#   "measured"  — the hours the panel's own realised curve is worst. Derived
#                 from the real M-Lab diurnal shape, so it reflects when this
#                 city's network is actually busiest (19:00-00:00).
#   "commuter"  — the original hardcoded assumption (weekday 08-10, 17-20).
#                 Kept selectable so the old behaviour is reproducible, but
#                 the measurement does not support it: mean throughput during
#                 that window is no worse than the rest of the day.
# Both definitions are always exported (is_peak_period follows this setting;
# is_commuter_period always holds the assumption) so nothing is lost.
PEAK_PERIOD_DEFINITION = "measured"

RELATIVE_DEGRADATION_PERCENTILE = 25.0

QOE_THRESHOLDS = {
    "dl_degraded_mbps": 10.0,   # class >= 1 ("degraded") and is_degraded=1
    "rtt_degraded_ms": 100.0,
    "dl_severe_mbps": 5.0,      # class == 2 ("severely degraded")
    "rtt_severe_ms": 150.0,
    "dl_excellent_mbps": 50.0,
    "rtt_excellent_ms": 30.0,
}

# --------------------------------------------------------------------------
# QoE Index Weights
# --------------------------------------------------------------------------
# Positive (utility) weights MUST sum to 1.0, otherwise the composite index's
# theoretical ceiling silently falls below 100 (previous bug: 0.5 + 0.3 = 0.8
# meant target_qoe_index could never exceed 80, and the derived 1-5 quality
# score could never exceed 4.2, despite being documented as spanning the
# full range). Verified against src/data_preprocessing/qoe_target.py.
QOE_INDEX_WEIGHTS = {
    "download": 0.6,
    "upload": 0.4,
    "rtt_penalty": 0.4,
    "loss_penalty": 0.4,
}

# --------------------------------------------------------------------------
# Modelling target
# --------------------------------------------------------------------------
# The dataset's default target used to be same-hour degradation, i.e. "is
# this cell degraded right now" using features measured in that same hour —
# not a forecast. The recommended primary MVP target is now the 1-hour-ahead
# shifted label, computed by qoe_target.compute_next_hour_targets().
# CORRECTED in 3.3.0. This constant used to name the hourly target, which
# PANEL_STRUCTURE_DISCLOSURE (below) then explains is analytically
# reconstructable from (cell, hour-of-week) — so the file recommended a
# target it also documents as memorisable. It flows into metadata.json's
# `primary_ml_target`, so anyone reading the metadata rather than HANDOFF.md
# was pointed at the wrong label.
#
# The primary target is the QUARTERLY relative one, on cell_quarter_dataset:
# a real observed Ookla measurement one quarter ahead, with no reconstructable
# structure. Prevalence ~22-25% by construction (25th percentile of the
# citywide distribution each quarter), which is also far better balanced than
# the absolute is_degraded flag (3.0% and falling as Malaysian speeds rise).
PRIMARY_ML_TARGET = "target_is_below_area_norm_next_q"
PRIMARY_ML_TABLE = "cell_quarter_dataset.parquet"

# The hourly panel's target. Kept because the panel still ships and the
# dashboard animates it, but it is a DEMONSTRATION label, not a benchmark:
# see PANEL_STRUCTURE_DISCLOSURE. Never report forecast skill on it.
PANEL_DEMO_TARGET = "target_is_degraded_next_1h"
FORECAST_HORIZON_HOURS = 1

# --------------------------------------------------------------------------
# Data honesty / simulation disclosure
# --------------------------------------------------------------------------
# No organiser QoE dataset and no per-hour QoE source of any kind exists for
# this project. Ookla provides one real, STATIC, quarterly average per cell
# for the ~41% of cells it covers — nothing hourly. Every hourly QoE value in
# cell_time_dataset.parquet (dl/ul/rtt/jitter/loss, whether or not a real
# Ookla anchor exists for that cell) is generated by a diurnal-shape formula,
# not observed. This is tracked explicitly (see data_downloader.py and
# metadata_utils.py) rather than left implicit in a single ambiguous flag.
# Per-component honesty, replacing the single ambiguous "the panel is
# simulated" boolean. Each hourly column is one of three things and the
# metadata now says which, per component, rather than collapsing them.
PANEL_COMPONENT_PROVENANCE = {
    "hourly_shape_dl_rtt": (
        "REAL — M-Lab NDT citywide diurnal curve (2.05M tests), keyed in UTC "
        "in the source CSV and converted to Asia/Kuala_Lumpur at load. ONE "
        "curve applied uniformly to every cell: a real shape, not per-cell "
        "resolution. Traffic mix not ASN-filtered, so it may include fixed "
        "as well as mobile tests."
    ),
    "hourly_level_dl_rtt_anchored_cells": (
        "REAL — the cell's own Ookla 2024 Q4 quarterly average, used as the "
        "level the real hourly shape is applied to."
    ),
    "hourly_level_dl_rtt_unanchored_cells": (
        "FORMULA-DERIVED — no Ookla tile coverage exists for these cells, so "
        "the level comes from a population/POI load function plus seeded "
        "Gaussian noise. Flagged per row by qoe_level_is_ookla_anchored."
    ),
    "hourly_jitter_loss_ul": (
        "SIMULATED — no free real source exists at this resolution. Sparse "
        "real ping/jitter/loss values (0.2% coverage) exist only inside the "
        "Sunway inset and are kept separately, never merged into these."
    ),
}

# The structural property that actually bounds what can be claimed from
# the panel — recorded explicitly because it is not implied by "the shape
# is real". For an Ookla-anchored cell the hourly value is exactly
# level(cell) x curve(hour) with no noise term, so the panel factorises
# (rank one) and the degradation target is analytically reconstructable
# from two static columns plus the hour. Verified empirically: the
# reconstruction matches the stored target on 100% of anchored rows.
PANEL_STRUCTURE_DISCLOSURE = (
    "For Ookla-anchored cells the hourly panel is exactly "
    "ookla_static(cell) x mlab_curve(hour-of-week), deterministic and "
    "rank-one: every anchored cell shares ONE identical temporal curve, "
    "differing only in level. Consequences the modelling team must know: "
    "(1) target_is_degraded_next_1h is fully determined by (cell, "
    "hour-of-week), so a model given cell identity and time recovers it by "
    "memorisation — reported forecast skill on this panel is not evidence "
    "of real predictive ability; (2) there is no cell-specific temporal "
    "dynamics, so no model can learn that one area degrades earlier than "
    "another, because that information is not present; (3) Getis-Ord Gi* is "
    "invariant to a global scale factor, so hotspot recurrence varies only "
    "through the Gaussian noise injected into unanchored cells — 'this area "
    "degraded in N of the last 7 evenings' is not currently supportable. "
    "Fixing this requires a spatially-varying real diurnal curve (per-"
    "district M-Lab), not a modelling change."
)

SIMULATION_DISCLAIMER = (
    "cell_time_dataset.parquet is a hierarchical panel, not a measurement: "
    "a REAL citywide hourly shape (M-Lab NDT) applied to a REAL per-cell "
    "level (Ookla quarterly) where one exists, and to a formula-derived "
    "level where it does not. Jitter, loss and upload remain simulated. "
    "No per-cell per-hour QoE measurement source exists for this project. "
    "Because one shape is shared by every cell, the anchored panel is "
    "rank-one and its forecast target is analytically reconstructable — see "
    "PANEL_STRUCTURE_DISCLOSURE before reporting any forecast metric. For a "
    "modelling claim that stands up to review, use "
    "cell_static_qoe_baseline.parquet (real Ookla targets, real GIS "
    "features, zero synthetic QoE)."
)

# --------------------------------------------------------------------------
# Pipeline / Metadata
# --------------------------------------------------------------------------
PIPELINE_VERSION = "3.3.0"  # 3.0.0: target-consistency, leakage, and boundary-
                             #   scope fixes.
                             # 3.1.0: raw-file extraction bounds fix
                             #   (extraction_bounds() derives from the real
                             #   grid) — README already documented this as
                             #   "v3.1.0" before this constant was bumped to
                             #   match; fixed here too.
                             # 3.2.0 (this pass): Kuala Lumpur merged into
                             #   the study boundary (4,659 -> 4,918 cells),
                             #   h3_parent_res6 spatial-holdout grouping key
                             #   added to every output table, and an
                             #   automated anchor-status/target-leakage audit
                             #   added to data_validator.py.
                             # See CHANGELOG.md for full detail.
                             # 3.3.0 (Phase 0 correctness pass):
                             #   - M-Lab curve converted UTC -> Asia/
                             #     Kuala_Lumpur at load (was applied 8h out
                             #     of phase); hard regression guard added.
                             #   - Sunway TESTDOWNLINK renamed *_dl_raw and
                             #     the Ookla-vs-Sunway correlation withdrawn
                             #     (unit unverified, n=8).
                             #   - VIIRS/WorldCover/Sunway merged BEFORE the
                             #     static QoE baseline is built, so all 24
                             #     real columns reach every output.
                             #   - All 11 WorldCover classes retained; tile
                             #     counts accumulated across the lat-3.0 seam.
                             #   - metadata.json registers all 11 sources,
                             #     records source vintages, and discloses the
                             #     panel's rank-one structure. Feature-
                             #     dictionary completeness is now asserted.
