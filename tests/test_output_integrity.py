"""
Independent verification of the shipped outputs.

WHY THIS EXISTS SEPARATELY FROM data_validator.py
--------------------------------------------------
The pipeline's own validators check internal consistency: do the columns
exist, do the row counts line up, is anything null that should not be. That
catches a broken build. It cannot catch a build that is wrong but
self-consistent -- a CRS mix-up that shifts every cell by the same offset, a
join that attaches the right-shaped data to the wrong hexagons, a target that
is secretly a copy of a feature. Every one of those passes an internal check
and fails reality.

So every test here asks one of two questions instead:

  1. Does this agree with something externally known about Klang Valley?
     (KLCC is bright and dense; Hulu Langat is forest; land cover sums to 100%)
  2. Do two sources that never coordinated agree with each other?
     (NASA VIIRS vs WorldPop vs ESA WorldCover vs OSM)

Independent agreement is the strongest evidence available without ground
truth: four organisations cannot make the same spatial-join error.

Run:  python -m pytest tests/test_output_integrity.py -v
      python tests/test_output_integrity.py        (also works directly)
"""
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "data" / "outputs"

pytestmark = pytest.mark.skipif(
    not (OUT / "cell_static_qoe_baseline.parquet").exists(),
    reason="outputs not built -- run `python run_pipeline.py` first",
)

# Landmarks with independently-known character. Chosen so a grid
# misalignment of even one hexagon breaks at least one expectation: KLIA in
# particular is heavily built-up but nearly unpopulated, a combination no
# generic "city centre" cell reproduces.
LANDMARKS = {
    "klcc":   (3.1578, 101.7117),
    "sunway": (3.0733, 101.6067),
    "klia":   (2.7456, 101.7099),
    "forest": (3.2200, 101.8700),
}


@pytest.fixture(scope="module")
def static_df():
    return pd.read_parquet(OUT / "cell_static_qoe_baseline.parquet")


@pytest.fixture(scope="module")
def quarterly_df():
    return pd.read_parquet(OUT / "cell_quarter_dataset.parquet")


def _at(df, name, col):
    cell = h3.latlng_to_cell(*LANDMARKS[name], 8)
    row = df[df.h3_index == cell]
    assert not row.empty, f"{name} is not in the grid -- study area or resolution changed"
    return row.iloc[0][col]


# --------------------------------------------------------------------------
# Spatial geometry
# --------------------------------------------------------------------------
def test_h3_resolution_is_uniform(static_df):
    assert {h3.get_resolution(c) for c in static_df.h3_index} == {8}


def test_cells_are_unique(static_df):
    assert static_df.h3_index.is_unique


def test_all_cells_inside_study_area(static_df):
    ll = np.array([h3.cell_to_latlng(c) for c in static_df.h3_index])
    lat, lon = ll[:, 0], ll[:, 1]
    assert lat.min() > 2.6 and lat.max() < 3.6
    assert lon.min() > 101.0 and lon.max() < 102.1


def test_parent_index_matches_independent_recomputation(static_df):
    """Guards the spatial-holdout key. If this drifts, CV folds leak."""
    expected = [h3.cell_to_parent(c, 6) for c in static_df.h3_index]
    assert (pd.Series(expected, index=static_df.index) == static_df.h3_parent_res6).all()


# --------------------------------------------------------------------------
# Ground truth -- does the data match the real Klang Valley?
# --------------------------------------------------------------------------
def test_city_centre_is_brighter_than_forest(static_df):
    assert _at(static_df, "klcc", "viirs_avg_rad_mean") > \
           _at(static_df, "forest", "viirs_avg_rad_mean") * 3


def test_land_cover_matches_known_character(static_df):
    assert _at(static_df, "klcc", "landcover_built_up_pct") > 70
    assert _at(static_df, "forest", "landcover_tree_cover_pct") > 70


def test_airport_is_built_up_but_unpopulated(static_df):
    """The sharpest grid-alignment test: KLIA is a large structure where
    almost nobody sleeps. A cell offset toward Sepang town would populate it."""
    assert _at(static_df, "klia", "landcover_built_up_pct") > 70
    assert _at(static_df, "klia", "population_density_per_km2") < 2000


# --------------------------------------------------------------------------
# Independent sources agree -- the spatial joins place data correctly
# --------------------------------------------------------------------------
@pytest.mark.parametrize("a,b,floor", [
    ("population_density_per_km2", "viirs_avg_rad_mean", 0.7),       # WorldPop vs NASA
    ("landcover_built_up_pct", "built_up_ratio", 0.7),               # ESA vs OSM
    ("population_density_per_km2", "road_density_km_per_km2", 0.6),  # WorldPop vs OSM
    ("viirs_avg_rad_mean", "poi_retail_food_count", 0.4),            # NASA vs OSM
])
def test_independent_sources_agree(static_df, a, b, floor):
    rho = static_df[[a, b]].dropna().corr(method="spearman").iloc[0, 1]
    assert rho > floor, f"{a} vs {b}: rho={rho:.3f} < {floor}"


def test_tree_cover_opposes_built_up(static_df):
    rho = static_df[["landcover_tree_cover_pct", "built_up_ratio"]].corr(
        method="spearman").iloc[0, 1]
    assert rho < -0.3


# --------------------------------------------------------------------------
# Physical plausibility
# --------------------------------------------------------------------------
def test_land_cover_fractions_sum_to_100(static_df):
    cols = [c for c in static_df.columns
            if c.startswith("landcover_") and c.endswith("_pct")]
    assert ((static_df[cols].sum(axis=1) - 100).abs() < 1.0).all()


def test_no_phantom_zero_elevation(static_df):
    """Regression test. Copernicus tiles declare nodata=None, so an
    out-of-bounds read returns 0 AND counts, which once gave 2,562 cells an
    elevation of exactly 0.0 m. Tiles are now selected by centroid bounds."""
    assert int((static_df.elevation_mean_m == 0).sum()) < 50


def test_elevation_range_is_realistic(static_df):
    """Slightly negative is legitimate: Copernicus is geoid-referenced and the
    study area reaches the Kuala Langat coast. Gunung Nuang caps the top."""
    assert static_df.elevation_mean_m.min() > -30
    assert 1000 < static_df.elevation_mean_m.max() < 1600
    assert 10 < static_df.elevation_mean_m.median() < 100


def test_download_exceeds_upload(quarterly_df):
    d = quarterly_df.dropna(subset=["ookla_dl_mbps", "ookla_ul_mbps"])
    assert (d.ookla_dl_mbps > d.ookla_ul_mbps).mean() > 0.9


def test_latency_is_plausible(quarterly_df):
    """p99 rather than max: single-test cell-quarters carry real Ookla
    artifacts (one row has rtt=0 from one test), which is upstream noise, not
    a pipeline defect. The bulk distribution is what must hold."""
    r = quarterly_df.ookla_rtt_ms.dropna()
    assert 5 < r.median() < 60
    assert r.quantile(0.99) < 500


# --------------------------------------------------------------------------
# Leakage -- the failure that produces a great score and a worthless model
# --------------------------------------------------------------------------
def test_no_feature_is_a_copy_of_the_target(quarterly_df):
    d = quarterly_df[quarterly_df.target_next_q_valid == 1].dropna(
        subset=["target_is_below_area_norm_next_q"])
    y = d.target_is_below_area_norm_next_q.astype(int)
    num = d.select_dtypes(include=[np.number]).drop(
        columns=[c for c in d.columns if c.startswith("target_")], errors="ignore")
    # Constant columns (e.g. landcover_snow_ice_pct, always 0 in the tropics)
    # have zero variance, so a correlation is undefined rather than suspicious.
    num = num.loc[:, num.std(numeric_only=True) > 0]
    worst = num.corrwith(y).abs().max()
    assert worst < 0.95, f"a feature correlates {worst:.3f} with the target"


def test_lag_columns_do_not_contain_the_present(quarterly_df):
    d = quarterly_df[quarterly_df.lag_1q_dl_valid == 1]
    assert (d.lag_1q_dl == d.ookla_dl_mbps).mean() < 0.02


# --------------------------------------------------------------------------
# Is the dataset actually useful?
# --------------------------------------------------------------------------
def test_primary_target_beats_persistence_on_a_future_split():
    """The honest benchmark is not chance -- it is "assume next quarter looks
    like this one". A dataset that cannot beat persistence adds nothing."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score

    q = pd.read_parquet(OUT / "cell_quarter_dataset.parquet")
    tgt = "target_is_below_area_norm_next_q"
    d = q[q.target_next_q_valid == 1].dropna(subset=[tgt]).sort_values("quarter_index")
    cut = d.quarter_index.quantile(0.7)
    tr, te = d[d.quarter_index <= cut], d[d.quarter_index > cut]

    drop = [c for c in d.columns if c.startswith("target_")] + [
        "quarter_index", "year", "quarter", "is_degraded",
        "dl_percentile_in_quarter", "is_below_area_norm",
        "ookla_dl_mbps", "ookla_ul_mbps", "ookla_rtt_ms"]
    feats = d.select_dtypes(include=[np.number]).drop(
        columns=drop, errors="ignore").columns.tolist()

    model = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    model.fit(tr[feats], tr[tgt])
    p = model.predict_proba(te[feats])[:, 1]
    ap = average_precision_score(te[tgt], p)
    persistence = average_precision_score(te[tgt], te.is_below_area_norm)
    assert ap > persistence, f"model PR-AUC {ap:.3f} does not beat persistence {persistence:.3f}"


def test_primary_target_is_the_quarterly_one():
    """The hourly panel is rank-one and its target is reconstructable from
    (cell, hour-of-week). config once named it PRIMARY_ML_TARGET while also
    documenting it as memorisable, so anyone reading config rather than
    HANDOFF.md would have benchmarked on a label a lookup table can solve."""
    import config
    assert config.PRIMARY_ML_TARGET == "target_is_below_area_norm_next_q"
    assert config.PANEL_DEMO_TARGET == "target_is_degraded_next_1h"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
