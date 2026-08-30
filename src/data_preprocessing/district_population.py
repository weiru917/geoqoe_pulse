"""
DOSM district population — the project's first TIME-VARYING demand feature.

WHY THIS MATTERS
----------------
Until now every demand proxy in the project was static: population, POIs,
built-up ratio and nighttime lights were all constant per cell across all
14 quarters. Zero of 3,566 cells had a demand feature that changed over
time. That made the proposal's second question — "is degradation
associated with INCREASING local demand?" — unanswerable, because demand
never increased in the data.

data/raw/population_district.parquet (data.gov.my / DOSM) is an ANNUAL
series, 2020-2025, by district. All eight study districts are present and
they genuinely diverge: Klang +8.6% and Ulu Langat +8.3% over the window
against Petaling's +3.2%. Joined to the quarterly Ookla panel by year,
that is a real demand signal that varies in BOTH space and time.

RESOLUTION — state this plainly wherever the feature is used
------------------------------------------------------------
District, not cell. Eight districts cover 4,918 cells, so every cell in
Petaling shares one population trajectory. This is a genuine improvement
over "constant forever" and a genuine step below per-cell. It supports
"degradation concentrates in districts whose population is growing"; it
does NOT support "this hexagon degraded because its own population grew".

CAUSALITY
---------
DOSM dates each estimate to 1 January of its year, so the value for year
Y is known during year Y. Joining quarter (Y, Q) to year Y's population
therefore uses information available before the quarter being predicted —
no leakage into the next-quarter target.

COVERAGE EDGE
-------------
The Ookla panel runs to 2026Q2; DOSM ends at 2025. Quarters beyond the
series carry the last observed year forward and are flagged by
`district_population_is_carried_forward`. Carried-forward rows are real
values applied to a later period, not estimates — filter them out if you
need strictly contemporaneous population.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from config import CRS_WGS84, RAW_DIR

logger = logging.getLogger(__name__)

DOSM_DISTRICT_PATH = RAW_DIR / "population_district.parquet"

# Boundary NAME_2 -> DOSM `district`. GADM writes Selangor districts without
# spaces and uses "Hulu"; DOSM uses "Ulu" and prefixes the federal
# territories with "W.P.". Mapped explicitly rather than fuzzy-matched so a
# rename in either source fails loudly instead of silently dropping a
# district.
BOUNDARY_TO_DOSM_DISTRICT = {
    "Gombak": "Gombak",
    "HuluLangat": "Ulu Langat",
    "Klang": "Klang",
    "KualaLangat": "Kuala Langat",
    "Petaling": "Petaling",
    "Sepang": "Sepang",
    "Putrajaya": "W.P. Putrajaya",
    "Kuala Lumpur": "W.P. Kuala Lumpur",
}

BASELINE_YEAR = 2020


def assign_cells_to_districts(grid_gdf: gpd.GeoDataFrame,
                               boundary_path=None) -> pd.DataFrame:
    """
    Point-in-polygon assign each H3 cell centroid to its district.

    Returns h3_index, district_name (the boundary's NAME_2) and
    dosm_district (the matching DOSM key). Cells that fall in no polygon —
    possible for cells clipped at the study edge — get NaN rather than a
    nearest-district guess.
    """
    from config import STUDY_BOUNDARY_PATH

    boundary = gpd.read_file(boundary_path or STUDY_BOUNDARY_PATH)
    if boundary.crs is None:
        boundary = boundary.set_crs(CRS_WGS84)
    else:
        boundary = boundary.to_crs(CRS_WGS84)

    centroids = gpd.GeoDataFrame(
        grid_gdf[["h3_index"]].copy(),
        geometry=gpd.points_from_xy(grid_gdf["centroid_lon"], grid_gdf["centroid_lat"]),
        crs=CRS_WGS84,
    )
    joined = gpd.sjoin(
        centroids, boundary[["NAME_2", "geometry"]], how="left", predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()

    out = pd.DataFrame({
        "h3_index": grid_gdf["h3_index"].values,
        "district_name": joined["NAME_2"].values,
    })
    out["dosm_district"] = out["district_name"].map(BOUNDARY_TO_DOSM_DISTRICT)

    n_missing = int(out["district_name"].isna().sum())
    unmapped = sorted(set(out.loc[out["dosm_district"].isna() & out["district_name"].notna(),
                                   "district_name"]))
    if n_missing:
        logger.warning(
            "%d cell(s) fall outside every district polygon and have no district assigned "
            "(NaN, not guessed). These are edge cells where the H3 centroid sits just outside "
            "the boundary.", n_missing,
        )
    if unmapped:
        raise ValueError(
            f"Boundary district(s) {unmapped} have no entry in BOUNDARY_TO_DOSM_DISTRICT. "
            "Add the mapping rather than letting the district drop out silently."
        )

    logger.info(
        "District assignment: %d cells across %d districts (%s).",
        int(out["district_name"].notna().sum()), out["district_name"].nunique(),
        ", ".join(sorted(out["district_name"].dropna().unique())),
    )
    return out


def load_district_population(path=DOSM_DISTRICT_PATH) -> pd.DataFrame:
    """
    Annual overall population per district, in PERSONS.

    DOSM publishes the file broken out by sex/age/ethnicity with population
    in THOUSANDS. Only the fully-aggregated rows (both / overall / overall)
    are taken — summing the breakdowns would double-count, which is the
    same trap the national file already caused once in this project.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the district-level population table "
            "from data.gov.my / OpenDOSM and place it there."
        )

    df = pd.read_parquet(path)
    required = {"state", "district", "date", "sex", "age", "ethnicity", "population"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"District population file is missing columns: {missing}")

    overall = df[
        (df["sex"] == "both") & (df["age"] == "overall") & (df["ethnicity"] == "overall")
    ].copy()
    overall["year"] = pd.to_datetime(overall["date"]).dt.year
    # DOSM reports thousands.
    overall["district_population"] = overall["population"] * 1000.0

    out = overall[["district", "year", "district_population"]].rename(
        columns={"district": "dosm_district"}
    )
    dupes = out.duplicated(["dosm_district", "year"]).sum()
    if dupes:
        raise ValueError(
            f"{dupes} duplicate (district, year) rows after filtering to overall aggregates — "
            "the filter is not isolating a single row per district-year."
        )

    logger.info(
        "District population: %d districts x %d years (%d-%d), overall aggregates only.",
        out["dosm_district"].nunique(), out["year"].nunique(),
        int(out["year"].min()), int(out["year"].max()),
    )
    return out


def build_district_population_features(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (district, year): level, year-on-year growth, and growth since the
    baseline year. Growth is NaN for the first year rather than 0 — there
    is no prior year to compare against, and 0 would read as "flat".
    """
    df = pop_df.sort_values(["dosm_district", "year"]).copy()
    g = df.groupby("dosm_district", sort=False)

    df["district_population_yoy_growth_pct"] = g["district_population"].pct_change() * 100.0

    baseline = (
        df[df["year"] == BASELINE_YEAR]
        .set_index("dosm_district")["district_population"]
    )
    df["district_population_growth_since_2020_pct"] = (
        df["district_population"] / df["dosm_district"].map(baseline) - 1.0
    ) * 100.0
    return df


def attach_to_quarterly_panel(quarterly_df: pd.DataFrame,
                               cell_districts: pd.DataFrame,
                               pop_features: pd.DataFrame) -> pd.DataFrame:
    """
    Join district population onto the quarterly panel by (district, year of
    quarter). Quarters past the end of the DOSM series carry the last
    observed year forward, flagged explicitly.
    """
    df = quarterly_df.copy()
    # The district columns may already be present: run_pipeline merges
    # cell_districts into static_df, and build_quarterly_ml_dataset then
    # merges static_df into the quarterly frame. Merging them again here
    # would produce _x/_y suffixes and break the join key. Only attach them
    # if they are genuinely absent, so this function works both inside the
    # pipeline and standalone.
    missing = [c for c in ("district_name", "dosm_district") if c not in df.columns]
    if missing:
        df = df.merge(cell_districts[["h3_index"] + missing], on="h3_index", how="left")

    if "dosm_district" not in df.columns:
        raise KeyError(
            "dosm_district is absent after the district merge — cell_districts must "
            "provide it, or the quarterly frame must already carry it."
        )

    max_year = int(pop_features["year"].max())
    df["_pop_year"] = df["year"].clip(upper=max_year)
    df["district_population_is_carried_forward"] = (df["year"] > max_year).astype(int)

    df = df.merge(
        pop_features.rename(columns={"year": "_pop_year"}),
        on=["dosm_district", "_pop_year"], how="left",
    ).drop(columns=["_pop_year"])

    n_carried = int(df["district_population_is_carried_forward"].sum())
    n_null = int(df["district_population"].isna().sum())
    logger.info(
        "District population joined to the quarterly panel: %d/%d rows have a value; "
        "%d rows (%.1f%%) carry the %d value forward because the Ookla panel runs past the "
        "end of the DOSM series. This is the project's FIRST demand feature that varies over "
        "time — district resolution, not per-cell.",
        len(df) - n_null, len(df), n_carried, 100 * n_carried / max(len(df), 1), max_year,
    )
    return df
