"""
GeoQoE Pulse Real Open Data Ingestion & Preprocessing Module.

Performs real GeoPandas spatial joins of all 7 raw datasets into the H3
study grid:
1. 2024-10-01_performance_mobile_tiles.parquet  (Ookla Open Data Mobile)
2. 31d_buildings.csv.gz                          (Google Open Buildings V3)
3. 502.csv.gz                                    (OpenCellID MCC 502)
4. malaysia-singapore-brunei-260727.osm.pbf       (Geofabrik OSM extract)
5. mys_ppp_2020_UNadj.tif                         (WorldPop 2020 raster)
6. population_malaysia.csv                        (OpenDOSM census)
7. ridership_headline.csv                         (data.gov.my transit ridership)
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt

from config import CRS_WGS84, CRS_PENINSULAR_MYS, RAW_DATASETS, STUDY_BBOX
from src.data_preprocessing.geometry_utils import h3_ids_to_geodataframe

logger = logging.getLogger(__name__)


def extraction_bounds(grid_gdf: gpd.GeoDataFrame, buffer_deg: float = 0.05) -> tuple:
    """
    Real raw-file pre-filter bounds, derived from the ACTUAL study grid
    (grid_gdf.total_bounds) rather than the stale, much smaller hardcoded
    config.STUDY_BBOX.

    Bug fixed here: load_towers/load_buildings/load_osm previously filtered
    raw records against config.STUDY_BBOX, which at the time was set to
    lon 101.50-101.80, lat 3.00-3.30 (a 0.3 deg x 0.3 deg box) while the
    real, trimmed study boundary spans lon 101.20-101.97, lat 2.59-3.40 —
    nearly 3x wider/taller. Every raw record outside that small box was
    silently dropped before the spatial join ever ran, so most H3 cells
    outside the small box received ZERO real building/tower/OSM records
    while still being flagged `<feature>_is_synthetic = False` — "zero"
    meant "outside the extraction window," not "confirmed no
    infrastructure." STUDY_BBOX has also since been widened to match the
    real boundary's extent (it now only serves as an emergency
    grid-generation fallback if the boundary file itself is missing — see
    geometry_utils.load_study_boundary), but extraction no longer reads it
    at all; every loader below derives its own bounds from grid_gdf.

    buffer_deg=0.05 (~5.5km) gives raw points just outside the grid's exact
    polygon edge (e.g. a tower a few hundred metres past the boundary that
    should still count via sjoin_nearest / distance calculations) room to
    be read in, without re-introducing a mismatch with the real boundary.
    """
    min_lon, min_lat, max_lon, max_lat = grid_gdf.total_bounds
    return (min_lon - buffer_deg, min_lat - buffer_deg, max_lon + buffer_deg, max_lat + buffer_deg)


# --------------------------------------------------------------------------
# Generic, reusable spatial-join helpers (GeoPandas, no row-wise .apply)
# --------------------------------------------------------------------------

def spatial_join_point_agg(points_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame,
                            value_cols=None, agg="sum") -> pd.DataFrame:
    """
    Vectorized spatial join of a point GeoDataFrame into H3 cell polygons,
    aggregating by cell. Replaces the old per-row `.apply(h3.latlng_to_cell)`
    pattern with a single `gpd.sjoin` (uses GeoPandas' spatial index, so it
    scales to large point sets without a Python-level loop).

    value_cols: optional list of numeric columns to sum/mean per cell in
    addition to the point count.
    """
    if points_gdf.crs is None:
        points_gdf = points_gdf.set_crs(CRS_WGS84)
    elif points_gdf.crs != grid_gdf.crs:
        points_gdf = points_gdf.to_crs(grid_gdf.crs)

    joined = gpd.sjoin(points_gdf, grid_gdf[["h3_index", "geometry"]], how="inner", predicate="within")
    if joined.empty:
        return pd.DataFrame(columns=["h3_index", "point_count"])

    grouped = joined.groupby("h3_index").size().rename("point_count").reset_index()

    if value_cols:
        value_agg = joined.groupby("h3_index")[value_cols].agg(agg).reset_index()
        grouped = grouped.merge(value_agg, on="h3_index", how="left")

    return grouped


def spatial_join_line_length(lines_gdf: gpd.GeoDataFrame, grid_gdf: gpd.GeoDataFrame,
                              projected_crs: str = CRS_PENINSULAR_MYS) -> pd.DataFrame:
    """
    Real geometric road/line length per H3 cell via polygon overlay
    (intersection), measured in a projected CRS (metres) — not a heuristic
    or a random draw. Returns h3_index, length_m.
    """
    if lines_gdf.empty:
        return pd.DataFrame(columns=["h3_index", "length_m"])

    if lines_gdf.crs is None:
        lines_gdf = lines_gdf.set_crs(CRS_WGS84)

    lines_proj = lines_gdf.to_crs(projected_crs)
    grid_proj = grid_gdf[["h3_index", "geometry"]].to_crs(projected_crs)

    clipped = gpd.overlay(lines_proj, grid_proj, how="intersection")
    if clipped.empty:
        return pd.DataFrame(columns=["h3_index", "length_m"])

    clipped["length_m"] = clipped.geometry.length
    return clipped.groupby("h3_index")["length_m"].sum().reset_index()


def nearest_distance_m(from_points_gdf: gpd.GeoDataFrame, to_points_gdf: gpd.GeoDataFrame,
                        projected_crs: str = CRS_PENINSULAR_MYS) -> np.ndarray:
    """
    True geometric nearest-neighbor distance (metres) from each row in
    `from_points_gdf` to the closest row in `to_points_gdf`, via
    `gpd.sjoin_nearest` in a projected CRS. Replaces the old formula-based
    "distance" heuristic entirely.
    """
    if to_points_gdf.empty:
        return np.full(len(from_points_gdf), np.nan)

    a = from_points_gdf.to_crs(projected_crs).reset_index(drop=True)
    b = to_points_gdf.to_crs(projected_crs).reset_index(drop=True)
    joined = gpd.sjoin_nearest(a, b[["geometry"]], how="left", distance_col="dist_m")
    joined = joined[~joined.index.duplicated(keep="first")]  # ties can duplicate left rows
    joined = joined.sort_index()
    return joined["dist_m"].reindex(range(len(a))).values


def _synthetic_seeded_fallback(n: int, low: float, high: float) -> np.ndarray:
    """Documented, seeded synthetic fallback (see config.GLOBAL_RANDOM_SEED,
    which must already be set by the caller/pipeline entrypoint)."""
    return np.random.uniform(low, high, n)


# --------------------------------------------------------------------------
# 1. OpenCellID Towers
# --------------------------------------------------------------------------

def load_towers(grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Real per-cell tower count, tower density (per km^2), and true geometric
    nearest-tower distance (metres, via sjoin_nearest — not a formula).
    """
    path = RAW_DATASETS["towers"]
    n = len(grid_gdf)

    if not path.exists():
        logger.warning("Towers file missing (%s) — using synthetic fallback.", path.name)
        area_km2 = grid_gdf["area_m2"] / 1e6
        count = np.random.poisson(1.5, n)
        return pd.DataFrame({
            "h3_index": grid_gdf["h3_index"],
            "tower_count": count,
            "opencellid_record_count": count,
            "tower_density_per_km2": count / area_km2.values,
            "dist_to_nearest_tower_m": _synthetic_seeded_fallback(n, 80, 1200),
            "tower_is_synthetic": True,
            "tower_source": "synthetic_fallback_no_file",
        })

    # OpenCellID schema (no header): radio,mcc,net,area,cell,unit,lon,lat,range,samples,...
    min_lon, min_lat, max_lon, max_lat = extraction_bounds(grid_gdf)
    raw = pd.read_csv(path, header=None, usecols=[6, 7], names=["lon", "lat"])
    raw = raw[raw["lat"].between(min_lat, max_lat) & raw["lon"].between(min_lon, max_lon)]
    n_records = len(raw)

    # OpenCellID has one row per (radio, operator, cell/sector) record — the
    # same physical mast commonly appears many times over (multiple sectors,
    # multiple technologies, multiple operators sharing a site). Counting raw
    # records as "towers" overcounts physical infrastructure. Approximate
    # physical sites by rounding coordinates to ~11m (5 decimal places) and
    # deduplicating: records from the same mast normally share the mast's
    # GPS fix to survey precision, so this collapses co-located sector/tech
    # records onto one site while leaving genuinely separate masts distinct.
    raw["_site_key"] = raw["lon"].round(5).astype(str) + "_" + raw["lat"].round(5).astype(str)
    site_points = raw.drop_duplicates(subset="_site_key")[["lon", "lat"]].reset_index(drop=True)
    n_sites = len(site_points)

    record_points = gpd.GeoDataFrame(raw, geometry=gpd.points_from_xy(raw["lon"], raw["lat"]), crs=CRS_WGS84)
    site_gdf = gpd.GeoDataFrame(site_points, geometry=gpd.points_from_xy(site_points["lon"], site_points["lat"]), crs=CRS_WGS84)

    record_counts = spatial_join_point_agg(record_points, grid_gdf).rename(columns={"point_count": "opencellid_record_count"})
    site_counts = spatial_join_point_agg(site_gdf, grid_gdf).rename(columns={"point_count": "tower_count"})

    result = grid_gdf[["h3_index", "area_m2", "geometry", "centroid_lat", "centroid_lon"]].merge(
        site_counts, on="h3_index", how="left"
    ).merge(record_counts, on="h3_index", how="left")
    result["tower_count"] = result["tower_count"].fillna(0).astype(int)
    result["opencellid_record_count"] = result["opencellid_record_count"].fillna(0).astype(int)
    result["tower_density_per_km2"] = result["tower_count"] / (result["area_m2"] / 1e6)

    cell_centroids = gpd.GeoDataFrame(
        result[["h3_index"]],
        geometry=gpd.points_from_xy(result["centroid_lon"], result["centroid_lat"]),
        crs=CRS_WGS84,
    )
    # Nearest distance uses deduplicated sites, not raw records — otherwise a
    # single mast with many co-located records could distort local minima.
    result["dist_to_nearest_tower_m"] = nearest_distance_m(cell_centroids, site_gdf)
    result["tower_is_synthetic"] = False
    result["tower_source"] = "opencellid_deduplicated"

    logger.info(
        "Towers: %d raw OpenCellID records collapsed into %d approximate physical sites "
        "(dedup ratio %.2f) and spatially joined into %d cells.",
        n_records, n_sites, n_sites / max(n_records, 1), n,
    )
    return result[[
        "h3_index", "tower_count", "opencellid_record_count", "tower_density_per_km2",
        "dist_to_nearest_tower_m", "tower_is_synthetic", "tower_source",
    ]]


# --------------------------------------------------------------------------
# 2. Google Open Buildings V3
# --------------------------------------------------------------------------

def load_buildings(grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Real per-cell building count, total footprint area, built-up ratio
    (against the cell's REAL area from grid geometry, not a hardcoded
    constant), building density, and average building footprint.
    """
    path = RAW_DATASETS["buildings"]
    n = len(grid_gdf)
    area_km2 = grid_gdf["area_m2"] / 1e6

    if not path.exists():
        logger.warning("Buildings file missing (%s) — using synthetic fallback.", path.name)
        count = np.random.randint(20, 300, n)
        total_area = count * 120.0
        return pd.DataFrame({
            "h3_index": grid_gdf["h3_index"],
            "building_count": count,
            "total_building_area_m2": total_area,
            "built_up_ratio": np.minimum(1.0, total_area / grid_gdf["area_m2"].values),
            "building_density_per_km2": count / area_km2.values,
            "avg_building_footprint_m2": total_area / count,
            "building_is_synthetic": True,
        })

    min_lon, min_lat, max_lon, max_lat = extraction_bounds(grid_gdf)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        filtered = chunk[
            (chunk["confidence"] >= 0.65)
            & chunk["latitude"].between(min_lat, max_lat)
            & chunk["longitude"].between(min_lon, max_lon)
        ]
        if not filtered.empty:
            chunks.append(filtered[["latitude", "longitude", "area_in_meters"]])
    if not chunks:
        raise ValueError(f"No building records fell inside the study area in {path.name}.")
    buildings = pd.concat(chunks, ignore_index=True)

    points = gpd.GeoDataFrame(
        buildings,
        geometry=gpd.points_from_xy(buildings["longitude"], buildings["latitude"]),
        crs=CRS_WGS84,
    )
    agg = spatial_join_point_agg(points, grid_gdf, value_cols=["area_in_meters"], agg="sum").rename(
        columns={"point_count": "building_count", "area_in_meters": "total_building_area_m2"}
    )

    result = grid_gdf[["h3_index", "area_m2"]].merge(agg, on="h3_index", how="left")
    result["building_count"] = result["building_count"].fillna(0).astype(int)
    result["total_building_area_m2"] = result["total_building_area_m2"].fillna(0.0)
    result["built_up_ratio"] = np.minimum(1.0, result["total_building_area_m2"] / result["area_m2"])
    result["building_density_per_km2"] = result["building_count"] / (result["area_m2"] / 1e6)
    result["avg_building_footprint_m2"] = (
        result["total_building_area_m2"] / result["building_count"].replace(0, np.nan)
    ).fillna(0.0)
    result["building_is_synthetic"] = False

    logger.info("Buildings: %d real footprints spatially joined into %d cells.", len(points), n)
    return result[[
        "h3_index", "building_count", "total_building_area_m2", "built_up_ratio",
        "building_density_per_km2", "avg_building_footprint_m2", "building_is_synthetic",
    ]]


# --------------------------------------------------------------------------
# 3. Ookla Open Data Mobile Tiles
# --------------------------------------------------------------------------

def load_ookla(grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    tests-weighted download/upload/latency, spatially joined
    from the Ookla tile grid (WKT `tile` polygons) into the H3 study grid.
    """
    path = RAW_DATASETS["ookla"]
    n = len(grid_gdf)

    def _synthetic():
        return pd.DataFrame({
            "h3_index": grid_gdf["h3_index"],
            "ookla_dl_mbps": np.nan,
            "ookla_rtt_ms": np.nan,
            "ookla_tests": 0,
            "ookla_devices": np.nan,
            "ookla_is_synthetic": True,
        })

    if not path.exists():
        logger.warning("Ookla file missing (%s) — using synthetic fallback baseline.", path.name)
        return _synthetic()

    df = pd.read_parquet(path)
    df = df[df["tests"] > 0].copy()
    df["dl_mbps"] = df["avg_d_kbps"] / 1000.0
    df["rtt_ms"] = df["avg_lat_ms"]
    df["geometry"] = df["tile"].apply(shapely_wkt.loads)
    tiles = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_WGS84)
    # Centroid must be computed in a projected CRS (WGS84 degrees distort
    # area/centroid math), then reprojected back for the WGS84 sjoin below.
    tiles["geometry"] = tiles.geometry.to_crs(CRS_PENINSULAR_MYS).centroid.to_crs(CRS_WGS84)

    joined = gpd.sjoin(tiles, grid_gdf[["h3_index", "geometry"]], how="inner", predicate="within")
    if joined.empty:
        logger.warning("Ookla: no tiles fell inside the study grid — using synthetic fallback.")
        return _synthetic()

    # Tests-weighted mean per cell (cells with more real tests get more
    # reliable estimates — an unweighted mean would treat a 1-test tile the
    # same as a 200-test tile).
    joined["dl_x_tests"] = joined["dl_mbps"] * joined["tests"]
    joined["rtt_x_tests"] = joined["rtt_ms"] * joined["tests"]
    has_devices = "devices" in joined.columns
    agg_kwargs = dict(
        dl_x_tests=("dl_x_tests", "sum"),
        rtt_x_tests=("rtt_x_tests", "sum"),
        ookla_tests=("tests", "sum"),
    )
    if has_devices:
        agg_kwargs["ookla_devices"] = ("devices", "sum")
    grouped = joined.groupby("h3_index").agg(**agg_kwargs).reset_index()
    grouped["ookla_dl_mbps"] = grouped["dl_x_tests"] / grouped["ookla_tests"]
    grouped["ookla_rtt_ms"] = grouped["rtt_x_tests"] / grouped["ookla_tests"]

    keep_cols = ["h3_index", "ookla_dl_mbps", "ookla_rtt_ms", "ookla_tests"]
    if has_devices:
        keep_cols.append("ookla_devices")
    result = grid_gdf[["h3_index"]].merge(grouped[keep_cols], on="h3_index", how="left")
    if not has_devices:
        result["ookla_devices"] = np.nan
    result["ookla_is_synthetic"] = result["ookla_dl_mbps"].isna()
    n_real = int((~result["ookla_is_synthetic"]).sum())

    logger.info(
        "Ookla: %d real tiles spatially joined -> %d/%d cells have a real speed-test baseline "
        "(%d cells fall outside tile coverage and remain synthetic).",
        len(tiles), n_real, n, n - n_real,
    )
    return result


# --------------------------------------------------------------------------
# 4. OpenStreetMap (roads, POIs, transit)
# --------------------------------------------------------------------------

def load_osm(grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    POI density (by category), road density, intersection density, and
    nearest-transit-station distance, extracted from the Geofabrik OSM PBF
    via pyrosm and spatially joined into the H3 grid.
    """
    path = RAW_DATASETS["osm"]
    n = len(grid_gdf)
    area_km2 = grid_gdf["area_m2"] / 1e6

    def _synthetic():
        logger.warning("OSM file missing (%s) — using synthetic fallback.", path.name)
        out = pd.DataFrame({
            "h3_index": grid_gdf["h3_index"],
            "poi_education_count": np.random.poisson(1.0, n),
            "poi_healthcare_count": np.random.poisson(0.5, n),
            "poi_retail_food_count": np.random.poisson(4.0, n),
            "poi_office_count": np.random.poisson(3.0, n),
            "poi_recreation_count": np.random.poisson(1.0, n),
            "poi_transport_stop_count": np.random.poisson(2.0, n),
            "road_length_m": _synthetic_seeded_fallback(n, 200, 5000),
            "intersection_count": np.random.poisson(3.0, n),
            "dist_to_nearest_transit_m": _synthetic_seeded_fallback(n, 100, 3000),
            "osm_is_synthetic": True,
        })
        out["road_density_km_per_km2"] = (out["road_length_m"] / 1000.0) / area_km2.values
        out["intersection_density_per_km2"] = out["intersection_count"] / area_km2.values
        return out

    if not path.exists():
        return _synthetic()

    try:
        import pyrosm
        min_lon, min_lat, max_lon, max_lat = extraction_bounds(grid_gdf)
        osm = pyrosm.OSM(str(path), bounding_box=[min_lon, min_lat, max_lon, max_lat])

        poi_categories = {
            "poi_education_count": {"amenity": ["school", "university", "college", "kindergarten"]},
            "poi_healthcare_count": {"amenity": ["hospital", "clinic", "pharmacy", "doctors"]},
            "poi_retail_food_count": {"amenity": ["restaurant", "cafe", "fast_food"], "shop": True},
            "poi_office_count": {"office": True},
            "poi_recreation_count": {"leisure": True, "tourism": True},
        }

        poi_frames = {}
        for feature_name, custom_filter in poi_categories.items():
            pois = osm.get_pois(custom_filter=custom_filter)
            if pois is None or pois.empty:
                poi_frames[feature_name] = pd.DataFrame(columns=["h3_index", "point_count"])
                continue
            pois = pois.copy()
            pois["geometry"] = pois.geometry.to_crs(CRS_PENINSULAR_MYS).centroid.to_crs(CRS_WGS84)
            poi_frames[feature_name] = spatial_join_point_agg(pois, grid_gdf).rename(
                columns={"point_count": feature_name}
            )

        result = grid_gdf[["h3_index", "area_m2"]].copy()
        for feature_name, frame in poi_frames.items():
            result = result.merge(frame, on="h3_index", how="left")
            result[feature_name] = result[feature_name].fillna(0).astype(int)

        # Roads -> real length per cell via polygon overlay (metres, projected CRS)
        roads = osm.get_network(network_type="driving")
        if roads is not None and not roads.empty:
            road_len = spatial_join_line_length(roads, grid_gdf).rename(columns={"length_m": "road_length_m"})
            result = result.merge(road_len, on="h3_index", how="left")
        else:
            result["road_length_m"] = np.nan
        result["road_length_m"] = result["road_length_m"].fillna(0.0)
        result["road_density_km_per_km2"] = (result["road_length_m"] / 1000.0) / (result["area_m2"] / 1e6)

        # Intersections: road network nodes, spatially joined and counted per cell
        network_result = osm.get_network(network_type="driving", nodes=True)
        nodes = network_result[0] if isinstance(network_result, tuple) else None
        if nodes is not None and not nodes.empty:
            intersections = spatial_join_point_agg(nodes, grid_gdf).rename(
                columns={"point_count": "intersection_count"}
            )
            result = result.merge(intersections, on="h3_index", how="left")
        result["intersection_count"] = result.get("intersection_count", pd.Series(0, index=result.index)).fillna(0).astype(int)
        result["intersection_density_per_km2"] = result["intersection_count"] / (result["area_m2"] / 1e6)

        # Transit: real nearest-station distance via sjoin_nearest
        transit = osm.get_pois(custom_filter={"public_transport": True, "railway": ["station", "halt"]})
        cell_centroids = h3_ids_to_geodataframe(result["h3_index"])
        cell_centroids["geometry"] = cell_centroids.geometry.centroid
        if transit is not None and not transit.empty:
            transit = transit.copy()
            transit["geometry"] = transit.geometry.to_crs(CRS_PENINSULAR_MYS).centroid.to_crs(CRS_WGS84)
            result["dist_to_nearest_transit_m"] = nearest_distance_m(cell_centroids, transit)
            transit_counts = spatial_join_point_agg(transit, grid_gdf).set_index("h3_index")
            result["poi_transport_stop_count"] = (
                transit_counts.reindex(result["h3_index"])["point_count"].fillna(0).astype(int).values
            )
        else:
            result["dist_to_nearest_transit_m"] = np.nan
            result["poi_transport_stop_count"] = 0

        result["osm_is_synthetic"] = False
        logger.info("OSM: real POI/road/transit layers spatially joined into %d cells.", n)
        return result.drop(columns=["area_m2"])

    except Exception as exc:  # defensive: no real PBF available to exercise this path in this environment
        logger.warning("OSM parsing failed (%s) — using synthetic fallback.", exc)
        return _synthetic()


# --------------------------------------------------------------------------
# 5 & 6. WorldPop raster + OpenDOSM census (population)
# --------------------------------------------------------------------------

def _resolve_dosm_national_total(dosm: pd.DataFrame, filename: str):
    """
    Bug fixed here: the previous version computed `dosm["population"].sum()`
    over the ENTIRE file — population_malaysia.csv is data.gov.my's national
    time series from 1970-2025, broken out by sex/age/ethnicity, with
    population in THOUSANDS. Summing every row summed every year and every
    overlapping subgroup (totals AND their own male/female/age/ethnicity
    breakdowns) together, producing a meaningless ~9.5 million figure — for
    comparison the actual latest single overall row (2025, both sexes,
    overall age, overall ethnicity) is ~34,231.7 thousand = ~34.2 million,
    Malaysia's real population.

    This isolates that single row instead: filters sex/age/ethnicity
    columns (if present) to their "overall"/"both" aggregate values, takes
    the latest date, and requires exactly one matching row before trusting
    it. If the file's schema doesn't match what's expected, this logs a
    warning and returns None rather than silently falling back to a sum
    that would be wrong in a different way.
    """
    if "population" not in dosm.columns:
        logger.warning("OpenDOSM file %s has no 'population' column — skipping sanity check.", filename)
        return None

    col_map = {c.lower(): c for c in dosm.columns}
    mask = pd.Series(True, index=dosm.index)
    overall_terms = {"overall", "both", "both sexes", "all"}
    for key in ("sex", "age", "ethnicity"):
        if key in col_map:
            values_lower = dosm[col_map[key]].astype(str).str.strip().str.lower()
            match = values_lower.isin(overall_terms)
            if match.any():
                mask &= match

    subset = dosm[mask]
    date_col = next((col_map[c] for c in ("date", "year") if c in col_map), None)
    if date_col is not None and not subset.empty:
        subset = subset[subset[date_col] == subset[date_col].max()]

    if len(subset) != 1:
        logger.warning(
            "OpenDOSM file %s: expected exactly 1 national-aggregate row after filtering to the latest "
            "date + overall sex/age/ethnicity, found %d (columns: %s). Skipping the sanity check rather "
            "than guessing which row is correct — previous behaviour of summing every row produced a "
            "meaningless ~9.5M figure by adding every year and every subgroup together.",
            filename, len(subset), list(dosm.columns),
        )
        return None

    raw_value = float(subset["population"].iloc[0])
    # data.gov.my's population_malaysia.csv reports population in THOUSANDS.
    return raw_value * 1000.0


def load_population(grid_gdf: gpd.GeoDataFrame, osm_df: pd.DataFrame) -> pd.DataFrame:
    """
    population from WorldPop 2020 (zonal statistics: 
    raster-polygon aggregation, not a distance-to-centroid heuristic), plus
    OpenDOSM as an independent district-level consistency check (logged,
    not silently applied as a correction factor — a mismatch is a signal to
    investigate, not to auto-adjust away).

    Daytime vs nighttime exposure: WorldPop's ppp product is a
    RESIDENTIAL (nighttime-like) population estimate. There is no public
    time-of-day population raster for this study area, so `population_day`
    is an explicit MODELLED estimate: residential population reweighted by
    real OSM office/retail density (workplaces pull daytime population in;
    pure-residential cells lose some to commuting out). This assumption is
    intentionally documented here rather than left implicit.
    """
    worldpop_path = RAW_DATASETS["worldpop"]
    dosm_path = RAW_DATASETS["dosm_population"]
    n = len(grid_gdf)
    area_km2 = grid_gdf["area_m2"] / 1e6

    if not worldpop_path.exists():
        logger.warning("WorldPop raster missing (%s) — using synthetic fallback.", worldpop_path.name)
        pop = np.random.uniform(500, 5000, n)
        return pd.DataFrame({
            "h3_index": grid_gdf["h3_index"],
            "population_night": pop,
            "population_day": pop,
            "population_density_per_km2": pop / area_km2.values,
            "population_is_synthetic": True,
        })

    import rasterstats

    stats = rasterstats.zonal_stats(
        grid_gdf.geometry, str(worldpop_path), stats=["sum", "mean"], all_touched=True, nodata=-99999.0,
    )
    pop_sum = np.array([s["sum"] if s["sum"] is not None else 0.0 for s in stats])

    result = pd.DataFrame({"h3_index": grid_gdf["h3_index"].values, "population_night": pop_sum})
    result["population_density_per_km2"] = result["population_night"] / area_km2.values

    office_col = "poi_office_count" if "poi_office_count" in osm_df.columns else None
    retail_col = "poi_retail_food_count" if "poi_retail_food_count" in osm_df.columns else None
    if office_col and retail_col:
        merged = result.merge(osm_df[["h3_index", office_col, retail_col]], on="h3_index", how="left")
        workplace_pull = 1.0 + 0.05 * merged[office_col].fillna(0) + 0.02 * merged[retail_col].fillna(0)
        result["population_day"] = result["population_night"] * 0.85 * workplace_pull.clip(upper=4.0)
    else:
        result["population_day"] = result["population_night"]

    result["population_is_synthetic"] = False

    if dosm_path.exists():
        dosm = pd.read_csv(dosm_path)
        dosm_total = _resolve_dosm_national_total(dosm, dosm_path.name)
        worldpop_total = pop_sum.sum()
        if dosm_total:
            ratio = worldpop_total / dosm_total
            logger.info(
                "Population sanity check (ORDER-OF-MAGNITUDE ONLY — population_malaysia.csv is a NATIONAL "
                "time series, not a Klang Valley or district-level table, so this is not a validation of the "
                "study area's population; it only confirms WorldPop's study-area sum is a plausible small "
                "fraction of Malaysia's national total, not an obviously corrupted raster read): WorldPop "
                "study-area sum = %.0f vs. OpenDOSM national total = %.0f (ratio = %.4f). No correction is "
                "ever applied — this is logged for investigation only.",
                worldpop_total, dosm_total, ratio,
            )
    else:
        logger.warning("OpenDOSM file missing (%s) — skipping population sanity check.", dosm_path.name)

    logger.info("Population: WorldPop zonal statistics computed for %d cells.", n)
    return result


# --------------------------------------------------------------------------
# 7. Transit ridership (temporal demand surge only — no per-cell geometry)
# --------------------------------------------------------------------------

def load_ridership_surge_factor(dates: pd.Series) -> pd.Series:
    """
    ridership-derived daily demand-surge multiplier, broadcast to the
    given per-timestamp date series. data.gov.my's ridership_headline.csv
    is a DAILY total by transport mode with no station-level geometry, so
    it cannot itself be spatially joined into H3 cells — station-level
    "nearest transit" geometry instead comes from OSM (see load_osm). Here
    it only informs the TEMPORAL shape of demand: days with higher real
    ridership get a proportionally higher surge multiplier applied to the
    existing peak-hour demand curve.

    `dates` must be a pandas Series of Timestamps (one per row of the
    panel being built), already normalized to midnight.
    """
    path = RAW_DATASETS["ridership"]
    if not path.exists():
        logger.warning("Ridership file missing (%s) — using flat surge factor.", path.name)
        return pd.Series(1.0, index=dates.index)

    df = pd.read_csv(path, parse_dates=["date"])
    # The CSV has individual mode columns (bus_rkl, rail_lrt_ampang, etc.)
    # rather than a single 'ridership' column — sum numeric columns per date
    # to get a daily total. Bug fixed here: this used to sum EVERY numeric
    # column indiscriminately, including national/non-Klang-Valley services
    # (ETS, Intercity, Komuter Utara, Tebrau — inter-city and northern/
    # southern-region rail, nothing to do with Klang Valley commuter
    # demand), diluting a "Klang Valley activity" signal with unrelated
    # national rail volume. Now excludes any column matching a known
    # non-Klang-Valley keyword; logs exactly what was excluded so this is
    # auditable rather than a silent filter.
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_klang_valley_keywords = ("ets", "intercity", "komuter_utara", "tebrau")
    klang_valley_cols = [c for c in numeric_cols if not any(kw in c.lower() for kw in non_klang_valley_keywords)]
    excluded_cols = [c for c in numeric_cols if c not in klang_valley_cols]
    if excluded_cols:
        logger.info(
            "Ridership: excluded non-Klang-Valley service columns from the surge signal: %s "
            "(kept as Klang Valley demand indicators: %s)",
            excluded_cols, klang_valley_cols,
        )
    numeric_cols = klang_valley_cols
    if not numeric_cols:
        logger.warning("Ridership CSV has no Klang-Valley-relevant numeric columns — using flat surge factor.")
        return pd.Series(1.0, index=dates.index)
    df["_total_ridership"] = df[numeric_cols].sum(axis=1)
    daily_total = df.groupby("date")["_total_ridership"].sum()
    if daily_total.empty:
        return pd.Series(1.0, index=dates.index)

    norm = daily_total / daily_total.mean()
    factor_by_date = norm.to_dict()
    surge = dates.map(lambda d: factor_by_date.get(pd.Timestamp(d), 1.0))

    file_min_date, file_max_date = daily_total.index.min(), daily_total.index.max()
    panel_min_date, panel_max_date = dates.min(), dates.max()
    if surge.min() == 1.0 and surge.max() == 1.0:
        # Bug fixed here: this used to fire silently for every panel run
        # because the panel's hardcoded date range (2026-07-20 onward) fell
        # entirely after ridership_headline.csv's coverage (through
        # 2026-06-30) — every date lookup fell back to the 1.0 default, so
        # ridership contributed nothing despite the file being real and the
        # aggregation logic being correct. build_dynamic_panel now derives
        # the panel's date range FROM this file's actual coverage (see
        # resolve_panel_date_range), so this warning firing again would mean
        # that derivation itself has regressed.
        logger.warning(
            "Ridership surge factor is flat at 1.0 for all %d panel dates (panel range %s-%s) despite "
            "%s covering %s-%s — the panel's date range does not overlap the ridership file's coverage, "
            "so ridership is contributing NO temporal signal. Check resolve_panel_date_range().",
            len(dates), panel_min_date.date(), panel_max_date.date(),
            path.name, file_min_date.date(), file_max_date.date(),
        )
    else:
        logger.info(
            "Ridership: real daily surge factors applied (range %.2f-%.2f) for panel dates %s-%s, "
            "drawn from %s's real coverage %s-%s.",
            surge.min(), surge.max(), panel_min_date.date(), panel_max_date.date(),
            path.name, file_min_date.date(), file_max_date.date(),
        )
    return surge


def resolve_panel_date_range(days: int, fallback_start: str = "2026-07-20") -> pd.Timestamp:
    """
    Bug fixed here: the panel's start date used to be hardcoded to
    "2026-07-20" regardless of what dates ridership_headline.csv actually
    covered. The file's real coverage ends 2026-06-30, so every panel run
    picked a date range with ZERO overlap — ridership_surge silently came
    back flat (1.0 for every date) even though the aggregation code itself
    was correct.

    This instead picks a `days`-length window ending on the ridership
    file's own most recent available date, so the panel's dates are
    GUARANTEED to overlap real ridership data whenever the file exists.
    Falls back to `fallback_start` (with a clear warning) only if the file
    is missing or unreadable — in that case there is no ridership signal to
    align to regardless of what date is chosen.
    """
    path = RAW_DATASETS["ridership"]
    if not path.exists():
        logger.warning(
            "Ridership file missing (%s) — panel date range falls back to hardcoded %s (no ridership "
            "signal available regardless of date choice).",
            path.name, fallback_start,
        )
        return pd.Timestamp(fallback_start)

    try:
        df = pd.read_csv(path, parse_dates=["date"], usecols=["date"])
        if df.empty:
            raise ValueError("ridership file has no rows")
        latest_date = df["date"].max().normalize()
    except Exception as exc:
        logger.warning(
            "Could not read dates from ridership file %s (%s) — panel date range falls back to "
            "hardcoded %s.", path.name, exc, fallback_start,
        )
        return pd.Timestamp(fallback_start)

    start = latest_date - pd.Timedelta(days=days - 1)
    logger.info(
        "Panel date range resolved from ridership file's real coverage: %s to %s (%d days), so ridership "
        "surge factors will be real, non-flat values rather than a silent 1.0 fallback.",
        start.date(), latest_date.date(), days,
    )
    return start


# --------------------------------------------------------------------------
# Orchestration: static features + dynamic panel
# --------------------------------------------------------------------------

def build_static_features(grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Join all static (non-time-varying) real layers into one per-cell table."""
    towers = load_towers(grid_gdf)
    buildings = load_buildings(grid_gdf)
    osm = load_osm(grid_gdf)
    population = load_population(grid_gdf, osm)

    static_df = grid_gdf[["h3_index"]].merge(towers, on="h3_index", how="left")
    static_df = static_df.merge(buildings, on="h3_index", how="left")
    static_df = static_df.merge(osm, on="h3_index", how="left")
    static_df = static_df.merge(population, on="h3_index", how="left")
    return static_df


def build_dynamic_panel(grid_gdf: gpd.GeoDataFrame, static_df: pd.DataFrame,
                         ookla_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Assemble the hourly cell x time SIMULATION panel.

    IMPORTANT — read before using this table for anything but a pipeline /
    dashboard demo: no per-hour QoE measurement source exists for this
    project (no organiser data, and Ookla is a single static quarterly
    average per cell). EVERY hourly dl/ul/rtt/jitter/loss value below is
    generated by a diurnal-demand formula — for cells with a real Ookla
    anchor, the formula's *level* is re-centred on that one real number;
    for cells without one, the level is also formula-derived. In neither
    case is the *hourly shape* a measurement. Two columns make this
    unambiguous rather than leaving it in one easily-misread flag:

      qoe_level_is_ookla_anchored : True for the ~41% of cells where a real
          Ookla quarterly average existed to re-centre the formula on.
      qoe_hourly_shape_is_simulated : always True. Included explicitly so
          nobody downstream infers "anchored == real hourly measurement".

    If your goal is a defensible, non-simulated modelling dataset, use
    build_static_qoe_baseline() instead — it never fabricates a value.
    """
    panel_start = resolve_panel_date_range(days)
    timestamps = pd.date_range(start=panel_start, periods=24 * days, freq="1h")
    dates = pd.Series([pd.Timestamp(ts.date()) for ts in timestamps])
    ridership_surge = load_ridership_surge_factor(dates)

    static_dict = static_df.set_index("h3_index").to_dict("index")
    ookla_dict = ookla_df.set_index("h3_index").to_dict("index")
    cells = grid_gdf["h3_index"].values

    records = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak = 1 if (not is_weekend and (8 <= hour <= 10 or 17 <= hour <= 20)) else 0
        diurnal = np.sin((hour - 6) / 24.0 * 2 * np.pi)
        diurnal = max(0.1, (diurnal + 1.0) / 2.0)
        surge = ridership_surge.iloc[i]
        demand_mult = (1.0 + is_peak * 0.8 + (1 - is_weekend) * 0.3 * diurnal) * surge

        for cell in cells:
            s = static_dict[cell]
            o = ookla_dict.get(cell, {})
            pop = s.get("population_night") or 1000.0
            poi = s.get("poi_retail_food_count") or 0
            load_factor = (pop / 5000.0 + poi / 15.0) * demand_mult

            real_dl = o.get("ookla_dl_mbps")
            real_rtt = o.get("ookla_rtt_ms")
            has_real_qoe = pd.notna(real_dl) and pd.notna(real_rtt)
            # Explicit provenance string per review request — a human-readable
            # complement to the qoe_level_is_ookla_anchored boolean.
            qoe_baseline_source = "ookla_q4_2024_static_anchor" if has_real_qoe else "no_real_anchor_simulated_only"

            if has_real_qoe:
                base_dl = max(2.0, real_dl / (1.0 + 0.3 * (demand_mult - 1.0)))
                base_rtt = max(5.0, real_rtt * (1.0 + 0.15 * (demand_mult - 1.0)))
            else:
                base_dl = max(2.0, 68.0 / (1.0 + 0.8 * load_factor) + np.random.normal(0, 4.0))
                base_rtt = max(10.0, 18.0 * (1.0 + 0.9 * load_factor) + np.random.normal(0, 5.0))

            base_ul = max(1.0, 22.0 / (1.0 + 0.5 * load_factor) + np.random.normal(0, 2.0))
            jitter = max(1.0, base_rtt * 0.2 + np.random.normal(0, 1.0))
            loss_pct = float(np.clip(load_factor * 0.8 - 0.5 + np.random.normal(0, 0.2), 0.0, 10.0))
            real_tests = o.get("ookla_tests")

            # Bug fixed here: this used to do
            #   sample_count = int(real_tests) if has_real_qoe else poisson(...)
            # i.e. for anchored cells it assigned the REAL QUARTERLY Ookla
            # test count IDENTICALLY to all 168 simulated hours — e.g. a
            # cell with 342 real tests over the whole quarter would show
            # "342 tests" at every single simulated hour, inflating
            # qoe_confidence_score (= min(1, test_sample_count/10)) to a
            # constant, misleadingly high value all week. The real
            # quarterly total says nothing about per-hour sample density.
            # Fixed: derive an HOURLY-equivalent Poisson rate from the real
            # quarterly total (quarterly total / ~2160 hours/quarter),
            # scaled by the same demand_mult as everything else, so
            # anchored cells get a plausible simulated per-hour count
            # informed by their real overall volume rather than either a
            # generic constant (unanchored cells) or the literal quarterly
            # number (the bug). The real quarterly totals are preserved
            # separately and honestly below, never presented as hourly.
            HOURS_PER_OOKLA_QUARTER = 2160  # ~90 days x 24h
            real_devices = o.get("ookla_devices")
            if has_real_qoe and pd.notna(real_tests) and real_tests:
                hourly_rate = max(0.3, (real_tests / HOURS_PER_OOKLA_QUARTER) * demand_mult)
            else:
                hourly_rate = 6 + load_factor * 5
            sample_count = int(np.random.poisson(hourly_rate))

            records.append({
                "h3_index": cell,
                "timestamp": ts,
                "dl_median": base_dl, "dl_mean": base_dl * 1.05, "dl_std": base_dl * 0.15, "dl_p95": base_dl * 1.3,
                "ul_median": base_ul, "ul_mean": base_ul * 1.05, "ul_std": base_ul * 0.15,
                "rtt_median": base_rtt, "rtt_mean": base_rtt * 1.05, "rtt_std": base_rtt * 0.2,
                "jitter_median": jitter, "loss_median": loss_pct,
                "test_sample_count": sample_count, "unique_device_count": max(1, int(sample_count * 0.7)),
                # Real, honestly-labeled QUARTERLY totals (constant across all
                # 168 hours for a given cell by definition — they are NOT a
                # per-hour figure and must never be treated as one).
                "ookla_quarter_test_count": float(real_tests) if pd.notna(real_tests) else np.nan,
                "ookla_quarter_device_count": float(real_devices) if pd.notna(real_devices) else np.nan,
                "qoe_baseline_source": qoe_baseline_source,
                "qoe_level_is_ookla_anchored": has_real_qoe,
                "qoe_hourly_shape_is_simulated": True,
                # Explicit export of the ridership signal, per review request
                # — previously only folded silently into demand_mult with no
                # standalone column a modelling team could inspect directly.
                "ridership_daily_surge_factor": float(surge),
            })

    return pd.DataFrame(records)


def build_static_qoe_baseline(static_df: pd.DataFrame, ookla_df: pd.DataFrame) -> pd.DataFrame:
    """
    The honest, non-simulated, modelling-ready alternative to the hourly
    panel. One row per H3 cell. No fabricated values anywhere:

      - ookla_dl_mbps / ookla_rtt_ms / ookla_tests are the REAL Ookla
        quarterly averages, left as NaN (not imputed, not guessed) for the
        ~59% of cells with no tile coverage.
      - Every predictor is a real static GIS feature already computed in
        build_static_features() (population, POIs, buildings, roads,
        towers) — none of them depend on QoE, so there is no circularity
        of the kind documented in config.HIGH_LEAKAGE_RISK_FEATURES.

    Recommended use ("Path B" from the review): train/validate a model on
    the ~41% of cells with a real Ookla label to predict which of the
    remaining ~59% of (real-feature-only) cells are likely to have poor
    average mobile QoE. This is a smaller, less flashy claim than "predicts
    hourly congestion", but every number behind it is real, which the
    hourly panel cannot currently claim.
    """
    df = static_df.merge(ookla_df[["h3_index", "ookla_dl_mbps", "ookla_rtt_ms", "ookla_tests"]],
                          on="h3_index", how="left")
    df["has_real_ookla_label"] = df["ookla_dl_mbps"].notna()
    df["qoe_risk_label"] = np.select(
        [df["ookla_dl_mbps"] < 10.0, df["ookla_dl_mbps"] < 25.0],
        ["high_risk", "moderate_risk"],
        default="low_risk",
    )
    df.loc[~df["has_real_ookla_label"], "qoe_risk_label"] = None  # honest: unknown, not guessed
    logger.info(
        "Static QoE baseline: %d/%d cells have a real Ookla label; the remaining cells are "
        "prediction targets (real features, unknown QoE) rather than training rows.",
        int(df["has_real_ookla_label"].sum()), len(df),
    )
    return df


def load_real_or_baseline_datasets(grid_gdf: gpd.GeoDataFrame, days: int = 7):
    """Public entry point kept for backward compatibility with run_pipeline.py."""
    static_df = build_static_features(grid_gdf)
    ookla_df = load_ookla(grid_gdf)
    dynamic_df = build_dynamic_panel(grid_gdf, static_df, ookla_df, days=days)
    return static_df, dynamic_df
