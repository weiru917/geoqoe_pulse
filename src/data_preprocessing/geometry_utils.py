"""
GeoQoE Pulse Geometry Module.

real GeoPandas geometry: a study-boundary polygon, complete/non-redundant H3
coverage via h3.polygon_to_cells, and CRS-correct area/perimeter figures
(computed in a projected CRS, stored/exported in WGS84).
"""
import logging
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box, shape

from config import (
    CRS_WGS84,
    CRS_PENINSULAR_MYS,
    STUDY_BOUNDARY_PATH,
    STUDY_BBOX,
    H3_PRIMARY_RESOLUTION,
    KLANG_VALLEY_DISTRICT_NAMES,
    SPATIAL_HOLDOUT_RESOLUTION,
)
from src.data_preprocessing.mlab_diurnal_shape import DISTRICT_SHAPE_RESOLUTION

logger = logging.getLogger(__name__)


def load_study_boundary() -> gpd.GeoDataFrame:
    """
    Load the real study-area boundary polygon (WGS84).

    Falls back to a rectangular polygon built from STUDY_BBOX only if the
    boundary file is missing, and logs a clear warning when that happens —
    the rectangular fallback should never be silently mistaken for a real
    administrative boundary.
    """
    if STUDY_BOUNDARY_PATH.exists():
        boundary = gpd.read_file(STUDY_BOUNDARY_PATH)
        if boundary.crs is None:
            boundary = boundary.set_crs(CRS_WGS84)
        else:
            boundary = boundary.to_crs(CRS_WGS84)

        # Defensive scope filter: study_boundary.geojson has already been
        # trimmed to Klang Valley core districts (see
        # scripts/trim_boundary.py), but if this file is ever regenerated
        # from a raw, unfiltered "NAME_1 == Selangor" GADM extract, this
        # filter stops out-of-scope districts (e.g. Hulu Selangor, Sabak
        # Bernam — rural/exurban, never part of any Klang Valley definition)
        # from silently re-entering the pipeline. Matches on NAME_2 (falls
        # back to NAME_1 for Putrajaya, which has no NAME_2).
        name_col = "NAME_2" if "NAME_2" in boundary.columns else None
        if name_col:
            name_1 = boundary["NAME_1"] if "NAME_1" in boundary.columns else pd.Series("", index=boundary.index)
            in_scope = boundary[name_col].isin(KLANG_VALLEY_DISTRICT_NAMES) | name_1.isin(KLANG_VALLEY_DISTRICT_NAMES)
            n_dropped = (~in_scope).sum()
            if n_dropped:
                dropped_names = boundary.loc[~in_scope, name_col].tolist()
                logger.warning(
                    "Study boundary scope filter dropped %d out-of-scope district(s) not in "
                    "KLANG_VALLEY_DISTRICT_NAMES: %s. If this fires on a file that was already "
                    "supposed to be pre-trimmed, check data/boundary/study_boundary.geojson.",
                    n_dropped, dropped_names,
                )
            boundary = boundary.loc[in_scope].copy()

        logger.info(
            "Loaded real study boundary polygon from %s (%d feature(s), scope: Klang Valley core, "
            "including Kuala Lumpur federal territory — see config.KLANG_VALLEY_DISTRICT_NAMES).",
            STUDY_BOUNDARY_PATH.name, len(boundary),
        )
        return boundary

    logger.warning(
        "STUDY_BOUNDARY_PATH (%s) not found — falling back to a rectangular "
        "bounding-box polygon. Replace data/boundary/study_boundary.geojson "
        "with a real administrative boundary (e.g. DOSM/GADM) for production use.",
        STUDY_BOUNDARY_PATH,
    )
    rect = box(STUDY_BBOX["min_lon"], STUDY_BBOX["min_lat"], STUDY_BBOX["max_lon"], STUDY_BBOX["max_lat"])
    return gpd.GeoDataFrame({"name": ["bbox_fallback"]}, geometry=[rect], crs=CRS_WGS84)


def _boundary_to_h3shape(boundary_geom: Polygon):
    """Convert a shapely (Multi)Polygon into the LatLngPoly/LatLngMultiPoly
    shape h3-py v4 expects (lat, lng order — the opposite of shapely's x, y)."""
    geo_interface = boundary_geom.__geo_interface__
    # h3.geo_to_h3shape expects a GeoJSON-like mapping with (lng, lat) coords,
    # exactly as shapely produces — h3 handles the axis-order internally.
    return h3.geo_to_h3shape(geo_interface)


def generate_h3_study_grid(res: int = H3_PRIMARY_RESOLUTION) -> gpd.GeoDataFrame:
    """
    Generate a complete, non-redundant H3 hex grid covering the real study
    boundary polygon

    Returns a GeoDataFrame with: h3_index, geometry (polygon, WGS84),
    centroid_lat, centroid_lon, area_m2, perimeter_m.
    """
    boundary = load_study_boundary()
    # Dissolve to a single (Multi)Polygon in case the boundary file has
    # multiple features (e.g. multiple districts).
    boundary_geom = boundary.geometry.union_all()

    h3shape = _boundary_to_h3shape(boundary_geom)
    cells = set(h3.h3shape_to_cells(h3shape, res))

    if not cells:
        raise ValueError(
            "H3 grid generation produced zero cells — check that the study "
            "boundary polygon and H3 resolution are compatible."
        )

    records = []
    for cell in cells:
        boundary_pts = h3.cell_to_boundary(cell)  # list of (lat, lng)
        polygon = Polygon([(lng, lat) for lat, lng in boundary_pts])
        lat, lng = h3.cell_to_latlng(cell)
        records.append({
            "h3_index": cell,
            "geometry": polygon,
            "centroid_lat": lat,
            "centroid_lon": lng,
            # Coarse parent-cell grouping key for SPATIAL HOLDOUT splits (see
            # config.SPATIAL_HOLDOUT_RESOLUTION). Group by this and hold out
            # whole groups; a random split on individual h3_index values can
            # still leak across spatially autocorrelated neighbours.
            "h3_parent_res6": h3.cell_to_parent(cell, SPATIAL_HOLDOUT_RESOLUTION),
            # Coarser parent (~20 km) used to bucket cells into "districts"
            # for PER-DISTRICT M-Lab diurnal curves. Applying one citywide
            # curve to every cell is what makes the anchored panel rank-one;
            # this key is the join handle that breaks it once a real
            # per-district M-Lab export exists (see mlab_diurnal_shape).
            "h3_parent_res5": h3.cell_to_parent(cell, DISTRICT_SHAPE_RESOLUTION),
        })

    grid_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_WGS84)

    # Area/perimeter must be computed in a projected (metres) CRS, never in
    # raw WGS84 degrees — reproject, measure, then keep the WGS84 geometry
    # for storage/export.
    projected = grid_gdf.geometry.to_crs(CRS_PENINSULAR_MYS)
    grid_gdf["area_m2"] = projected.area.values
    grid_gdf["perimeter_m"] = projected.length.values

    # Defensive uniqueness check (H3 cell IDs are unique by construction,
    # but a duplicate would indicate a bug upstream, e.g. a boundary that
    # crosses the antimeridian or a resolution mismatch).
    n_dupes = grid_gdf["h3_index"].duplicated().sum()
    if n_dupes:
        raise ValueError(f"H3 grid generation produced {n_dupes} duplicate cell(s).")

    logger.info(
        "Generated %d H3 res-%d cells covering the real study boundary "
        "(mean cell area = %.1f m^2, expected h3.average_hexagon_area = %.1f m^2; "
        "%d distinct res-%d parent groups for spatial holdout splits; %d res-%d district groups "
        "for per-district diurnal curves).",
        len(grid_gdf), res, grid_gdf["area_m2"].mean(),
        h3.average_hexagon_area(res, unit="m^2"),
        grid_gdf["h3_parent_res6"].nunique(), SPATIAL_HOLDOUT_RESOLUTION,
        grid_gdf["h3_parent_res5"].nunique(), DISTRICT_SHAPE_RESOLUTION,
    )
    return grid_gdf.sort_values("h3_index").reset_index(drop=True)


def export_grid_geometry(grid_gdf: gpd.GeoDataFrame, processed_dir: Path) -> Tuple[Path, Path]:
    """
    Export the static grid geometry once, in both GeoParquet and GeoJSON, 
    can be used in dashboard
    """
    out_cols = ["h3_index", "geometry", "centroid_lat", "centroid_lon", "area_m2", "perimeter_m",
                "h3_parent_res6", "h3_parent_res5"]
    export_gdf = grid_gdf[out_cols].copy()

    parquet_path = processed_dir / "grid_geometry.parquet"
    geojson_path = processed_dir / "grid_geometry.geojson"

    export_gdf.to_parquet(parquet_path, index=False)
    export_gdf.to_file(geojson_path, driver="GeoJSON")

    logger.info("Exported grid geometry -> %s, %s", parquet_path.name, geojson_path.name)
    return parquet_path, geojson_path


def h3_ids_to_geodataframe(h3_ids, crs: str = CRS_WGS84) -> gpd.GeoDataFrame:
    """Utility: build a GeoDataFrame of H3 polygons from a plain list of h3_index
    strings (used by modules that need geometry mid-pipeline, e.g. spatial joins)."""
    records = []
    for cell in h3_ids:
        boundary_pts = h3.cell_to_boundary(cell)
        polygon = Polygon([(lng, lat) for lat, lng in boundary_pts])
        records.append({"h3_index": cell, "geometry": polygon})
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
