"""
build_dashboard_data.py — converts the REAL geoqoe_pulse pipeline output
(parquet + geojson + metadata.json) into compact JSON files the static
HTML/JS dashboard can fetch directly. No synthetic data is generated here;
everything is read from the actual pipeline output in data/outputs and
data/processed.

Run once from the repo root:
    python3 build_dashboard_data.py
"""

import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import pyparquet_min as pq  # noqa: E402
import klang_valley_gazetteer as gaz  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

STATIC_PARQUET = REPO / "data/outputs/cell_static_features.parquet"
PANEL_PARQUET = REPO / "data/outputs/cell_time_dataset.parquet"
GRID_GEOJSON = REPO / "data/processed/grid_geometry.geojson"
METADATA_JSON = REPO / "data/outputs/metadata.json"
QUARTERLY_PARQUET = REPO / "data/outputs/cell_quarter_dataset.parquet"
FINALS_DIR = REPO / "handoff/finals_ai"
FINALS_ASSET = FINALS_DIR / "dashboard_assets/2026Q1_to_2026Q2_cells.geojson"
FINALS_BASE_METADATA = FINALS_DIR / "dashboard_assets/dashboard_metadata.json"
FINALS_FREEZE = FINALS_DIR / "audit/08_model_freeze.json"
FINALS_METRICS = FINALS_DIR / "audit/09_final_test_metrics.csv"
FINALS_TOPK = FINALS_DIR / "audit/09_final_test_topk.csv"
FINALS_SHAP = FINALS_DIR / "audit/10_shap_family_importance.csv"

# Hotspot confidence-band code -> label (from feature_dictionary.xlsx)
HOTSPOT_CODES = {
    0: "cold_99", 1: "cold_95", 2: "cold_90", 3: "not_significant",
    4: "hot_90", 5: "hot_95", 6: "hot_99",
}
MORAN_CODES = {0: "not_significant", 1: "LL", 2: "LH", 3: "HL", 4: "HH"}


def r(v, nd=2):
    return None if v is None else round(v, nd)


def build_quarterly_assets(grid):
    """Export observed 2026Q1 status separately from the AI forecast."""
    columns = [
        "h3_index", "quarter_label", "is_below_area_norm", "ookla_dl_mbps",
        "ookla_ul_mbps", "ookla_rtt_ms", "ookla_tests",
        "dl_percentile_in_quarter", "population_density_per_km2",
        "population_day", "district_population", "viirs_rad_quarter",
        "viirs_rad_yoy_log_ratio", "tower_density_per_km2",
        "dist_to_nearest_tower_m", "building_density_per_km2", "built_up_ratio",
        "landcover_built_up_pct", "road_density_km_per_km2",
        "intersection_density_per_km2", "dist_to_nearest_transit_m",
    ]
    _, quarterly, n_quarterly = pq.read_parquet(str(QUARTERLY_PARQUET), columns=columns)
    geometry_by_h3 = {
        feature["properties"]["h3_index"]: feature["geometry"]
        for feature in grid["features"]
    }
    features = []
    for i in range(n_quarterly):
        if quarterly["quarter_label"][i] != "2026Q1":
            continue
        h3 = quarterly["h3_index"][i]
        geometry = geometry_by_h3.get(h3)
        if geometry is None:
            continue
        underperforming = bool(quarterly["is_below_area_norm"][i])
        properties = {name: quarterly[name][i] for name in columns}
        properties["current_status"] = (
            "currently_underperforming" if underperforming
            else "currently_not_underperforming"
        )
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})

    observed = {
        "type": "FeatureCollection",
        "name": "GeoQoE 2026Q1 observed quarterly status",
        "features": features,
    }
    (OUT / "quarterly_observed_2026Q1.geojson").write_text(json.dumps(observed))
    print(f"  wrote quarterly_observed_2026Q1.geojson ({len(features)} measured cells)")

    shutil.copyfile(FINALS_ASSET, OUT / "finals_quarterly.geojson")
    metadata = json.loads(FINALS_BASE_METADATA.read_text())
    freeze = json.loads(FINALS_FREEZE.read_text())
    with FINALS_METRICS.open(newline="") as handle:
        metrics = {row["model"]: row for row in csv.DictReader(handle)}
    with FINALS_TOPK.open(newline="") as handle:
        top_k = [row for row in csv.DictReader(handle) if row["model"] == "xgb_full"]
    with FINALS_SHAP.open(newline="") as handle:
        shap_families = list(csv.DictReader(handle))

    selected = metrics["xgb_full"]
    baseline = metrics["constant"]
    metadata.update({
        "pipeline_version": "3.3.0",
        "model_family": freeze["model_family"],
        "feature_count": freeze["feature_count"],
        "task": freeze["task"],
        "score_interpretation": freeze["score_interpretation"],
        "risk_thresholds": {"high_max_rank": 50, "elevated_max_rank": 200, "lower_min_rank": 201},
        "final_test": {
            "pr_auc": float(selected["pr_auc"]),
            "roc_auc": float(selected["roc_auc"]),
            "no_skill_pr_auc": float(baseline["pr_auc"]),
            "total_events": int(top_k[0]["total_events"]),
            "event_rate": int(top_k[0]["total_events"]) / metadata["eligible_cells"],
            "top_k": [
                {
                    "k": int(row["k"]),
                    "captured_events": int(row["captured_events"]),
                    "total_events": int(row["total_events"]),
                    "precision": float(row["precision_at_k"]),
                    "recall": float(row["recall_at_k"]),
                    "lift": float(row["lift_at_k"]),
                }
                for row in top_k
            ],
        },
        "shap_family_importance": [
            {"family": row["family"], "importance_share_pct": float(row["importance_share_pct"])}
            for row in shap_families
        ],
        "observed_layer": {
            "quarter": "2026Q1",
            "measured_cells": len(features),
            "underperforming_cells": sum(
                f["properties"]["current_status"] == "currently_underperforming"
                for f in features
            ),
        },
    })
    (OUT / "finals_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("  copied finals_quarterly.geojson and wrote enriched finals_metadata.json")


def main():
    print("Reading metadata.json …")
    metadata = json.loads(METADATA_JSON.read_text())

    print("Reading study_boundary/grid_geometry.geojson …")
    grid = json.loads(GRID_GEOJSON.read_text())

    print("Reading cell_static_features.parquet (real GIS features) …")
    _, static, n_static = pq.read_parquet(str(STATIC_PARQUET), columns=[
        "h3_index", "h3_parent_res6", "tower_count", "tower_density_per_km2",
        "dist_to_nearest_tower_m", "tower_is_synthetic", "building_count",
        "built_up_ratio", "building_is_synthetic", "poi_education_count",
        "poi_healthcare_count", "poi_retail_food_count", "poi_office_count",
        "poi_recreation_count", "road_density_km_per_km2",
        "poi_transport_stop_count", "dist_to_nearest_transit_m",
        "osm_is_synthetic", "population_night", "population_day",
        "population_is_synthetic", "demand_pressure_base",
    ])
    print(f"  {n_static} static rows")

    static_by_h3 = {static["h3_index"][i]: i for i in range(n_static)}

    # --- Enrich the real hex geometry with static attributes, for the map ---
    print("Joining static features onto hex geometry …")
    for feat in grid["features"]:
        h3 = feat["properties"]["h3_index"]
        i = static_by_h3.get(h3)
        if i is None:
            continue
        feat["properties"].update({
            "tower_count": static["tower_count"][i],
            "population_night": r(static["population_night"][i], 1),
            "population_day": r(static["population_day"][i], 1),
            "built_up_ratio": r(static["built_up_ratio"][i], 4),
            "road_density_km_per_km2": r(static["road_density_km_per_km2"][i], 2),
            "poi_transport_stop_count": static["poi_transport_stop_count"][i],
            "poi_office_count": static["poi_office_count"][i],
            "poi_retail_food_count": static["poi_retail_food_count"][i],
            "poi_education_count": static["poi_education_count"][i],
            "poi_healthcare_count": static["poi_healthcare_count"][i],
            "dist_to_nearest_tower_m": r(static["dist_to_nearest_tower_m"][i], 1),
            "dist_to_nearest_transit_m": r(static["dist_to_nearest_transit_m"][i], 1),
            "demand_pressure_base": r(static["demand_pressure_base"][i], 1),
        })

    # --- Human-readable label for each hex, nearest-neighbour matched to a
    # curated gazetteer of real Klang Valley localities (dashboard.js shows
    # this instead of the raw H3 index everywhere a cell is listed) ---
    for feat in grid["features"]:
        p = feat["properties"]
        p["place_name"], _dist_km = gaz.nearest_place_label(p["centroid_lat"], p["centroid_lon"])

    (OUT / "h3_grid.geojson").write_text(json.dumps(grid))
    print(f"  wrote h3_grid.geojson ({len(grid['features'])} hexes)")

    # --- Panel (real simulated-scenario QoE data) ---
    print("Reading cell_time_dataset.parquet (this is the big one, ~1-2 min) …")
    panel_cols = [
        "h3_index", "timestamp", "dl_median", "ul_median", "rtt_median",
        "jitter_median", "loss_median", "test_sample_count", "target_qoe_index",
        "target_service_quality_1to5", "target_is_degraded",
        "target_degradation_class", "target_is_degraded_next_1h",
        "demand_pressure_index", "gi_star_zscore_rtt_median",
        "hotspot_category_rtt_median_encoded",
        "local_moran_quadrant_rtt_median_encoded",
        "historical_hotspot_freq_rtt_median", "qoe_confidence_score",
        "priority_exposure_score", "priority_base_score",
        "qoe_level_is_ookla_anchored",
    ]
    _, panel, n_panel = pq.read_parquet(str(PANEL_PARQUET), columns=panel_cols)
    print(f"  {n_panel} panel rows")

    # Sort into a regular (timestamp-major, cell-minor) grid so the frontend
    # can slice a full map snapshot with simple index arithmetic.
    h3_ids = sorted(set(panel["h3_index"]))
    h3_index_of = {h: i for i, h in enumerate(h3_ids)}
    timestamps = sorted(set(panel["timestamp"]))
    ts_index_of = {t: i for i, t in enumerate(timestamps)}
    C, T = len(h3_ids), len(timestamps)
    print(f"  {C} cells x {T} timestamps = {C * T} (panel has {n_panel})")

    size = C * T
    qoe = [None] * size
    degr_class = [None] * size
    demand = [None] * size
    gi_z = [None] * size
    hotspot_code = [None] * size
    moran_code = [None] * size
    hist_freq = [None] * size
    confidence = [None] * size
    dl = [None] * size
    ul = [None] * size
    rtt = [None] * size
    jitter = [None] * size
    loss = [None] * size
    next_risk = [None] * size
    priority = [None] * size
    anchored = [None] * size

    for i in range(n_panel):
        c = h3_index_of[panel["h3_index"][i]]
        t = ts_index_of[panel["timestamp"][i]]
        idx = t * C + c
        qoe[idx] = r(panel["target_qoe_index"][i], 1)
        degr_class[idx] = panel["target_degradation_class"][i]
        demand[idx] = r(panel["demand_pressure_index"][i], 1)
        gi_z[idx] = r(panel["gi_star_zscore_rtt_median"][i], 2)
        hotspot_code[idx] = panel["hotspot_category_rtt_median_encoded"][i]
        moran_code[idx] = panel["local_moran_quadrant_rtt_median_encoded"][i]
        hist_freq[idx] = r(panel["historical_hotspot_freq_rtt_median"][i], 3)
        confidence[idx] = r(panel["qoe_confidence_score"][i], 2)
        dl[idx] = r(panel["dl_median"][i], 1)
        ul[idx] = r(panel["ul_median"][i], 1)
        rtt[idx] = r(panel["rtt_median"][i], 1)
        jitter[idx] = r(panel["jitter_median"][i], 1)
        loss[idx] = r(panel["loss_median"][i], 2)
        nr = panel["target_is_degraded_next_1h"][i]
        next_risk[idx] = int(nr) if nr is not None else None
        priority[idx] = r(panel["priority_base_score"][i], 4)
        anchored[idx] = bool(panel["qoe_level_is_ookla_anchored"][i])

    print("Writing panel.json …")
    (OUT / "panel.json").write_text(json.dumps({
        "h3_ids": h3_ids,
        "timestamps_iso": [
            __import__("datetime").datetime.utcfromtimestamp(t / 1e6)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
            for t in timestamps
        ],
        "num_cells": C,
        "num_timestamps": T,
        "order": "index = t * num_cells + c",
        "fields": {
            "qoe": qoe, "degradation_class": degr_class, "demand_pressure": demand,
            "gi_star_z": gi_z, "hotspot_code": hotspot_code, "moran_code": moran_code,
            "historical_hotspot_freq": hist_freq, "confidence": confidence,
            "dl_mbps": dl, "ul_mbps": ul, "rtt_ms": rtt, "jitter_ms": jitter, "loss_pct": loss,
            "next_hour_degraded_historical": next_risk,
            "priority_base_score": priority, "ookla_anchored": anchored,
        },
        "hotspot_code_labels": HOTSPOT_CODES,
        "moran_code_labels": MORAN_CODES,
    }))

    # --- City-wide time series (tiny, drives the trend charts) ---
    print("Building city_timeseries.json …")
    ts_rows = []
    for t in range(T):
        sl = slice(t * C, t * C + C)
        qs = [v for v in qoe[sl] if v is not None]
        ds = degr_class[sl]
        dps = [v for v in demand[sl] if v is not None]
        normal = sum(1 for v in ds if v == 0)
        degraded = sum(1 for v in ds if v == 1)
        severe = sum(1 for v in ds if v == 2)
        ts_rows.append({
            "timestamp": timestamps[t] // 1000,  # ms epoch, JS-friendly
            "avg_qoe": r(statistics.mean(qs), 2) if qs else None,
            "avg_demand_pressure": r(statistics.mean(dps), 2) if dps else None,
            "normal_count": normal, "degraded_count": degraded, "severe_count": severe,
        })
    (OUT / "city_timeseries.json").write_text(json.dumps(ts_rows))

    # --- Summary / provenance ---
    print("Building summary_stats.json …")
    synth_fracs = {}
    for key, col in [("tower", "tower_is_synthetic"), ("building", "building_is_synthetic"),
                      ("osm", "osm_is_synthetic"), ("population", "population_is_synthetic")]:
        vals = static[col]
        synth_fracs[key] = round(100 * sum(1 for v in vals if v) / len(vals), 2)

    summary = {
        "study_area": metadata.get("study_area"),
        "pipeline_version": metadata.get("pipeline_version"),
        "generated_at_utc": metadata.get("generated_at_utc"),
        "h3_resolution": metadata.get("h3_resolution"),
        "total_cells": C,
        "total_timestamps": T,
        "time_range": {
            "start": ts_rows[0]["timestamp"],
            "end": ts_rows[-1]["timestamp"],
        },
        "synthetic_pct_by_source": synth_fracs,
        "ookla_anchored_pct": round(
            100 * metadata["synthetic_fallback_summary"]["qoe_level_ookla_anchored"], 1
        ),
        "source_datasets": {
            k: v.get("present", False) for k, v in metadata.get("source_datasets", {}).items()
        },
        "qoe_thresholds": {
            "dl_degraded_mbps": 10.0, "rtt_degraded_ms": 100.0,
            "dl_severe_mbps": 5.0, "rtt_severe_ms": 150.0,
        },
        # Verbatim from config.QOE_INDEX_WEIGHTS / QOE_THRESHOLDS and the
        # compute_qoe_targets() formula in src/data_preprocessing/qoe_target.py,
        # so the dashboard's client-side "Balanced" application profile and
        # "what-if" capacity simulator reproduce the pipeline's REAL composite
        # index exactly (not an approximation) when nothing is perturbed.
        "qoe_formula": {
            "source": "src/data_preprocessing/qoe_target.py:compute_qoe_targets()",
            "weights_balanced": {"download": 0.6, "upload": 0.4, "rtt_penalty": 0.4, "loss_penalty": 0.4},
            "dl_log_saturation_mbps": 50.0,
            "ul_log_saturation_mbps": 20.0,
            "rtt_penalty_free_ms": 20.0,
            "rtt_penalty_scale_ms": 50.0,
            "loss_penalty_scale_pct": 3.0,
            "note": (
                "qoe_index = w_dl*min(100, log1p(dl)/log1p(50)*100) + "
                "w_ul*min(100, log1p(ul)/log1p(20)*100) - "
                "w_rtt*100*(1-exp(-max(0,rtt-20)/50)) - "
                "w_loss*100*(1-exp(-max(0,loss)/3)), clipped to [0,100]. "
                "This is a custom composite index inspired by Weber-Fechner "
                "perceptual scaling, NOT a validated ITU-T G.107 E-model — "
                "see qoe_target.py's own docstring."
            ),
        },
        # Demand-elasticity coefficients from build_dynamic_panel() in
        # data_downloader.py — the REAL relationship the simulator used
        # between city-wide demand_mult / per-cell load_factor and the
        # resulting dl/rtt. The dashboard's what-if capacity simulator uses
        # these same coefficients (see dashboard.js applyWhatIf()) rather
        # than inventing its own sensitivity — see that function's comments
        # for why it's a documented approximation, not an exact inverse.
        "demand_elasticity": {
            "source": "src/data_preprocessing/data_downloader.py:build_dynamic_panel()",
            "anchored_dl_sensitivity": 0.3, "anchored_rtt_sensitivity": 0.15,
            "unanchored_dl_sensitivity": 0.8, "unanchored_rtt_sensitivity": 0.9,
        },
        "disclaimer": (
            "The hourly QoE panel (cell_time_dataset) is a simulated diurnal-demand "
            "scenario built on top of 100% real static GIS features (towers, buildings, "
            "roads, POIs, population) — it is a software-integration and dashboard-demo "
            "fixture, not observed network measurements. "
            f"{round(100 * metadata['synthetic_fallback_summary']['qoe_level_ookla_anchored'], 1)}% "
            "of cells have their simulated values re-centred on a real quarterly Ookla "
            "average; the rest are simulated-only. See cell_static_qoe_baseline.parquet "
            "for the zero-synthetic-QoE real Ookla baseline."
        ),
    }
    (OUT / "summary_stats.json").write_text(json.dumps(summary, indent=2))

    print("Building quarterly observed and finals metadata assets …")
    build_quarterly_assets(grid)

    print("Done. Output files:")
    for f in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.geojson")):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")

    print("\nValidating against metadata.json row_counts …")
    checks = [
        ("grid_cells", len(grid["features"]), metadata["row_counts"]["grid_cells"]),
        ("static_rows", n_static, metadata["row_counts"]["static_rows"]),
        ("panel_rows", n_panel, metadata["row_counts"]["panel_rows"]),
        ("panel cells x timestamps", C * T, metadata["row_counts"]["panel_rows"]),
    ]
    all_ok = True
    for name, got, expected in checks:
        ok = got == expected
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {expected}")
    if not all_ok:
        print("  WARNING: one or more checks failed — the exported JSON may not "
              "fully represent the pipeline output. Do not present it as-is.")
    else:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
