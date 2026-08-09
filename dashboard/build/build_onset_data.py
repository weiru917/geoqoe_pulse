#!/usr/bin/env python3
"""
build_onset_data.py
--------------------
Converts the frozen GeoQoE_AI competition package (the "Trained AI Model"
deliverable — a real, already-trained XGBoost onset model plus a real
static-QoE ensemble, both evaluated on real held-out data as described in
ai_model/metadata/geoqoe_model_card.json) into two small JSON files the
dashboard fetches directly:

  data/onset_model.json  — per-cell AI outputs (static estimate + onset
                            risk), reindexed to the SAME cell order as
                            panel.json (state.h3IndexOf) so the two data
                            sets can be joined by cell index with zero
                            extra lookups in dashboard.js.
  data/model_card.json   — the frozen model's headline metrics and stated
                            limitations, for the "AI Model Info" modal.

This is a single, fixed snapshot (2026-06-30 23:00:00 — the last hour of
the existing 168-hour panel), NOT a per-hour time series: the onset model
was only run once, against the latest hour, per
ai_model/metadata/geoqoe_model_card.json -> dashboard_snapshot. The
dashboard therefore treats this layer as time-slider-independent (like the
tower layer), not as another frame of the animated panel.

Run from the dashboard/ directory:
    python3 build_onset_data.py
Requires: pandas (no pyarrow needed — reads the CSV twin of the frozen
parquet file, which ships alongside it in ai_model/dashboard/).
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
AI_DIR = ROOT / "ai_model"
DATA_DIR = ROOT / "data"

ONSET_CSV = AI_DIR / "dashboard" / "latest_hour_onset_risk.csv"
MODEL_CARD_SRC = AI_DIR / "metadata" / "geoqoe_model_card.json"
MANIFEST_SRC = AI_DIR / "package_manifest.json"
PANEL_JSON = DATA_DIR / "panel.json"

OUT_ONSET = DATA_DIR / "onset_model.json"
OUT_MODEL_CARD = DATA_DIR / "model_card.json"

# Columns copied through verbatim from the frozen package, keyed by
# h3_index, reindexed to panel.json's cell order. Kept columnar (one array
# per field, same convention panel.json already uses) rather than
# one-object-per-cell — meaningfully smaller over the wire for 4,918 cells.
FIELDS = [
    "static_data_status", "display_dl_mbps", "ookla_dl_mbps",
    "ai_ensemble_estimate_mbps", "ensemble_std_mbps",
    "ensemble_disagreement_level", "feature_space_support_level",
    "relative_prediction_reliability",
    "eligible_for_onset_scoring", "onset_operational_status",
    "onset_priority_tier", "next_hour_onset_risk_score", "onset_risk_rank",
    "onset_risk_percentile", "alert_at_frozen_threshold",
    "top_20_onset_risk", "top_50_onset_risk", "top_100_onset_risk",
    "top_1_percent_onset_risk", "top_5_percent_onset_risk",
    "target_is_degraded",
]


def clean(v):
    """NaN/NaT -> None so json.dumps doesn't emit invalid `NaN` literals."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (bool,)):
        return bool(v)
    return v


def main():
    for p in (ONSET_CSV, MODEL_CARD_SRC, PANEL_JSON):
        if not p.exists():
            sys.exit(f"Missing required input: {p}. Run this from the dashboard/ directory "
                      f"with ai_model/ (the unzipped competition_package) sitting alongside data/.")

    panel = json.loads(PANEL_JSON.read_text())
    h3_order = panel["h3_ids"]

    df = pd.read_csv(ONSET_CSV)
    missing_fields = [f for f in FIELDS if f not in df.columns]
    if missing_fields:
        sys.exit(f"latest_hour_onset_risk.csv is missing expected column(s): {missing_fields}")

    timestamps = df["timestamp"].unique().tolist()
    if len(timestamps) != 1:
        sys.exit(f"Expected a single-hour onset snapshot, found {len(timestamps)}: {timestamps}")
    snapshot_timestamp = timestamps[0]

    df = df.set_index("h3_index")
    missing_cells = [h for h in h3_order if h not in df.index]
    if missing_cells:
        sys.exit(f"{len(missing_cells)} panel.json cells have no row in latest_hour_onset_risk.csv "
                  f"(e.g. {missing_cells[:3]}) — panel.json and the AI package must come from the "
                  f"same pipeline run.")
    df = df.loc[h3_order]  # reindex to panel.json's cell order

    fields_out = {f: [clean(v) for v in df[f].tolist()] for f in FIELDS}

    model_card = json.loads(MODEL_CARD_SRC.read_text())
    onset_model = model_card["hourly_onset_model"]
    manifest = json.loads(MANIFEST_SRC.read_text()) if MANIFEST_SRC.exists() else {}

    out = {
        "snapshot_timestamp": snapshot_timestamp,
        "num_cells": len(h3_order),
        "h3_ids": h3_order,  # same order/length as panel.h3_ids, included for standalone use
        "model_name": onset_model["selected_model"],
        "model_best_iteration": onset_model["frozen_best_iteration"],
        "model_frozen_threshold": onset_model["frozen_validation_threshold"],
        "static_observed_cells": model_card["static_real_data_model"]["observed_cells"],
        "static_ai_estimated_cells": model_card["static_real_data_model"]["ai_estimated_cells"],
        "dashboard_snapshot": model_card["dashboard_snapshot"],
        "fields": fields_out,
    }
    DATA_DIR.mkdir(exist_ok=True)
    OUT_ONSET.write_text(json.dumps(out))
    print(f"Wrote {OUT_ONSET} ({OUT_ONSET.stat().st_size / 1024:.0f} KB, "
          f"{len(h3_order)} cells x {len(FIELDS)} fields, snapshot {snapshot_timestamp})")

    # ---- model_card.json: trimmed summary for the AI Model Info modal ----
    static_m = model_card["static_real_data_model"]
    onset_val = onset_model["validation_result"]["threshold_metrics"]
    onset_test = onset_model["final_test_result"]["test_threshold_metrics"]
    onset_val_ranking = onset_model["validation_result"]["ranking_metrics"]
    onset_test_ranking = onset_model["final_test_result"]["test_ranking_metrics"]

    card_out = {
        "project": model_card.get("project"),
        "package_status": model_card.get("package_status"),
        "recommended_pitch_wording": model_card.get("recommended_pitch_wording"),
        "limitations": model_card.get("limitations", []),
        "static_model": {
            "selected_model": static_m["selected_model"],
            "purpose": static_m["purpose"],
            "observed_cells": static_m["observed_cells"],
            "ai_estimated_cells": static_m["ai_estimated_cells"],
            "spatial_oof_mae_mbps": static_m["honest_spatial_oof_metrics"]["mae_mbps"],
            "spatial_oof_rmse_mbps": static_m["honest_spatial_oof_metrics"]["rmse_mbps"],
            "spatial_oof_r2": static_m["honest_spatial_oof_metrics"]["r2"],
            "usage": static_m["usage"],
        },
        "onset_model": {
            "selected_model": onset_model["selected_model"],
            "purpose": onset_model["purpose"],
            "feature_count": onset_model["feature_count"],
            "frozen_best_iteration": onset_model["frozen_best_iteration"],
            "frozen_validation_threshold": onset_model["frozen_validation_threshold"],
            "selection_basis": onset_model["selection_basis"],
            "validation": {
                "pr_auc": onset_val_ranking["pr_auc"],
                "roc_auc": onset_val_ranking["roc_auc"],
                "precision": onset_val["precision"],
                "recall": onset_val["recall"],
                "f1": onset_val["f1"],
                "true_positives": onset_val["true_positives"],
                "false_positives": onset_val["false_positives"],
                "true_negatives": onset_val["true_negatives"],
                "false_negatives": onset_val["false_negatives"],
                "predicted_positive_rows": onset_val["predicted_positive_rows"],
            },
            "final_test": {
                "pr_auc": onset_test_ranking["pr_auc"],
                "roc_auc": onset_test_ranking["roc_auc"],
                "precision": onset_test["precision"],
                "recall": onset_test["recall"],
                "f1": onset_test["f1"],
                "true_positives": onset_test["true_positives"],
                "false_positives": onset_test["false_positives"],
                "true_negatives": onset_test["true_negatives"],
                "false_negatives": onset_test["false_negatives"],
                "predicted_positive_rows": onset_test["predicted_positive_rows"],
            },
            # Real alert-volume/precision/recall trade-off at different cutoffs,
            # straight from the frozen final-test evaluation (top_20/50/100 per
            # hour, top_1%/5% per hour) — used by the Forecast Validation panel's
            # "what happens at a different alert budget" table. Not simulated
            # for the dashboard; this is the model card's own reported sweep.
            "final_test_top_k": onset_model.get("final_test_top_k", []),
            "important_observation": onset_model.get("important_observation"),
            "score_interpretation": onset_model.get("score_interpretation"),
        },
        "dashboard_snapshot": model_card.get("dashboard_snapshot"),
        "package_manifest_files": [f["path"] for f in manifest.get("files", [])],
    }
    OUT_MODEL_CARD.write_text(json.dumps(card_out, indent=2))
    print(f"Wrote {OUT_MODEL_CARD}")


if __name__ == "__main__":
    main()
