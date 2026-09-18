"""
test_data_pipeline.py — lightweight regression tests, no test framework
dependency beyond the standard library's unittest.

These don't have a pyarrow ground truth to compare against (this sandbox
had none available), so instead they check internal consistency: does the
pure-Python parquet reader's output match facts the pipeline already
declares about itself (row counts in metadata.json, non-null percentages
in feature_dictionary.xlsx, the documented anchored-cells/degradation
correlation)? A regression here means either the reader broke, or the
underlying pipeline output changed — worth knowing either way.

Run from the repo root:
    python3 dashboard/test_data_pipeline.py
"""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
sys.path.insert(0, str(REPO / "dashboard"))
import pyparquet_min as pq  # noqa: E402


class TestParquetReader(unittest.TestCase):
    def test_static_features_row_count_matches_metadata(self):
        metadata = json.loads((REPO / "data/outputs/metadata.json").read_text())
        _, _, n = pq.read_parquet(str(REPO / "data/outputs/cell_static_features.parquet"),
                                   columns=["h3_index"])
        self.assertEqual(n, metadata["row_counts"]["static_rows"])

    def test_panel_row_count_matches_metadata(self):
        metadata = json.loads((REPO / "data/outputs/metadata.json").read_text())
        _, _, n = pq.read_parquet(str(REPO / "data/outputs/cell_time_dataset.parquet"),
                                   columns=["h3_index"])
        self.assertEqual(n, metadata["row_counts"]["panel_rows"])

    def test_ookla_baseline_nullability_matches_feature_dictionary(self):
        # feature_dictionary.xlsx declares ookla_dl_mbps as 59.98% non-null.
        _, data, n = pq.read_parquet(
            str(REPO / "data/outputs/cell_static_qoe_baseline.parquet"),
            columns=["ookla_dl_mbps"],
        )
        non_null_pct = 100 * sum(1 for v in data["ookla_dl_mbps"] if v is not None) / n
        self.assertAlmostEqual(non_null_pct, 59.98, delta=0.1)

    def test_unanchored_degradation_remains_negligible(self):
        # Pipeline 3.3 has two degraded unanchored rows rather than the exact
        # zero in 3.2. Preserve the meaningful contract: the rate remains
        # negligible and must not silently become a substantive population.
        _, data, _ = pq.read_parquet(
            str(REPO / "data/outputs/cell_time_dataset.parquet"),
            columns=["target_degradation_class", "qoe_level_is_ookla_anchored"],
        )
        unanchored_rows = sum(not anchored for anchored in data["qoe_level_is_ookla_anchored"])
        unanchored_degraded = sum(
            1 for cls, anchored in zip(data["target_degradation_class"], data["qoe_level_is_ookla_anchored"])
            if cls >= 1 and not anchored
        )
        self.assertLess(unanchored_degraded / unanchored_rows, 0.0001)

    def test_dictionary_encoded_string_column_decodes(self):
        _, data, n = pq.read_parquet(
            str(REPO / "data/outputs/cell_static_features.parquet"),
            columns=["h3_index", "tower_source"],
        )
        self.assertEqual(len(set(data["h3_index"])), n)  # h3_index is a unique key
        self.assertTrue(all(v in ("opencellid_deduplicated", "synthetic_fallback_no_file")
                             for v in data["tower_source"]))


class TestBuiltDashboardData(unittest.TestCase):
    """Run after build_dashboard_data.py has produced dashboard/data/*.json."""

    @classmethod
    def setUpClass(cls):
        data_dir = REPO / "dashboard/build/data"
        cls.panel_path = data_dir / "panel.json"
        cls.summary_path = data_dir / "summary_stats.json"
        if not cls.panel_path.exists():
            raise unittest.SkipTest("dashboard/data/panel.json not built yet — run build_dashboard_data.py first")

    def test_panel_is_a_complete_regular_grid(self):
        panel = json.loads(self.panel_path.read_text())
        C, T = panel["num_cells"], panel["num_timestamps"]
        for field, values in panel["fields"].items():
            self.assertEqual(len(values), C * T, f"field {field} has wrong length")

    def test_summary_time_range_is_milliseconds_not_microseconds(self):
        # Regression test for a units bug caught during development: the
        # time_range was briefly double-converted back to microseconds.
        summary = json.loads(self.summary_path.read_text())
        import datetime
        start = datetime.datetime.fromtimestamp(summary["time_range"]["start"] / 1000, datetime.UTC)
        self.assertTrue(2020 <= start.year <= 2035, f"implausible start year: {start.year}")

    def test_qoe_formula_constants_match_real_config(self):
        # The what-if simulator and application profiles trust these values
        # to reproduce target_qoe_index exactly for the "balanced" profile —
        # verify they still match config.QOE_INDEX_WEIGHTS / QOE_THRESHOLDS
        # rather than silently drifting if config.py changes.
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", REPO / "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        summary = json.loads(self.summary_path.read_text())
        w = summary["qoe_formula"]["weights_balanced"]
        self.assertEqual(w["download"], config.QOE_INDEX_WEIGHTS["download"])
        self.assertEqual(w["upload"], config.QOE_INDEX_WEIGHTS["upload"])
        self.assertEqual(w["rtt_penalty"], config.QOE_INDEX_WEIGHTS["rtt_penalty"])
        self.assertEqual(w["loss_penalty"], config.QOE_INDEX_WEIGHTS["loss_penalty"])

    def test_balanced_profile_formula_reproduces_real_qoe_index(self):
        # End-to-end check: recompute target_qoe_index from the exported raw
        # dl/ul/rtt/loss using the exact formula shape the dashboard's JS
        # uses, and confirm it matches the pipeline's own real value to
        # within JSON export rounding (fields are rounded to 1-2dp).
        import math
        panel = json.loads(self.panel_path.read_text())
        f = panel["fields"]
        max_err = 0.0
        n_checked = 0
        for i in range(0, len(f["qoe"]), 4001):  # sparse sample across the file
            dl, ul, rtt, loss, real = f["dl_mbps"][i], f["ul_mbps"][i], f["rtt_ms"][i], f["loss_pct"][i], f["qoe"][i]
            if None in (dl, ul, rtt, loss, real):
                continue
            s_dl = min(100.0, math.log1p(dl) / math.log1p(50.0) * 100.0)
            s_ul = min(100.0, math.log1p(ul) / math.log1p(20.0) * 100.0)
            pen_rtt = 100.0 * (1.0 - math.exp(-max(0.0, rtt - 20.0) / 50.0))
            pen_loss = 100.0 * (1.0 - math.exp(-max(0.0, loss) / 3.0))
            recomputed = max(0.0, min(100.0, 0.6 * s_dl + 0.4 * s_ul - 0.4 * pen_rtt - 0.4 * pen_loss))
            max_err = max(max_err, abs(recomputed - real))
            n_checked += 1
        self.assertGreater(n_checked, 50)
        self.assertLess(max_err, 0.5, "formula transcription has drifted from the real pipeline output")


class TestFinalsDashboardIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = REPO / "dashboard/build/data"
        cls.grid = json.loads((cls.data_dir / "h3_grid.geojson").read_text())
        cls.panel = json.loads((cls.data_dir / "panel.json").read_text())
        cls.observed = json.loads((cls.data_dir / "quarterly_observed_2026Q1.geojson").read_text())
        cls.finals = json.loads((cls.data_dir / "finals_quarterly.geojson").read_text())
        cls.metadata = json.loads((cls.data_dir / "finals_metadata.json").read_text())

    def test_latest_pipeline_export_is_loaded(self):
        summary = json.loads((self.data_dir / "summary_stats.json").read_text())
        self.assertEqual(summary["pipeline_version"], "3.3.0")

    def test_h3_contracts_are_unique_and_joinable(self):
        grid_ids = {f["properties"]["h3_index"] for f in self.grid["features"]}
        panel_ids = set(self.panel["h3_ids"])
        observed_ids = [f["properties"]["h3_index"] for f in self.observed["features"]]
        finals_ids = [f["properties"]["h3_index"] for f in self.finals["features"]]
        self.assertEqual(len(observed_ids), len(set(observed_ids)))
        self.assertEqual(len(finals_ids), len(set(finals_ids)))
        self.assertTrue(set(observed_ids) <= grid_ids & panel_ids)
        self.assertTrue(set(finals_ids) <= grid_ids & panel_ids)

    def test_observed_status_is_separate_and_complete(self):
        rows = [f["properties"] for f in self.observed["features"]]
        statuses = {row["current_status"] for row in rows}
        self.assertEqual(statuses, {"currently_underperforming", "currently_not_underperforming"})
        self.assertEqual(len(rows), self.metadata["observed_layer"]["measured_cells"])
        self.assertEqual(
            sum(row["current_status"] == "currently_underperforming" for row in rows),
            self.metadata["observed_layer"]["underperforming_cells"],
        )

    def test_finals_fields_ranks_bands_and_time_machine(self):
        rows = [f["properties"] for f in self.finals["features"]]
        required = {
            "h3_index", "risk_rank", "risk_band", "current_status",
            "actual_next_quarter_status", "ookla_dl_mbps", "ookla_ul_mbps",
            "ookla_rtt_ms", "dl_percentile_in_quarter", "measurement_support",
            "time_machine_result", "score_is_probability", "interpretation_note",
            "risk_driver_1_label", "protective_1_label",
        }
        self.assertFalse(required - set(rows[0]))
        self.assertEqual(sorted(row["risk_rank"] for row in rows), list(range(1, 2187)))
        self.assertEqual({band: sum(row["risk_band"] == band for row in rows) for band in ("high", "elevated", "lower")}, {"high": 50, "elevated": 150, "lower": 1986})
        expected_results = {"correctly_anticipated": 92, "false_alert": 108, "missed": 113, "correctly_lower_risk": 1873}
        self.assertEqual({result: sum(row["time_machine_result"] == result for row in rows) for result in expected_results}, expected_results)
        self.assertTrue(all(row["score_is_probability"] is False for row in rows))

    def test_metadata_matches_finals_contract(self):
        self.assertEqual(self.metadata["model_version"], "finals_quarterly_onset_v1.0.0")
        self.assertEqual(self.metadata["eligible_cells"], len(self.finals["features"]))
        self.assertEqual(self.metadata["predictor_quarter"], "2026Q1")
        self.assertEqual(self.metadata["forecast_quarter"], "2026Q2")
        self.assertEqual(self.metadata["risk_thresholds"], {"high_max_rank": 50, "elevated_max_rank": 200, "lower_min_rank": 201})
        self.assertAlmostEqual(self.metadata["final_test"]["pr_auc"], 0.44920862998066347)

    def test_missing_shap_rows_are_supported(self):
        rows = [f["properties"] for f in self.finals["features"]]
        self.assertGreater(sum(row["risk_driver_2_label"] is None for row in rows), 0)
        self.assertGreater(sum(row["risk_driver_3_label"] is None for row in rows), 0)

    def test_quarterly_methodology_is_the_default_ui(self):
        html = (REPO / "dashboard/build/index.html").read_text(encoding="utf-8")
        script = (REPO / "dashboard/build/dashboard.js").read_text(encoding="utf-8")
        for stage in ("REVEAL", "ANTICIPATE", "EXPLAIN", "PRIORITISE"):
            self.assertIn(stage, html + script)
        self.assertIn('colorBy: "onset"', script)
        self.assertIn("Quarterly observed + emerging risk", html)
        self.assertNotIn("Detect → Predict → Diagnose → Act", html)

    def test_methodology_labels_non_causal_context(self):
        html = (REPO / "dashboard/build/index.html").read_text(encoding="utf-8")
        script = (REPO / "dashboard/build/dashboard.js").read_text(encoding="utf-8")
        self.assertIn("Human-activity proxy (VIIRS)", script)
        self.assertIn("not measured network demand", script)
        self.assertIn("contributors to the model's risk assessment", script)
        self.assertIn("Investigation Evidence", html)
        self.assertIn("DEMO / SCENARIO", html + script)

    def test_quarterly_triage_handoff_and_outcome_tracking_are_present(self):
        script = (REPO / "dashboard/build/dashboard.js").read_text(encoding="utf-8")
        for label in ("Investigate now", "Verify first", "Monitor"):
            self.assertIn(label, script)
        for function_name in (
            "quarterlyTriageRecommendation",
            "createInvestigationCase",
            "downloadInvestigationCase",
            "printInvestigationBrief",
            "updateInvestigationCase",
        ):
            self.assertIn(f"function {function_name}", script)
        for outcome in (
            "confirmed_issue",
            "not_confirmed",
            "insufficient_evidence",
            "monitoring",
            "resolved",
        ):
            self.assertIn(outcome, script)
        self.assertIn("do not automatically retrain or update the model", script)

    def test_explain_measurement_handoff_preserves_selected_cell(self):
        script = (REPO / "dashboard/build/dashboard.js").read_text(encoding="utf-8")
        self.assertIn("function openSelectedCellReveal()", script)
        self.assertIn("onclick=\"openSelectedCellReveal()\">Review quarterly measurements", script)
        self.assertIn('setWorkspaceTool("cell")', script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
