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

REPO = Path(__file__).resolve().parent.parent
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

    def test_no_unanchored_cell_ever_degrades(self):
        # config.py explicitly documents this near-perfect correlation on the
        # simulated panel; if this stops holding, either the reader or the
        # pipeline's generator changed.
        _, data, _ = pq.read_parquet(
            str(REPO / "data/outputs/cell_time_dataset.parquet"),
            columns=["target_degradation_class", "qoe_level_is_ookla_anchored"],
        )
        unanchored_degraded = sum(
            1 for cls, anchored in zip(data["target_degradation_class"], data["qoe_level_is_ookla_anchored"])
            if cls >= 1 and not anchored
        )
        self.assertEqual(unanchored_degraded, 0)

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
        data_dir = REPO / "dashboard/data"
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
        start = datetime.datetime.utcfromtimestamp(summary["time_range"]["start"] / 1000)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
