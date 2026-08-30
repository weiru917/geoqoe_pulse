"""
DEPRECATED — use `python run_pipeline.py` instead.

This script used to run run_pipeline.main() and then bolt the real
VIIRS / WorldCover / Sunway layers onto cell_static_features.parquet
afterwards. That ordering was the cause of a significant handoff bug: by
the time the merge ran, run_pipeline had already built and exported
cell_static_qoe_baseline.parquet (the recommended modelling dataset), the
hourly panel, and feature_dictionary.xlsx. The 24 real columns therefore
landed in exactly one file and reached no consumer — not the model, not
the dashboard, not the dictionary the README tells the modelling team to
query.

All of that work now happens INSIDE run_pipeline.main(), before the
baseline is built:

    run_pipeline.merge_static_new_layers()      -> VIIRS + WorldCover
    run_pipeline.merge_sunway_inset()           -> Sunway field inset
    run_pipeline.summarize_sunway_ground_truth()-> ground-truth coverage
                                                   (replaces the withdrawn
                                                    Ookla-vs-Sunway
                                                    correlation)

This file is kept only so an existing command or note that references it
still works. It forwards to the single entry point and warns.

Prerequisites are unchanged: the raw files in data/raw/, plus the three
extraction steps that produce the derived layers —

    python scripts/extract/viirs.py
    python scripts/extract/worldcover.py
    python scripts/extract/merge_layers.py
    python -m src.data_preprocessing.ingest_sunway_real
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_full_data_pipeline")


def main():
    logger.warning(
        "run_full_data_pipeline.py is DEPRECATED. The VIIRS/WorldCover/Sunway merges now run "
        "inside run_pipeline.main() BEFORE the static QoE baseline is built, so those layers "
        "reach every output instead of only cell_static_features.parquet. Forwarding to "
        "run_pipeline.main() — call `python run_pipeline.py` directly in future."
    )
    return run_pipeline.main()


if __name__ == "__main__":
    main()
