from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. Find repository root
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    if (candidate / "data" / "outputs").exists():
        ROOT = candidate
        break

if ROOT is None:
    raise RuntimeError("Could not locate repository root.")

DATA_PATH = ROOT / "data" / "outputs" / "cell_quarter_dataset.parquet"

OUTPUT_DIR = ROOT / "outputs" / "finals_quarterly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load
# ============================================================

df = pd.read_parquet(DATA_PATH)

print("=" * 80)
print("GeoQoE Pulse Finals — Target & Temporal Split Check")
print("=" * 80)


# ============================================================
# 3. Verify REAL quarterly timeline
# ============================================================

quarters = (
    df[["quarter_index", "quarter_label"]]
    .drop_duplicates()
    .sort_values("quarter_index")
)

print("\n" + "=" * 80)
print("REAL QUARTERLY TIMELINE")
print("=" * 80)

print(quarters.to_string(index=False))
print(f"\nUnique real quarters: {len(quarters)}")


# ============================================================
# 4. Check what 'below area norm' actually means
# ============================================================

print("\n" + "=" * 80)
print("CHECK EXISTING UNDERPERFORMANCE LABEL")
print("=" * 80)

needed = [
    "dl_percentile_in_quarter",
    "is_below_area_norm",
    "ookla_dl_mbps",
    "ookla_ul_mbps",
    "ookla_rtt_ms",
]

for col in needed:
    if col not in df.columns:
        raise RuntimeError(f"Missing expected column: {col}")


# Test whether the existing label is simply bottom 25% download percentile
expected_lt = (
    df["dl_percentile_in_quarter"] < 0.25
).astype(int)

expected_le = (
    df["dl_percentile_in_quarter"] <= 0.25
).astype(int)

agree_lt = (
    expected_lt == df["is_below_area_norm"]
).mean()

agree_le = (
    expected_le == df["is_below_area_norm"]
).mean()

print(f"Agreement with DL percentile < 25% : {agree_lt:.4%}")
print(f"Agreement with DL percentile <=25% : {agree_le:.4%}")


# Compare actual network metrics between the two groups
group_summary = (
    df.groupby("is_below_area_norm")
    .agg(
        rows=("h3_index", "size"),
        median_dl_mbps=("ookla_dl_mbps", "median"),
        median_ul_mbps=("ookla_ul_mbps", "median"),
        median_rtt_ms=("ookla_rtt_ms", "median"),
        mean_dl_mbps=("ookla_dl_mbps", "mean"),
        mean_ul_mbps=("ookla_ul_mbps", "mean"),
        mean_rtt_ms=("ookla_rtt_ms", "mean"),
    )
)

print("\nNetwork metrics by current underperformance state:")
print(group_summary)


# ============================================================
# 5. Build OUR finals modelling population
#
# Only:
#   currently normal
#   +
#   real next-quarter label
# ============================================================

eligible = df[
    (df["target_next_q_valid"] == 1)
    & (df["is_below_area_norm"] == 0)
].copy()

eligible["onset_target"] = (
    eligible["target_is_below_area_norm_next_q"]
    .astype(int)
)

print("\n" + "=" * 80)
print("FINALS AI TASK")
print("=" * 80)

print("Current state: NORMAL")
print("Prediction: Will it become underperforming NEXT QUARTER?")

print(f"\nEligible rows: {len(eligible):,}")
print(f"Onset events: {eligible['onset_target'].sum():,}")
print(f"Onset rate:   {eligible['onset_target'].mean():.4%}")


# ============================================================
# 6. Show the onset rate for EACH REAL quarter
# ============================================================

quarter_stats = (
    eligible
    .groupby(["quarter_index", "quarter_label"])
    ["onset_target"]
    .agg(
        eligible_rows="count",
        onset_events="sum",
        onset_rate="mean",
    )
    .reset_index()
    .sort_values("quarter_index")
)

print("\n" + "=" * 80)
print("EVENTS BY REAL QUARTER")
print("=" * 80)

print(quarter_stats.to_string(index=False))


# ============================================================
# 7. Lock chronological train / validation / test periods
#
# Final test:
# latest predictor quarter with a real future label
#
# Validation:
# previous two predictor quarters
#
# Training:
# everything earlier
# ============================================================

usable_quarters = sorted(
    eligible["quarter_index"].unique()
)

if len(usable_quarters) < 4:
    raise RuntimeError("Not enough quarters for temporal split.")

test_quarter = usable_quarters[-1]

validation_quarters = usable_quarters[-3:-1]

train_quarters = [
    q for q in usable_quarters
    if q < validation_quarters[0]
]


def assign_split(q):
    if q == test_quarter:
        return "test"
    elif q in validation_quarters:
        return "validation"
    elif q in train_quarters:
        return "train"
    else:
        return "unused"


eligible["split"] = eligible["quarter_index"].apply(assign_split)


split_summary = (
    eligible
    .groupby("split")
    .agg(
        rows=("h3_index", "size"),
        onset_events=("onset_target", "sum"),
        onset_rate=("onset_target", "mean"),
        first_quarter=("quarter_label", "first"),
        last_quarter=("quarter_label", "last"),
    )
)

print("\n" + "=" * 80)
print("PROPOSED TEMPORAL SPLIT")
print("=" * 80)

print(f"Train quarter_index values:      {train_quarters}")
print(f"Validation quarter_index values: {validation_quarters}")
print(f"Final test quarter_index:        {test_quarter}")

print("\nSplit summary:")
print(split_summary)


# ============================================================
# 8. Print actual quarter labels for each split
# ============================================================

print("\nActual quarters:")

for split_name in ["train", "validation", "test"]:

    labels = (
        eligible.loc[
            eligible["split"] == split_name,
            ["quarter_index", "quarter_label"]
        ]
        .drop_duplicates()
        .sort_values("quarter_index")
    )

    print(f"\n{split_name.upper()}:")
    print(labels.to_string(index=False))


# ============================================================
# 9. Save audit outputs
# ============================================================

quarter_stats.to_csv(
    OUTPUT_DIR / "02_onset_by_real_quarter.csv",
    index=False
)

eligible[
    [
        "h3_index",
        "quarter_index",
        "quarter_label",
        "is_below_area_norm",
        "target_is_below_area_norm_next_q",
        "onset_target",
        "split",
    ]
].to_parquet(
    OUTPUT_DIR / "02_modelling_split_assignments.parquet",
    index=False
)

group_summary.to_csv(
    OUTPUT_DIR / "02_target_group_summary.csv"
)


print("\n" + "=" * 80)
print("CHECKPOINT 2 COMPLETE")
print("=" * 80)

print("\nNo model has been trained.")
print("The final test quarter has NOT been used for model selection.")