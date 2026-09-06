from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. Find repository root automatically
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    if (candidate / "data" / "outputs").exists():
        ROOT = candidate
        break

if ROOT is None:
    raise RuntimeError(
        "Could not find repository root containing data/outputs/"
    )

DATA_PATH = ROOT / "data" / "outputs" / "cell_quarter_dataset.parquet"

OUTPUT_DIR = ROOT / "outputs" / "finals_quarterly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("GeoQoE Pulse Finals — Quarterly Dataset Audit")
print("=" * 80)
print(f"\nRepository root:\n{ROOT}")
print(f"\nDataset:\n{DATA_PATH}")


# ============================================================
# 2. Load dataset
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{DATA_PATH}\n"
        "Check whether your teammate's latest data pipeline has been pulled."
    )

df = pd.read_parquet(DATA_PATH)

print("\n" + "=" * 80)
print("BASIC DATASET INFORMATION")
print("=" * 80)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns):,}")

if "h3_index" in df.columns:
    print(f"Unique H3 cells: {df['h3_index'].nunique():,}")


# ============================================================
# 3. Find quarter/time column
# ============================================================

possible_time_cols = [
    "quarter",
    "year_quarter",
    "period",
    "timestamp",
    "date",
]

time_col = None

for col in possible_time_cols:
    if col in df.columns:
        time_col = col
        break

print("\n" + "=" * 80)
print("TIME COVERAGE")
print("=" * 80)

if time_col is None:
    print("No obvious quarter/time column was automatically detected.")
else:
    print(f"Detected time column: {time_col}")

    unique_times = (
        df[time_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
    )

    print(f"Number of unique periods: {len(unique_times)}")

    print("\nPeriods:")
    for x in unique_times:
        print(f"  {x}")


# ============================================================
# 4. Print every column
# ============================================================

print("\n" + "=" * 80)
print("COLUMN LIST")
print("=" * 80)

for i, col in enumerate(df.columns, start=1):
    print(
        f"{i:3d}. {col:50s} "
        f"{str(df[col].dtype):15s} "
        f"missing={df[col].isna().mean() * 100:6.2f}%"
    )


# ============================================================
# 5. Search for target / future-looking columns
# ============================================================

keywords = [
    "target",
    "next",
    "future",
    "below",
    "norm",
    "qoe",
    "degrad",
    "risk",
    "dl_",
    "rtt",
]

interesting_cols = [
    col for col in df.columns
    if any(k in col.lower() for k in keywords)
]

print("\n" + "=" * 80)
print("TARGET / QoE / FUTURE-RELATED COLUMNS")
print("=" * 80)

for col in interesting_cols:
    print(col)


# ============================================================
# 6. Inspect planned finals target if it already exists
# ============================================================

current_candidates = [
    "target_is_below_area_norm",
    "is_below_area_norm",
]

future_candidates = [
    "target_is_below_area_norm_next_q",
    "is_below_area_norm_next_q",
]

current_col = next(
    (c for c in current_candidates if c in df.columns),
    None
)

future_col = next(
    (c for c in future_candidates if c in df.columns),
    None
)

print("\n" + "=" * 80)
print("PROPOSED EMERGING-PROBLEM TARGET")
print("=" * 80)

print(f"Current-status column: {current_col}")
print(f"Next-quarter column:   {future_col}")


if current_col is not None and future_col is not None:

    usable = df[
        df[current_col].notna()
        & df[future_col].notna()
    ].copy()

    print(f"\nRows with both current and future labels: {len(usable):,}")

    print("\nCurrent status counts:")
    print(usable[current_col].value_counts(dropna=False))

    print("\nNext-quarter status counts:")
    print(usable[future_col].value_counts(dropna=False))

    # --------------------------------------------------------
    # Finals task:
    # CURRENTLY NORMAL -> UNDERPERFORMING NEXT QUARTER
    # --------------------------------------------------------

    currently_normal = usable[
        usable[current_col] == 0
    ].copy()

    onset_positive = (
        currently_normal[future_col] == 1
    )

    n_normal = len(currently_normal)
    n_onset = int(onset_positive.sum())

    onset_rate = (
        n_onset / n_normal
        if n_normal > 0
        else np.nan
    )

    print("\n" + "-" * 80)
    print("FINALS HEADLINE TASK")
    print("Currently normal -> underperforming next quarter")
    print("-" * 80)

    print(f"Eligible currently-normal rows: {n_normal:,}")
    print(f"New underperformance events:    {n_onset:,}")
    print(f"Event rate:                      {onset_rate:.4%}")

    # --------------------------------------------------------
    # Transition counts
    # --------------------------------------------------------

    transition_table = pd.crosstab(
        usable[current_col],
        usable[future_col],
        rownames=["Current quarter"],
        colnames=["Next quarter"],
    )

    print("\nTransition table:")
    print(transition_table)

    transition_table.to_csv(
        OUTPUT_DIR / "01_transition_table.csv"
    )

    # --------------------------------------------------------
    # Event rate by quarter
    # --------------------------------------------------------

    if time_col is not None:

        quarter_summary = (
            currently_normal
            .groupby(time_col)[future_col]
            .agg(
                eligible_rows="count",
                onset_events="sum",
                onset_rate="mean",
            )
            .reset_index()
        )

        print("\nEmerging-problem events by quarter:")
        print(quarter_summary.to_string(index=False))

        quarter_summary.to_csv(
            OUTPUT_DIR / "01_onset_by_quarter.csv",
            index=False
        )

else:
    print(
        "\nThe expected target columns were not found.\n"
        "That is NOT a failure — we will inspect the available columns "
        "and build the target properly in Checkpoint 2."
    )


# ============================================================
# 7. Flag obvious leakage columns
# ============================================================

leakage_keywords = [
    "next_q",
    "nextquarter",
    "future",
    "target",
]

potential_leakage = [
    col for col in df.columns
    if any(k in col.lower() for k in leakage_keywords)
]

print("\n" + "=" * 80)
print("POTENTIAL FUTURE / TARGET LEAKAGE COLUMNS")
print("=" * 80)

for col in potential_leakage:
    print(f"  {col}")


# ============================================================
# 8. Save full column audit
# ============================================================

audit = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_rows": [int(df[c].isna().sum()) for c in df.columns],
    "missing_pct": [
        float(df[c].isna().mean() * 100)
        for c in df.columns
    ],
    "unique_values": [
        int(df[c].nunique(dropna=True))
        for c in df.columns
    ],
})

audit.to_csv(
    OUTPUT_DIR / "01_column_audit.csv",
    index=False
)


print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print("\nSaved:")
print(OUTPUT_DIR / "01_column_audit.csv")

if current_col is not None and future_col is not None:
    print(OUTPUT_DIR / "01_transition_table.csv")

    if time_col is not None:
        print(OUTPUT_DIR / "01_onset_by_quarter.csv")