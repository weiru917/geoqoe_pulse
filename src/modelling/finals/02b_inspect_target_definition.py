from pathlib import Path
import pandas as pd


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


print("=" * 80)
print("GeoQoE Pulse Finals — Exact Target Definition Audit")
print("=" * 80)


# ============================================================
# 2. Read feature dictionary
# ============================================================

DICT_PATH = ROOT / "data" / "outputs" / "feature_dictionary.xlsx"

if DICT_PATH.exists():

    dictionary = pd.read_excel(DICT_PATH)

    print("\n" + "=" * 80)
    print("FEATURE DICTIONARY ENTRIES")
    print("=" * 80)

    target_terms = [
        "is_below_area_norm",
        "dl_percentile_in_quarter",
        "target_is_below_area_norm_next_q",
        "is_degraded",
    ]

    for term in target_terms:

        matches = dictionary[
            dictionary.astype(str)
            .apply(
                lambda row: row.str.contains(
                    term,
                    case=False,
                    regex=False
                ).any(),
                axis=1
            )
        ]

        print(f"\n--- {term} ---")

        if len(matches) == 0:
            print("No dictionary entry found.")
        else:
            print(matches.to_string(index=False))

else:
    print("\nFeature dictionary not found.")


# ============================================================
# 3. Search source code for exact target creation
# ============================================================

print("\n" + "=" * 80)
print("SOURCE-CODE REFERENCES")
print("=" * 80)

SEARCH_TERMS = [
    "is_below_area_norm",
    "dl_percentile_in_quarter",
    "target_is_below_area_norm_next_q",
]

extensions = {".py", ".md"}

found_any = False

for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    if path.suffix.lower() not in extensions:
        continue

    # Skip virtual environment and Git internals
    if ".venv" in path.parts or ".git" in path.parts:
        continue

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    except Exception:
        continue

    relevant = []

    for line_no, line in enumerate(lines, start=1):

        if any(
            term.lower() in line.lower()
            for term in SEARCH_TERMS
        ):
            relevant.append((line_no, line))

    if relevant:

        found_any = True

        print("\nFILE:")
        print(path.relative_to(ROOT))

        for line_no, line in relevant:
            print(f"  Line {line_no}: {line.strip()}")


if not found_any:
    print("No source-code references found.")


# ============================================================
# 4. Inspect actual target prevalence per real quarter
# ============================================================

DATA_PATH = ROOT / "data" / "outputs" / "cell_quarter_dataset.parquet"

df = pd.read_parquet(DATA_PATH)

summary = (
    df.groupby(["quarter_index", "quarter_label"])
    ["is_below_area_norm"]
    .agg(
        rows="count",
        underperforming="sum",
        rate="mean",
    )
    .reset_index()
    .sort_values("quarter_index")
)

print("\n" + "=" * 80)
print("CURRENT TARGET RATE BY QUARTER")
print("=" * 80)

print(summary.to_string(index=False))


print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)