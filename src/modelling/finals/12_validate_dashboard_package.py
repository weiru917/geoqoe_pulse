from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. Locate repository
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None

for candidate in [
    SCRIPT_PATH.parent,
    *SCRIPT_PATH.parents,
]:
    if (
        candidate
        / "outputs"
        / "finals_dashboard"
    ).exists():

        ROOT = candidate
        break


if ROOT is None:
    raise RuntimeError(
        "Could not locate repository root."
    )


DASHBOARD_DIR = (
    ROOT
    / "outputs"
    / "finals_dashboard"
)


CSV_PATH = (
    DASHBOARD_DIR
    / "2026Q1_to_2026Q2_cells.csv"
)

PARQUET_PATH = (
    DASHBOARD_DIR
    / "2026Q1_to_2026Q2_cells.parquet"
)

GEOJSON_PATH = (
    DASHBOARD_DIR
    / "2026Q1_to_2026Q2_cells.geojson"
)

METADATA_PATH = (
    DASHBOARD_DIR
    / "dashboard_metadata.json"
)


# ============================================================
# 2. File existence
# ============================================================

print("=" * 80)
print(
    "GeoQoE Pulse Finals — "
    "Dashboard Package QA"
)
print("=" * 80)


required_files = [
    CSV_PATH,
    PARQUET_PATH,
    GEOJSON_PATH,
    METADATA_PATH,
]


for path in required_files:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    print(
        f"FOUND: {path.name}"
    )


# ============================================================
# 3. Load package
# ============================================================

csv_df = pd.read_csv(
    CSV_PATH
)

parquet_df = pd.read_parquet(
    PARQUET_PATH
)


with open(
    GEOJSON_PATH,
    "r",
    encoding="utf-8",
) as f:

    geojson = json.load(f)


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


# ============================================================
# 4. Basic row-count consistency
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "ROW COUNTS"
)

print("=" * 80)


print(
    f"CSV rows:       "
    f"{len(csv_df):,}"
)

print(
    f"Parquet rows:   "
    f"{len(parquet_df):,}"
)

print(
    f"GeoJSON cells:  "
    f"{len(geojson['features']):,}"
)


if not (
    len(csv_df)
    == len(parquet_df)
    == len(geojson["features"])
):

    raise RuntimeError(
        "Row counts do not match."
    )


# ============================================================
# 5. Required fields
# ============================================================

REQUIRED_COLUMNS = [

    "h3_index",

    "risk_rank",
    "risk_band",
    "risk_band_label",
    "risk_percentile",

    "onset_target",
    "actual_next_quarter_status",
    "time_machine_result",

    "ookla_dl_mbps",
    "ookla_ul_mbps",
    "ookla_rtt_ms",
    "ookla_tests",

    "measurement_support",

    "risk_driver_1_label",
    "risk_driver_1_value",

    "risk_driver_2_label",
    "risk_driver_2_value",

    "risk_driver_3_label",
    "risk_driver_3_value",
]


missing_columns = [

    col
    for col
    in REQUIRED_COLUMNS

    if col
    not in parquet_df.columns
]


if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + str(
            missing_columns
        )
    )


print(
    "\nAll required dashboard "
    "columns are present."
)


# ============================================================
# 6. H3 uniqueness
# ============================================================

duplicate_h3 = int(

    parquet_df[
        "h3_index"
    ]
    .duplicated()
    .sum()
)


print(
    f"\nDuplicate H3 cells: "
    f"{duplicate_h3}"
)


if duplicate_h3 != 0:

    raise RuntimeError(
        "Duplicate H3 IDs found."
    )


# ============================================================
# 7. Risk-rank integrity
# ============================================================

n_cells = len(
    parquet_df
)


ranks = sorted(

    parquet_df[
        "risk_rank"
    ]
    .astype(int)
    .tolist()
)


expected_ranks = list(
    range(
        1,
        n_cells + 1,
    )
)


if ranks != expected_ranks:

    raise RuntimeError(
        "Risk ranks are not a complete "
        "1..N sequence."
    )


print(
    f"Risk ranks valid: "
    f"1 to {n_cells:,}"
)


# ============================================================
# 8. Frozen risk-band policy
# ============================================================

def expected_band(rank):

    if rank <= 50:
        return "high"

    if rank <= 200:
        return "elevated"

    return "lower"


expected = (

    parquet_df[
        "risk_rank"
    ]

    .apply(
        expected_band
    )
)


band_mismatches = int(

    (
        expected
        != parquet_df[
            "risk_band"
        ]
    )
    .sum()
)


if band_mismatches != 0:

    raise RuntimeError(
        f"Risk-band mismatches: "
        f"{band_mismatches}"
    )


print(
    "\nRisk-band policy valid."
)


print(

    parquet_df[
        "risk_band"
    ]

    .value_counts()

    .to_string()
)


# ============================================================
# 9. Time Machine logic
# ============================================================

def expected_result(row):

    flagged = (
        row[
            "risk_rank"
        ]
        <= 200
    )

    actual = (
        row[
            "onset_target"
        ]
        == 1
    )

    if flagged and actual:
        return (
            "correctly_anticipated"
        )

    if flagged and not actual:
        return (
            "false_alert"
        )

    if (
        not flagged
        and actual
    ):
        return "missed"

    return (
        "correctly_lower_risk"
    )


expected_results = (

    parquet_df

    .apply(
        expected_result,
        axis=1,
    )
)


result_mismatches = int(

    (
        expected_results
        != parquet_df[
            "time_machine_result"
        ]
    )
    .sum()
)


if result_mismatches != 0:

    raise RuntimeError(
        "Time Machine categories "
        "are inconsistent."
    )


print(
    "\nTime Machine logic valid."
)


print(

    parquet_df[
        "time_machine_result"
    ]

    .value_counts()

    .to_string()
)


# ============================================================
# 10. Explanation completeness
# ============================================================

EXPLANATION_COLUMNS = [

    "risk_driver_1_label",
    "risk_driver_1_value",

    "risk_driver_2_label",
    "risk_driver_2_value",

    "risk_driver_3_label",
    "risk_driver_3_value",
]


missing_explanations = (

    parquet_df[
        EXPLANATION_COLUMNS
    ]

    .isna()

    .any(
        axis=1
    )

    .sum()
)


print(
    f"\nCells with incomplete "
    f"top-3 explanations: "
    f"{missing_explanations:,}"
)


# ============================================================
# 11. GeoJSON H3 integrity
# ============================================================

geo_ids = {

    feature[
        "properties"
    ][
        "h3_index"
    ]

    for feature
    in geojson[
        "features"
    ]
}


table_ids = set(

    parquet_df[
        "h3_index"
    ]
    .astype(str)
)


missing_geometry = (

    table_ids
    - geo_ids
)


extra_geometry = (

    geo_ids
    - table_ids
)


print(
    f"\nMissing GeoJSON IDs: "
    f"{len(missing_geometry)}"
)

print(
    f"Extra GeoJSON IDs:   "
    f"{len(extra_geometry)}"
)


if (
    missing_geometry
    or extra_geometry
):

    raise RuntimeError(
        "GeoJSON H3 IDs do not "
        "exactly match dashboard table."
    )


# ============================================================
# 12. Metadata integrity
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "METADATA"
)

print("=" * 80)


print(
    f"Package version:   "
    f"{metadata['package_version']}"
)

print(
    f"Model version:     "
    f"{metadata['model_version']}"
)

print(
    f"Predictor quarter: "
    f"{metadata['predictor_quarter']}"
)

print(
    f"Forecast quarter:  "
    f"{metadata['forecast_quarter']}"
)

print(
    f"Eligible cells:    "
    f"{metadata['eligible_cells']:,}"
)


if (
    metadata[
        "eligible_cells"
    ]
    != n_cells
):

    raise RuntimeError(
        "Metadata cell count "
        "does not match package."
    )


# ============================================================
# 13. Final frontend summary
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "FRONTEND HANDOFF SUMMARY"
)

print("=" * 80)


high = parquet_df[
    parquet_df[
        "risk_band"
    ] == "high"
]


elevated = parquet_df[
    parquet_df[
        "risk_band"
    ] == "elevated"
]


print(
    f"\nHigh-risk cells: "
    f"{len(high):,}"
)

print(
    f"High-risk actual event rate: "
    f"{high['onset_target'].mean():.1%}"
)

print(
    f"\nElevated-risk cells: "
    f"{len(elevated):,}"
)

print(
    f"Elevated-risk actual event rate: "
    f"{elevated['onset_target'].mean():.1%}"
)


flagged = parquet_df[
    parquet_df[
        "risk_rank"
    ] <= 200
]


captured = int(

    flagged[
        "onset_target"
    ].sum()
)


total_events = int(

    parquet_df[
        "onset_target"
    ].sum()
)


print(
    f"\nTop-200 captured: "
    f"{captured}/{total_events} "
    f"events "
    f"({captured / total_events:.1%})"
)


# ============================================================
# 14. Complete
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CHECKPOINT 12 COMPLETE"
)

print("=" * 80)


print(
    "\nFrontend package passed "
    "all structural QA checks."
)