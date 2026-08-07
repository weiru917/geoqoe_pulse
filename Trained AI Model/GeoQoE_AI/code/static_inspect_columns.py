from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
)

STATIC_DATA_PATH = (
    DATA_DIRECTORY
    / "cell_static_qoe_baseline.parquet"
)

FEATURE_DICTIONARY_PATH = (
    DATA_DIRECTORY
    / "feature_dictionary.xlsx"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
)

COLUMN_AUDIT_PATH = (
    OUTPUT_DIRECTORY
    / "static_column_audit.csv"
)


# These columns are known to be IDs, targets, provenance fields,
# or observation-status information.
#
# They are not automatically approved as model inputs.
KNOWN_NON_FEATURE_COLUMNS = {
    "h3_index",
    "h3_parent_res6",
    "has_real_ookla_label",
    "ookla_dl_mbps",
    "ookla_rtt_ms",
    "ookla_tests",
    "ookla_devices",
    "ookla_quarter_test_count",
    "ookla_quarter_device_count",
    "qoe_risk_label",
    "qoe_level_is_ookla_anchored",
    "qoe_baseline_source",
}


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static Model Column Audit")
    print("=" * 70)

    if not STATIC_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Static dataset not found:\n{STATIC_DATA_PATH}"
        )

    print(f"\nLoading static dataset from:\n{STATIC_DATA_PATH}")

    df = pd.read_parquet(STATIC_DATA_PATH)

    print("\nDataset loaded successfully.")
    print("Total rows:", f"{len(df):,}")
    print("Total columns:", len(df.columns))

    if "has_real_ookla_label" not in df.columns:
        raise ValueError(
            "Required column 'has_real_ookla_label' is missing."
        )

    labelled_df = df[
        df["has_real_ookla_label"].astype(bool)
    ].copy()

    print(
        "Real Ookla-labelled rows:",
        f"{len(labelled_df):,}",
    )

    # ---------------------------------------------------------
    # 1. Build a column-level audit table.
    # ---------------------------------------------------------

    audit_rows = []

    for column in df.columns:
        dtype = df[column].dtype

        all_missing_fraction = float(
            df[column].isna().mean()
        )

        labelled_missing_fraction = float(
            labelled_df[column].isna().mean()
        )

        all_unique = int(
            df[column].nunique(dropna=True)
        )

        labelled_unique = int(
            labelled_df[column].nunique(dropna=True)
        )

        is_numeric = bool(
            pd.api.types.is_numeric_dtype(dtype)
        )

        is_constant_in_labelled_data = (
            labelled_unique <= 1
        )

        is_known_non_feature = (
            column in KNOWN_NON_FEATURE_COLUMNS
            or column.startswith("target_")
        )

        preliminary_candidate = (
            is_numeric
            and not is_constant_in_labelled_data
            and not is_known_non_feature
            and labelled_missing_fraction < 1.0
        )

        audit_rows.append(
            {
                "column": column,
                "dtype": str(dtype),
                "is_numeric": is_numeric,
                "all_rows_missing_percent": (
                    all_missing_fraction * 100
                ),
                "labelled_rows_missing_percent": (
                    labelled_missing_fraction * 100
                ),
                "all_rows_unique_values": all_unique,
                "labelled_rows_unique_values": (
                    labelled_unique
                ),
                "constant_in_labelled_data": (
                    is_constant_in_labelled_data
                ),
                "known_non_feature": (
                    is_known_non_feature
                ),
                "preliminary_numeric_candidate": (
                    preliminary_candidate
                ),
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    # ---------------------------------------------------------
    # 2. Print every column.
    # ---------------------------------------------------------

    print("\n--- All columns ---")

    for index, row in audit_df.iterrows():
        candidate_text = (
            "candidate"
            if row["preliminary_numeric_candidate"]
            else "not selected"
        )

        print(
            f"{index + 1:>3}. "
            f"{row['column']:<48} "
            f"{row['dtype']:<16} "
            f"missing in labelled rows: "
            f"{row['labelled_rows_missing_percent']:>6.2f}% "
            f"{candidate_text}"
        )

    # ---------------------------------------------------------
    # 3. Print constant columns.
    # ---------------------------------------------------------

    constant_columns = audit_df.loc[
        audit_df["constant_in_labelled_data"],
        "column",
    ].tolist()

    print("\n--- Constant columns in real-labelled rows ---")

    if constant_columns:
        for column in constant_columns:
            print(f"  - {column}")
    else:
        print("No constant columns found.")

    # ---------------------------------------------------------
    # 4. Print non-numeric columns and values.
    # ---------------------------------------------------------

    non_numeric_columns = audit_df.loc[
        ~audit_df["is_numeric"],
        "column",
    ].tolist()

    print("\n--- Non-numeric columns ---")

    if not non_numeric_columns:
        print("No non-numeric columns found.")
    else:
        for column in non_numeric_columns:
            values = (
                labelled_df[column]
                .dropna()
                .astype(str)
                .value_counts()
                .head(10)
            )

            print(f"\n{column}:")
            print(values)

    # ---------------------------------------------------------
    # 5. Print preliminary numeric candidates.
    # ---------------------------------------------------------

    preliminary_candidates = audit_df.loc[
        audit_df["preliminary_numeric_candidate"],
        "column",
    ].tolist()

    print("\n--- Preliminary numeric feature candidates ---")
    print(
        "Candidate count:",
        len(preliminary_candidates),
    )

    for column in preliminary_candidates:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # 6. Inspect target and observation fields.
    # ---------------------------------------------------------

    print("\n--- Real target summary ---")

    if "ookla_dl_mbps" in labelled_df.columns:
        print("\nookla_dl_mbps:")
        print(
            labelled_df["ookla_dl_mbps"]
            .describe()
        )

    if "ookla_rtt_ms" in labelled_df.columns:
        print("\nookla_rtt_ms:")
        print(
            labelled_df["ookla_rtt_ms"]
            .describe()
        )

    if "ookla_tests" in labelled_df.columns:
        print("\nookla_tests:")
        print(
            labelled_df["ookla_tests"]
            .describe()
        )

    # ---------------------------------------------------------
    # 7. Inspect the feature dictionary workbook.
    # ---------------------------------------------------------

    print("\n--- Feature dictionary workbook ---")

    if not FEATURE_DICTIONARY_PATH.exists():
        print(
            "Feature dictionary was not found, "
            "so workbook inspection was skipped."
        )
    else:
        workbook = pd.ExcelFile(
            FEATURE_DICTIONARY_PATH
        )

        print(
            "Workbook sheets:",
            workbook.sheet_names,
        )

        for sheet_name in workbook.sheet_names:
            sheet_df = pd.read_excel(
                FEATURE_DICTIONARY_PATH,
                sheet_name=sheet_name,
            )

            print(
                f"\nSheet '{sheet_name}' "
                f"shape: {sheet_df.shape}"
            )

            print(
                sheet_df.head(10).to_string(
                    index=False
                )
            )

    # ---------------------------------------------------------
    # 8. Save the audit.
    # ---------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        COLUMN_AUDIT_PATH,
        index=False,
    )

    print(
        f"\nColumn audit saved to:\n"
        f"{COLUMN_AUDIT_PATH}"
    )

    print("\nStatic column audit completed.")


if __name__ == "__main__":
    main()