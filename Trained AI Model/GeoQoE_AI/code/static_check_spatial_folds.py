from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from static_feature_registry import (
    APPROVED_STATIC_FEATURES,
    ID_COLUMN,
    LABEL_STATUS_COLUMN,
    SAMPLE_WEIGHT_COLUMN,
    SPATIAL_GROUP_COLUMN,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
    / "cell_static_qoe_baseline.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "static_model"
)

FOLD_ASSIGNMENT_PATH = (
    OUTPUT_DIRECTORY
    / "static_spatial_fold_assignments.csv"
)

NUMBER_OF_FOLDS = 5


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Static Spatial Fold Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Static dataset not found:\n{DATA_PATH}"
        )

    required_columns = [
        ID_COLUMN,
        SPATIAL_GROUP_COLUMN,
        LABEL_STATUS_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        TARGET_COLUMN,
        *APPROVED_STATIC_FEATURES,
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    print(f"\nLoading labelled static data from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    labelled_df = df[
        df[LABEL_STATUS_COLUMN].astype(bool)
    ].copy()

    labelled_df = labelled_df.reset_index(drop=True)

    X = labelled_df[
        APPROVED_STATIC_FEATURES
    ].astype("float32")

    y = labelled_df[
        TARGET_COLUMN
    ].astype("float32")

    groups = labelled_df[
        SPATIAL_GROUP_COLUMN
    ].astype(str)

    print("\n--- Dataset summary ---")
    print("Labelled cells:", f"{len(labelled_df):,}")
    print("Approved features:", X.shape[1])
    print(
        "Spatial parent groups:",
        f"{groups.nunique():,}",
    )

    print("\nOverall target summary:")
    print(y.describe())

    group_kfold = GroupKFold(
        n_splits=NUMBER_OF_FOLDS
    )

    # Each row will eventually receive one validation-fold number.
    fold_assignment = np.full(
        shape=len(labelled_df),
        fill_value=-1,
        dtype="int8",
    )

    fold_summaries = []
    all_checks_passed = True

    print("\n--- Fold summaries ---")

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        train_groups = set(
            groups.iloc[train_indices]
        )

        validation_groups = set(
            groups.iloc[validation_indices]
        )

        overlapping_groups = (
            train_groups.intersection(
                validation_groups
            )
        )

        fold_assignment[
            validation_indices
        ] = fold_number

        y_train = y.iloc[train_indices]
        y_validation = y.iloc[
            validation_indices
        ]

        train_weights = labelled_df.iloc[
            train_indices
        ][SAMPLE_WEIGHT_COLUMN]

        validation_weights = labelled_df.iloc[
            validation_indices
        ][SAMPLE_WEIGHT_COLUMN]

        fold_checks_passed = (
            len(train_indices) > 0
            and len(validation_indices) > 0
            and len(overlapping_groups) == 0
            and np.isfinite(
                y_train.to_numpy()
            ).all()
            and np.isfinite(
                y_validation.to_numpy()
            ).all()
        )

        if not fold_checks_passed:
            all_checks_passed = False

        print(f"\nFOLD {fold_number}")

        print(
            "Training cells:",
            f"{len(train_indices):,}",
        )

        print(
            "Validation cells:",
            f"{len(validation_indices):,}",
        )

        print(
            "Training spatial groups:",
            f"{len(train_groups):,}",
        )

        print(
            "Validation spatial groups:",
            f"{len(validation_groups):,}",
        )

        print(
            "Overlapping spatial groups:",
            len(overlapping_groups),
        )

        print(
            "Training target mean:",
            f"{y_train.mean():.4f} Mbps",
        )

        print(
            "Validation target mean:",
            f"{y_validation.mean():.4f} Mbps",
        )

        print(
            "Validation target median:",
            f"{y_validation.median():.4f} Mbps",
        )

        print(
            "Validation target minimum:",
            f"{y_validation.min():.4f} Mbps",
        )

        print(
            "Validation target maximum:",
            f"{y_validation.max():.4f} Mbps",
        )

        print(
            "Validation Ookla-test median:",
            f"{validation_weights.median():.1f}",
        )

        print(
            "Fold check passed:",
            fold_checks_passed,
        )

        fold_summaries.append(
            {
                "fold": fold_number,
                "training_cells": len(
                    train_indices
                ),
                "validation_cells": len(
                    validation_indices
                ),
                "training_groups": len(
                    train_groups
                ),
                "validation_groups": len(
                    validation_groups
                ),
                "overlapping_groups": len(
                    overlapping_groups
                ),
                "training_target_mean_mbps": float(
                    y_train.mean()
                ),
                "validation_target_mean_mbps": float(
                    y_validation.mean()
                ),
                "validation_target_median_mbps": float(
                    y_validation.median()
                ),
                "validation_target_min_mbps": float(
                    y_validation.min()
                ),
                "validation_target_max_mbps": float(
                    y_validation.max()
                ),
                "validation_ookla_tests_median": float(
                    validation_weights.median()
                ),
            }
        )

    print("\n--- Fold-assignment reconciliation ---")

    unassigned_rows = int(
        (fold_assignment == -1).sum()
    )

    fold_counts = pd.Series(
        fold_assignment
    ).value_counts().sort_index()

    print("Unassigned labelled cells:", unassigned_rows)

    print("\nValidation cells assigned to each fold:")
    print(fold_counts)

    every_row_assigned_once = (
        unassigned_rows == 0
        and fold_counts.sum() == len(labelled_df)
        and len(fold_counts) == NUMBER_OF_FOLDS
    )

    print(
        "Every labelled cell assigned to one validation fold:",
        every_row_assigned_once,
    )

    assignment_df = labelled_df[
        [
            ID_COLUMN,
            SPATIAL_GROUP_COLUMN,
            TARGET_COLUMN,
            SAMPLE_WEIGHT_COLUMN,
        ]
    ].copy()

    assignment_df["spatial_cv_fold"] = (
        fold_assignment
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignment_df.to_csv(
        FOLD_ASSIGNMENT_PATH,
        index=False,
    )

    final_result = (
        all_checks_passed
        and every_row_assigned_once
    )

    print("\n--- Final result ---")

    if final_result:
        print("Static spatial-fold check PASSED.")
        print(
            "The labelled cells can be evaluated using "
            f"{NUMBER_OF_FOLDS}-fold spatial GroupKFold."
        )
        print(
            "No parent H3 spatial group is shared between "
            "training and validation inside a fold."
        )
    else:
        print("Static spatial-fold check FAILED.")

    print(
        f"\nFold assignments saved to:\n"
        f"{FOLD_ASSIGNMENT_PATH}"
    )


if __name__ == "__main__":
    main()