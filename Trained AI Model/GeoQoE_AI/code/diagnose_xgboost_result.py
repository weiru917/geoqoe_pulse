from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_registry import APPROVED_FEATURES, TARGET_COLUMN
from temporal_split import create_temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_development.json"
)

METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

PERSISTENCE_METRICS_PATH = (
    METRICS_DIRECTORY
    / "persistence_baseline_validation_metrics.json"
)

TRANSITION_TABLE_PATH = (
    METRICS_DIRECTORY
    / "degradation_transition_table.csv"
)

FEATURE_IMPORTANCE_PATH = (
    METRICS_DIRECTORY
    / "xgboost_feature_importance_gain.csv"
)


def calculate_binary_metrics(
    y_true: pd.Series,
    predictions: pd.Series,
    scores: pd.Series,
) -> dict:
    """
    Calculate classification metrics.

    predictions:
        Final 0-or-1 decision.

    scores:
        Ranking score used for PR-AUC and ROC-AUC.
        For the persistence baseline, this is also 0 or 1.
    """

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — XGBoost Result Diagnosis")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    required_columns = [
        "h3_index",
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        "target_is_degraded",
        TARGET_COLUMN,
        "dl_median",
        "rtt_median",
        *APPROVED_FEATURES,
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    print(f"\nLoading selected columns from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    df["forecast_timestamp"] = pd.to_datetime(
        df["forecast_timestamp"],
        errors="raise",
    )

    clean_df = df[
        df["in_warmup_window"] == 0
    ].copy()

    clean_df = clean_df.sort_values(
        ["forecast_timestamp", "h3_index"]
    ).reset_index(drop=True)

    train_df, validation_df, test_df = create_temporal_split(
        clean_df,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    print("\n--- Split confirmation ---")
    print(
        "Training rows:",
        f"{len(train_df):,}",
    )
    print(
        "Validation rows used for diagnosis:",
        f"{len(validation_df):,}",
    )
    print(
        "Test rows kept untouched:",
        f"{len(test_df):,}",
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype("int8")

    current_degradation = validation_df[
        "target_is_degraded"
    ].astype("int8")

    # ----------------------------------------------------------
    # Diagnosis 1:
    # Does current degradation simply persist into the next hour?
    # ----------------------------------------------------------

    persistence_metrics = calculate_binary_metrics(
        y_true=y_validation,
        predictions=current_degradation,
        scores=current_degradation,
    )

    print("\n--- Persistence baseline ---")
    print(
        "Rule: predict next-hour degradation "
        "using the current-hour degradation label."
    )

    print(
        "Accuracy:",
        f"{persistence_metrics['accuracy']:.4f}",
    )
    print(
        "Precision:",
        f"{persistence_metrics['precision']:.4f}",
    )
    print(
        "Recall:",
        f"{persistence_metrics['recall']:.4f}",
    )
    print(
        "F1 score:",
        f"{persistence_metrics['f1']:.4f}",
    )
    print(
        "PR-AUC:",
        f"{persistence_metrics['pr_auc']:.4f}",
    )
    print(
        "ROC-AUC:",
        f"{persistence_metrics['roc_auc']:.4f}",
    )

    print("\nConfusion matrix:")
    print(
        np.array(
            persistence_metrics["confusion_matrix"]
        )
    )

    # ----------------------------------------------------------
    # Diagnosis 2:
    # Check whether the current target is directly generated from
    # download speed and RTT thresholds.
    # ----------------------------------------------------------

    threshold_rule_current = (
        (validation_df["dl_median"] < 10.0)
        | (validation_df["rtt_median"] > 100.0)
    ).astype("int8")

    rule_matches_current_target = (
        threshold_rule_current
        == current_degradation
    )

    exact_rule_match_rate = float(
        rule_matches_current_target.mean()
    )

    mismatched_rule_rows = int(
        (~rule_matches_current_target).sum()
    )

    print("\n--- Current-label threshold-rule check ---")
    print(
        "Tested rule:"
        "\ncurrent degraded = "
        "(dl_median < 10 Mbps) OR (rtt_median > 100 ms)"
    )

    print(
        "Exact match rate with target_is_degraded:",
        f"{exact_rule_match_rate * 100:.4f}%",
    )

    print(
        "Rows that do not match the rule:",
        f"{mismatched_rule_rows:,}",
    )

    # ----------------------------------------------------------
    # Diagnosis 3:
    # Examine transitions from the current hour to the next hour.
    # ----------------------------------------------------------

    transition_table = pd.crosstab(
        current_degradation,
        y_validation,
        rownames=["current_hour_degraded"],
        colnames=["next_hour_degraded"],
        margins=True,
    )

    transition_percentages = pd.crosstab(
        current_degradation,
        y_validation,
        normalize="index",
    ) * 100

    print("\n--- Degradation transition counts ---")
    print(transition_table)

    print("\n--- Transition percentages by current state ---")
    print(transition_percentages)

    same_state_rate = float(
        (
            current_degradation
            == y_validation
        ).mean()
    )

    print(
        "\nPercentage retaining the same state next hour:",
        f"{same_state_rate * 100:.4f}%",
    )

    # ----------------------------------------------------------
    # Diagnosis 4:
    # Inspect which variables the XGBoost model used.
    # ----------------------------------------------------------

    print("\nLoading XGBoost model...")

    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    print("XGBoost model loaded.")

    gain_importance = model.get_score(
        importance_type="gain"
    )

    importance_df = pd.DataFrame(
        {
            "feature": list(
                gain_importance.keys()
            ),
            "gain_importance": list(
                gain_importance.values()
            ),
        }
    )

    if importance_df.empty:
        print(
            "\nNo feature importance information was returned."
        )
    else:
        importance_df = importance_df.sort_values(
            "gain_importance",
            ascending=False,
        ).reset_index(drop=True)

        total_gain = importance_df[
            "gain_importance"
        ].sum()

        importance_df[
            "gain_fraction"
        ] = (
            importance_df["gain_importance"]
            / total_gain
        )

        print("\n--- Top XGBoost features by gain ---")

        top_features = importance_df.head(15)

        for index, row in top_features.iterrows():
            print(
                f"{index + 1:>2}. "
                f"{row['feature']:<45} "
                f"gain fraction: "
                f"{row['gain_fraction'] * 100:>8.4f}%"
            )

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PERSISTENCE_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model": "current-hour persistence baseline",
                "evaluation_split": "validation",
                "metrics": persistence_metrics,
                "current_threshold_rule_match_rate": (
                    exact_rule_match_rate
                ),
                "current_threshold_rule_mismatch_rows": (
                    mismatched_rule_rows
                ),
                "same_state_next_hour_rate": (
                    same_state_rate
                ),
                "test_set_used": False,
            },
            file,
            indent=4,
        )

    transition_table.to_csv(
        TRANSITION_TABLE_PATH,
    )

    if not importance_df.empty:
        importance_df.to_csv(
            FEATURE_IMPORTANCE_PATH,
            index=False,
        )

    print(
        f"\nPersistence results saved to:\n"
        f"{PERSISTENCE_METRICS_PATH}"
    )

    print(
        f"\nTransition table saved to:\n"
        f"{TRANSITION_TABLE_PATH}"
    )

    if not importance_df.empty:
        print(
            f"\nFeature importance saved to:\n"
            f"{FEATURE_IMPORTANCE_PATH}"
        )

    print("\nDiagnosis completed.")
    print(
        "The test set was not used during this checkpoint."
    )


if __name__ == "__main__":
    main()