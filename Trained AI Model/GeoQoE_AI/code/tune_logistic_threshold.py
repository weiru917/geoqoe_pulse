from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from feature_registry import APPROVED_FEATURES, TARGET_COLUMN
from temporal_split import create_temporal_split


# Main GeoQoE_AI folder.
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
    / "logistic_regression_development.joblib"
)

METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

PLOTS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
)

THRESHOLD_RESULTS_PATH = (
    METRICS_DIRECTORY
    / "logistic_threshold_results.csv"
)

SELECTED_THRESHOLD_PATH = (
    METRICS_DIRECTORY
    / "logistic_selected_threshold.json"
)

PLOT_PATH = (
    PLOTS_DIRECTORY
    / "logistic_threshold_comparison.png"
)


def calculate_threshold_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """
    Convert model scores into class predictions and calculate
    precision, recall, F1, and warning frequency.
    """

    predictions = (
        probabilities >= threshold
    ).astype("int8")

    confusion = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    true_negative = int(confusion[0, 0])
    false_positive = int(confusion[0, 1])
    false_negative = int(confusion[1, 0])
    true_positive = int(confusion[1, 1])

    return {
        "threshold": float(threshold),
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
        "predicted_positive_rate": float(
            predictions.mean()
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def print_candidate(
    title: str,
    result: pd.Series,
) -> None:
    """Print one selected threshold and its results."""

    print(f"\n--- {title} ---")

    print(
        "Threshold:",
        f"{result['threshold']:.2f}",
    )

    print(
        "Precision:",
        f"{result['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{result['recall']:.4f}",
    )

    print(
        "F1 score:",
        f"{result['f1']:.4f}",
    )

    print(
        "Predicted-positive rate:",
        f"{result['predicted_positive_rate'] * 100:.4f}%",
    )

    print(
        "True positives:",
        f"{int(result['true_positive']):,}",
    )

    print(
        "False positives:",
        f"{int(result['false_positive']):,}",
    )

    print(
        "False negatives:",
        f"{int(result['false_negative']):,}",
    )

    print(
        "True negatives:",
        f"{int(result['true_negative']):,}",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Logistic Threshold Tuning")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved Logistic Regression model not found:\n{MODEL_PATH}"
        )

    required_columns = [
        "h3_index",
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        TARGET_COLUMN,
        *APPROVED_FEATURES,
    ]

    required_columns = list(
        dict.fromkeys(required_columns)
    )

    print(f"\nLoading validation data from:\n{DATA_PATH}")

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

    # Remove the first 24 hours.
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
        "Training rows not used for tuning:",
        f"{len(train_df):,}",
    )
    print(
        "Validation rows used for tuning:",
        f"{len(validation_df):,}",
    )
    print(
        "Test rows kept untouched:",
        f"{len(test_df):,}",
    )

    X_validation = validation_df[
        APPROVED_FEATURES
    ].astype("float32")

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype("int8")

    print("\nLoading saved Logistic Regression model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    print("\nTesting thresholds from 0.01 to 0.99...")

    threshold_values = np.linspace(
        0.01,
        0.99,
        99,
    )

    results = []

    for threshold in threshold_values:
        result = calculate_threshold_metrics(
            y_true=y_validation,
            probabilities=validation_probability,
            threshold=float(threshold),
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # Candidate 1:
    # Threshold with the highest validation F1 score.
    best_f1_index = results_df["f1"].idxmax()

    best_f1_result = results_df.loc[
        best_f1_index
    ]

    # Candidate 2:
    # Among thresholds retaining at least 90% recall,
    # choose the one with the highest precision.
    high_recall_candidates = results_df[
        results_df["recall"] >= 0.90
    ]

    if high_recall_candidates.empty:
        high_recall_result = None
    else:
        high_recall_index = (
            high_recall_candidates["precision"]
            .idxmax()
        )

        high_recall_result = results_df.loc[
            high_recall_index
        ]

    # Retrieve the current 0.50-threshold result for comparison.
    default_result = results_df[
        np.isclose(
            results_df["threshold"],
            0.50,
        )
    ].iloc[0]

    print_candidate(
        "DEFAULT THRESHOLD",
        default_result,
    )

    print_candidate(
        "BEST F1 THRESHOLD",
        best_f1_result,
    )

    if high_recall_result is not None:
        print_candidate(
            "BEST PRECISION WHILE KEEPING RECALL AT LEAST 90%",
            high_recall_result,
        )

    # For this development stage, save the best-F1 threshold
    # as the provisional selected threshold.
    selected_threshold = float(
        best_f1_result["threshold"]
    )

    selected_information = {
        "model": "LogisticRegression",
        "model_status": "development",
        "selection_dataset": "validation",
        "selection_rule": "maximum validation F1",
        "selected_threshold": selected_threshold,
        "precision": float(
            best_f1_result["precision"]
        ),
        "recall": float(
            best_f1_result["recall"]
        ),
        "f1": float(
            best_f1_result["f1"]
        ),
        "predicted_positive_rate": float(
            best_f1_result[
                "predicted_positive_rate"
            ]
        ),
        "true_positive": int(
            best_f1_result["true_positive"]
        ),
        "false_positive": int(
            best_f1_result["false_positive"]
        ),
        "false_negative": int(
            best_f1_result["false_negative"]
        ),
        "true_negative": int(
            best_f1_result["true_negative"]
        ),
        "test_set_used": False,
    }

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        THRESHOLD_RESULTS_PATH,
        index=False,
    )

    with open(
        SELECTED_THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected_information,
            file,
            indent=4,
        )

    # Plot precision, recall, and F1 against threshold.
    plt.figure(figsize=(9, 6))

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        label="Precision",
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        label="Recall",
    )

    plt.plot(
        results_df["threshold"],
        results_df["f1"],
        label="F1 score",
    )

    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            "Selected threshold "
            f"({selected_threshold:.2f})"
        ),
    )

    plt.xlabel("Classification threshold")
    plt.ylabel("Metric value")
    plt.title(
        "Logistic Regression Threshold Comparison"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=200,
    )

    plt.close()

    print(
        "\nProvisional selected threshold:",
        f"{selected_threshold:.2f}",
    )

    print(f"\nFull threshold table saved to:\n{THRESHOLD_RESULTS_PATH}")

    print(f"\nSelected threshold saved to:\n{SELECTED_THRESHOLD_PATH}")

    print(f"\nThreshold plot saved to:\n{PLOT_PATH}")

    print("\nThreshold tuning completed.")
    print(
        "The test set was not used during this checkpoint."
    )


if __name__ == "__main__":
    main()