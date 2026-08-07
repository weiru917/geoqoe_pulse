from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_registry import TARGET_COLUMN
from temporal_split import create_temporal_split


# Locate the main GeoQoE_AI project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)

METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

METRICS_PATH = (
    METRICS_DIRECTORY
    / "dummy_validation_metrics.json"
)


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Dummy Baseline")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    # The dummy model does not use real features.
    # We only need the columns required for filtering and splitting.
    required_columns = [
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        TARGET_COLUMN,
    ]

    print(f"\nLoading required columns from:\n{DATA_PATH}")

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

    # Remove the first 24 hours because they do not contain
    # complete historical lag information.
    clean_df = df[
        df["in_warmup_window"] == 0
    ].copy()

    clean_df = clean_df.sort_values(
        "forecast_timestamp"
    ).reset_index(drop=True)

    # Use exactly the same chronological split as before.
    train_df, validation_df, test_df = create_temporal_split(
        clean_df,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    print("\n--- Split sizes ---")
    print("Training rows:", f"{len(train_df):,}")
    print("Validation rows:", f"{len(validation_df):,}")
    print("Test rows kept untouched:", f"{len(test_df):,}")

    y_train = train_df[TARGET_COLUMN].astype(int)
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    print("\n--- Training target distribution ---")
    print(y_train.value_counts().sort_index())

    print(
        "Training positive rate:",
        f"{y_train.mean() * 100:.4f}%",
    )

    print("\n--- Validation target distribution ---")
    print(y_validation.value_counts().sort_index())

    print(
        "Validation positive rate:",
        f"{y_validation.mean() * 100:.4f}%",
    )

    # DummyClassifier requires an X input even though it does not
    # actually learn from the features.
    #
    # We therefore provide one column containing only zeros.
    X_train_dummy = np.zeros(
        shape=(len(train_df), 1),
        dtype=np.int8,
    )

    X_validation_dummy = np.zeros(
        shape=(len(validation_df), 1),
        dtype=np.int8,
    )

    model = DummyClassifier(
        strategy="prior",
        random_state=42,
    )

    print("\nTraining dummy classifier...")

    model.fit(
        X_train_dummy,
        y_train,
    )

    print("Dummy classifier trained.")

    # Find which probability column represents class 1.
    positive_class_index = int(
        np.where(model.classes_ == 1)[0][0]
    )

    validation_probability = model.predict_proba(
        X_validation_dummy
    )[:, positive_class_index]

    # Use 0.5 only as an initial reference threshold.
    threshold = 0.50

    validation_prediction = (
        validation_probability >= threshold
    ).astype(int)

    confusion = confusion_matrix(
        y_validation,
        validation_prediction,
        labels=[0, 1],
    )

    metrics = {
        "model": "DummyClassifier",
        "strategy": "prior",
        "evaluation_split": "validation",
        "threshold": threshold,
        "training_positive_rate": float(
            y_train.mean()
        ),
        "validation_positive_rate": float(
            y_validation.mean()
        ),
        "constant_predicted_probability": float(
            validation_probability[0]
        ),
        "predicted_positive_rate": float(
            validation_prediction.mean()
        ),
        "accuracy": float(
            accuracy_score(
                y_validation,
                validation_prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_validation,
                validation_prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                validation_prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_validation,
                validation_prediction,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_validation,
                validation_probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_validation,
                validation_probability,
            )
        ),
        "confusion_matrix": confusion.tolist(),
    }

    print("\n--- Dummy validation results ---")

    print(
        "Constant predicted degradation probability:",
        f"{metrics['constant_predicted_probability'] * 100:.4f}%",
    )

    print(
        "Predicted-positive rate:",
        f"{metrics['predicted_positive_rate'] * 100:.4f}%",
    )

    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}",
    )

    print(
        "Precision:",
        f"{metrics['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{metrics['recall']:.4f}",
    )

    print(
        "F1 score:",
        f"{metrics['f1']:.4f}",
    )

    print(
        "PR-AUC:",
        f"{metrics['pr_auc']:.4f}",
    )

    print(
        "ROC-AUC:",
        f"{metrics['roc_auc']:.4f}",
    )

    print("\nConfusion matrix:")
    print(confusion)

    print(
        "\nConfusion matrix arrangement:"
        "\n[[true normal predicted normal, "
        "true normal predicted degraded],"
        "\n [true degraded predicted normal, "
        "true degraded predicted degraded]]"
    )

    # Create the output folder automatically.
    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"\nMetrics saved to:\n{METRICS_PATH}")

    print("\nDummy baseline completed.")
    print(
        "The test set was not used during this checkpoint."
    )


if __name__ == "__main__":
    main()