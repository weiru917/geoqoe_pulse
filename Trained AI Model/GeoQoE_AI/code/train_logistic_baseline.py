from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_registry import APPROVED_FEATURES, TARGET_COLUMN
from temporal_split import create_temporal_split


# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)

DUMMY_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "dummy_validation_metrics.json"
)

METRICS_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

METRICS_PATH = (
    METRICS_DIRECTORY
    / "logistic_validation_metrics.json"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "logistic_regression_development.joblib"
)


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Logistic Regression Baseline")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    required_columns = [
        "h3_index",
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        TARGET_COLUMN,
        *APPROVED_FEATURES,
    ]

    # Remove repeated column names while preserving their order.
    required_columns = list(dict.fromkeys(required_columns))

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

    # Remove rows without a complete 24-hour history.
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

    print("\n--- Split sizes ---")
    print("Training rows:", f"{len(train_df):,}")
    print("Validation rows:", f"{len(validation_df):,}")
    print("Test rows kept untouched:", f"{len(test_df):,}")

    # X contains the model input features.
    # y contains the correct answer that the model must learn.
    X_train = train_df[
        APPROVED_FEATURES
    ].astype("float32")

    y_train = train_df[
        TARGET_COLUMN
    ].astype("int8")

    X_validation = validation_df[
        APPROVED_FEATURES
    ].astype("float32")

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype("int8")

    print("\n--- Training data ---")
    print("Input features:", X_train.shape[1])
    print("Training examples:", f"{X_train.shape[0]:,}")
    print(
        "Training positive rate:",
        f"{y_train.mean() * 100:.4f}%",
    )

    print("\n--- Validation data ---")
    print("Validation examples:", f"{X_validation.shape[0]:,}")
    print(
        "Validation positive rate:",
        f"{y_validation.mean() * 100:.4f}%",
    )

    # Confirm that all model inputs contain valid finite numbers.
    train_is_finite = np.isfinite(
        X_train.to_numpy(copy=False)
    ).all()

    validation_is_finite = np.isfinite(
        X_validation.to_numpy(copy=False)
    ).all()

    print("\n--- Numerical input check ---")
    print("All training values are finite:", train_is_finite)
    print(
        "All validation values are finite:",
        validation_is_finite,
    )

    if not train_is_finite or not validation_is_finite:
        raise ValueError(
            "The model inputs contain NaN or infinite values."
        )

    # A Pipeline connects the scaler and classifier.
    #
    # During model.fit():
    # 1. StandardScaler learns scaling values from training data.
    # 2. It transforms the training features.
    # 3. LogisticRegression learns from the transformed features.
    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=300,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression...")
    print(
        "This may take several minutes depending on your laptop."
    )

    training_start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter() - training_start_time
    )

    print("Training completed.")
    print(
        "Training time:",
        f"{training_seconds:.2f} seconds",
    )

    classifier = model.named_steps["classifier"]

    print(
        "Iterations used:",
        int(classifier.n_iter_[0]),
    )

    # Produce probabilities for validation rows.
    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    # This threshold is only provisional.
    # We will tune it later using the validation set.
    threshold = 0.50

    validation_prediction = (
        validation_probability >= threshold
    ).astype("int8")

    confusion = confusion_matrix(
        y_validation,
        validation_prediction,
        labels=[0, 1],
    )

    metrics = {
        "model": "LogisticRegression",
        "model_status": "development",
        "evaluation_split": "validation",
        "number_of_features": len(APPROVED_FEATURES),
        "threshold": threshold,
        "class_weight": "balanced",
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows_untouched": len(test_df),
        "training_positive_rate": float(
            y_train.mean()
        ),
        "validation_positive_rate": float(
            y_validation.mean()
        ),
        "training_seconds": float(
            training_seconds
        ),
        "iterations_used": int(
            classifier.n_iter_[0]
        ),
        "predicted_positive_rate": float(
            validation_prediction.mean()
        ),
        "mean_predicted_probability": float(
            validation_probability.mean()
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
        "feature_names": APPROVED_FEATURES,
    }

    print("\n--- Logistic Regression validation results ---")

    print(
        "Mean predicted degradation probability:",
        f"{metrics['mean_predicted_probability'] * 100:.4f}%",
    )

    print(
        "Predicted-positive rate at threshold 0.50:",
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

    # Compare against the previously saved dummy result.
    if DUMMY_METRICS_PATH.exists():
        with open(
            DUMMY_METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            dummy_metrics = json.load(file)

        dummy_pr_auc = float(
            dummy_metrics["pr_auc"]
        )

        improvement = (
            metrics["pr_auc"] - dummy_pr_auc
        )

        ratio = (
            metrics["pr_auc"] / dummy_pr_auc
            if dummy_pr_auc > 0
            else None
        )

        print("\n--- Comparison with dummy baseline ---")
        print(
            "Dummy PR-AUC:",
            f"{dummy_pr_auc:.4f}",
        )
        print(
            "Logistic Regression PR-AUC:",
            f"{metrics['pr_auc']:.4f}",
        )
        print(
            "Absolute PR-AUC improvement:",
            f"{improvement:.4f}",
        )

        if ratio is not None:
            print(
                "PR-AUC multiple over dummy:",
                f"{ratio:.2f} times",
            )

        metrics["dummy_pr_auc"] = dummy_pr_auc
        metrics["absolute_pr_auc_improvement"] = float(
            improvement
        )
        metrics["pr_auc_multiple_over_dummy"] = (
            float(ratio)
            if ratio is not None
            else None
        )
    else:
        print(
            "\nDummy metrics file was not found, "
            "so automatic comparison was skipped."
        )

    # Save model and results.
    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
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

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(f"\nMetrics saved to:\n{METRICS_PATH}")
    print(f"\nDevelopment model saved to:\n{MODEL_PATH}")

    print("\nLogistic Regression baseline completed.")
    print(
        "The test set was not used during this checkpoint."
    )


if __name__ == "__main__":
    main()