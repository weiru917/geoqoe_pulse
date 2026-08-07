from __future__ import annotations

import gc
import json
import time
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


# Main GeoQoE_AI project folder.
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

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

XGBOOST_METRICS_PATH = (
    METRICS_DIRECTORY
    / "xgboost_validation_metrics.json"
)

TRAINING_HISTORY_PATH = (
    METRICS_DIRECTORY
    / "xgboost_training_history.json"
)

MODEL_PATH = (
    MODEL_DIRECTORY
    / "xgboost_development.json"
)

DUMMY_METRICS_PATH = (
    METRICS_DIRECTORY
    / "dummy_validation_metrics.json"
)

LOGISTIC_METRICS_PATH = (
    METRICS_DIRECTORY
    / "logistic_validation_metrics.json"
)


def load_saved_pr_auc(
    path: Path,
) -> float | None:
    """
    Read the PR-AUC from a previously saved metrics file.

    Returns None when the file does not exist or does not
    contain a valid PR-AUC value.
    """

    if not path.exists():
        return None

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        return float(metrics["pr_auc"])

    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — XGBoost Development Model")
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

    # Remove repeated names while preserving order.
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

    # Remove the first 24 hours because complete historical
    # information is unavailable during that warm-up period.
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

    test_rows = len(test_df)

    print("\n--- Split sizes ---")
    print("Training rows:", f"{len(train_df):,}")
    print("Validation rows:", f"{len(validation_df):,}")
    print("Test rows kept untouched:", f"{test_rows:,}")

    # Select the 39 approved input features.
    #
    # float32 uses less memory than float64 and is sufficient
    # for this development model.
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
    print("Training examples:", f"{len(X_train):,}")
    print("Input features:", X_train.shape[1])
    print(
        "Training positive rate:",
        f"{y_train.mean() * 100:.4f}%",
    )

    print("\n--- Validation data ---")
    print("Validation examples:", f"{len(X_validation):,}")
    print(
        "Validation positive rate:",
        f"{y_validation.mean() * 100:.4f}%",
    )

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

    # Calculate the class-imbalance weight.
    #
    # The degraded class is rare, so positive examples receive
    # more influence during model training.
    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "The training set contains no degraded examples."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\n--- Class imbalance ---")
    print(
        "Normal training examples:",
        f"{negative_count:,}",
    )
    print(
        "Degraded training examples:",
        f"{positive_count:,}",
    )
    print(
        "scale_pos_weight:",
        f"{scale_pos_weight:.4f}",
    )

    # XGBoost uses a DMatrix as its optimised data container.
    print("\nCreating XGBoost data matrices...")

    dtrain = xgb.DMatrix(
        data=X_train,
        label=y_train,
        feature_names=APPROVED_FEATURES,
    )

    dvalidation = xgb.DMatrix(
        data=X_validation,
        label=y_validation,
        feature_names=APPROVED_FEATURES,
    )

    print("Data matrices created.")

    # Release some unnecessary DataFrame objects before training.
    del df
    del clean_df
    del train_df
    del validation_df
    del test_df

    gc.collect()

    # These are reasonable initial development settings.
    #
    # They are not claimed to be the final optimal settings.
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 5,
        "eta": 0.05,
        "min_child_weight": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "verbosity": 1,
    }

    maximum_boosting_rounds = 500
    early_stopping_rounds = 30

    evaluation_history: dict = {}

    print("\nTraining XGBoost...")
    print(
        "Maximum boosting rounds:",
        maximum_boosting_rounds,
    )
    print(
        "Early-stopping patience:",
        early_stopping_rounds,
        "rounds",
    )
    print(
        "Training updates will appear every 25 rounds."
    )
    print(
        "This may take several minutes depending on your laptop."
    )

    training_start_time = time.perf_counter()

    model = xgb.train(
        params=parameters,
        dtrain=dtrain,
        num_boost_round=maximum_boosting_rounds,
        evals=[
            (dtrain, "training"),
            (dvalidation, "validation"),
        ],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evaluation_history,
        verbose_eval=25,
    )

    training_seconds = (
        time.perf_counter() - training_start_time
    )

    best_iteration = int(
        model.best_iteration
    )

    best_validation_aucpr = float(
        model.best_score
    )

    print("\nTraining completed.")
    print(
        "Training time:",
        f"{training_seconds:.2f} seconds",
    )
    print(
        "Best boosting iteration:",
        best_iteration,
    )
    print(
        "Best validation AUC-PR reported by XGBoost:",
        f"{best_validation_aucpr:.6f}",
    )

    # Use only the trees up to the best iteration.
    validation_probability = model.predict(
        dvalidation,
        iteration_range=(
            0,
            best_iteration + 1,
        ),
    )

    # This is only the initial reference threshold.
    # We will tune the XGBoost threshold in the next checkpoint.
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
        "model": "XGBoost",
        "model_status": "development",
        "evaluation_split": "validation",
        "target": TARGET_COLUMN,
        "number_of_features": len(APPROVED_FEATURES),
        "training_rows": len(y_train),
        "validation_rows": len(y_validation),
        "test_rows_untouched": test_rows,
        "training_positive_rate": float(
            y_train.mean()
        ),
        "validation_positive_rate": float(
            y_validation.mean()
        ),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "maximum_boosting_rounds": (
            maximum_boosting_rounds
        ),
        "early_stopping_rounds": (
            early_stopping_rounds
        ),
        "best_iteration": best_iteration,
        "best_validation_aucpr_xgboost": (
            best_validation_aucpr
        ),
        "training_seconds": float(
            training_seconds
        ),
        "threshold": threshold,
        "mean_predicted_probability": float(
            validation_probability.mean()
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
        "feature_names": APPROVED_FEATURES,
        "parameters": parameters,
        "test_set_used": False,
    }

    print("\n--- XGBoost validation results at threshold 0.50 ---")

    print(
        "Mean predicted degradation score:",
        f"{metrics['mean_predicted_probability'] * 100:.4f}%",
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

    dummy_pr_auc = load_saved_pr_auc(
        DUMMY_METRICS_PATH
    )

    logistic_pr_auc = load_saved_pr_auc(
        LOGISTIC_METRICS_PATH
    )

    print("\n--- Model comparison ---")

    if dummy_pr_auc is not None:
        print(
            "Dummy PR-AUC:",
            f"{dummy_pr_auc:.4f}",
        )

        metrics["dummy_pr_auc"] = dummy_pr_auc

    else:
        print("Dummy PR-AUC file was unavailable.")

    if logistic_pr_auc is not None:
        print(
            "Logistic Regression PR-AUC:",
            f"{logistic_pr_auc:.4f}",
        )

        metrics[
            "logistic_regression_pr_auc"
        ] = logistic_pr_auc

        metrics[
            "pr_auc_improvement_over_logistic"
        ] = float(
            metrics["pr_auc"] - logistic_pr_auc
        )

        print(
            "XGBoost PR-AUC:",
            f"{metrics['pr_auc']:.4f}",
        )

        print(
            "Absolute improvement over Logistic Regression:",
            f"{metrics['pr_auc'] - logistic_pr_auc:.4f}",
        )

    else:
        print(
            "Logistic Regression PR-AUC file was unavailable."
        )
        print(
            "XGBoost PR-AUC:",
            f"{metrics['pr_auc']:.4f}",
        )

    # Create output folders.
    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save evaluation results.
    with open(
        XGBOOST_METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save training PR-AUC values for every boosting round.
    with open(
        TRAINING_HISTORY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_history,
            file,
            indent=4,
        )

    # Save the development model.
    model.save_model(MODEL_PATH)

    print(f"\nMetrics saved to:\n{XGBOOST_METRICS_PATH}")

    print(
        f"\nTraining history saved to:\n"
        f"{TRAINING_HISTORY_PATH}"
    )

    print(f"\nDevelopment model saved to:\n{MODEL_PATH}")

    print("\nXGBoost development training completed.")
    print(
        "The test set was not used during this checkpoint."
    )

    print(
        "\nReminder: these results describe the current "
        "development simulation dataset, not verified "
        "real-world forecasting performance."
    )


if __name__ == "__main__":
    main()