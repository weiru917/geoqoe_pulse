from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from onset_feature_registry import (
    ANCHOR_COLUMN,
    CONTROL_COLUMNS,
    CURRENT_STATE_COLUMN,
    ID_COLUMN,
    NEXT_STATE_COLUMN,
    QOE_HISTORY_ONLY_FEATURES,
    SPATIAL_GROUP_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    TIME_ONLY_FEATURES,
    WARMUP_COLUMN,
    assign_temporal_split,
    unique_preserving_order,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "pipeline_v3_2"
    / "cell_time_dataset.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "onset_logistic"
)

METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_validation_metrics.json"
)

METRICS_TABLE_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_validation_metrics.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_validation_predictions.parquet"
)

TOP_K_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_top_k_metrics.csv"
)

COEFFICIENTS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_coefficients.csv"
)

HOURLY_SCORE_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_validation_scores_by_hour.csv"
)

THRESHOLD_PATH = (
    OUTPUT_DIRECTORY
    / "onset_logistic_selected_thresholds.json"
)


RANDOM_SEED = 42

LOGISTIC_C = 0.10
LOGISTIC_MAX_ITERATIONS = 2000


# Treat these as categorical rather than continuous.
# This allows the time-only model to learn that, for example,
# 07:00 and 16:00 may have distinct behaviour.
TIME_CATEGORICAL_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_peak_period",
]

TIME_NUMERIC_FEATURES = [
    "ridership_daily_surge_factor",
]


def calculate_ranking_metrics(
    y_true: np.ndarray,
    risk_score: np.ndarray,
) -> dict:
    """
    Calculate threshold-independent ranking metrics.

    PR-AUC is the primary metric because onset events are rare.

    ROC-AUC is included as a secondary diagnostic but should not
    be treated as the headline result.
    """

    return {
        "pr_auc": float(
            average_precision_score(
                y_true,
                risk_score,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                risk_score,
            )
        ),
    }


def calculate_threshold_metrics(
    y_true: np.ndarray,
    risk_score: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate classification metrics at one threshold."""

    prediction = (
        risk_score >= threshold
    ).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "predicted_positive_rows": int(
            prediction.sum()
        ),
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    risk_score: np.ndarray,
) -> tuple[float, dict]:
    """
    Select the threshold that maximises validation F1.

    The final test set must not be used for this decision.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            risk_score,
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Threshold selection failed because no "
            "candidate thresholds were produced."
        )

    usable_precision = precision[:-1]
    usable_recall = recall[:-1]

    denominator = (
        usable_precision
        + usable_recall
    )

    f1_values = np.divide(
        2
        * usable_precision
        * usable_recall,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype="float64",
        ),
        where=denominator > 0,
    )

    best_index = int(
        np.nanargmax(
            f1_values
        )
    )

    best_threshold = float(
        thresholds[
            best_index
        ]
    )

    metrics = calculate_threshold_metrics(
        y_true=y_true,
        risk_score=risk_score,
        threshold=best_threshold,
    )

    return (
        best_threshold,
        metrics,
    )


def calculate_top_k_metrics(
    y_true: np.ndarray,
    risk_score: np.ndarray,
    model_name: str,
) -> list[dict]:
    """
    Calculate operational ranking metrics.

    Top-K recall answers:

        Of all real onset events in validation, how many appear
        among the K highest-risk rows?
    """

    number_of_rows = len(
        y_true
    )

    total_positives = int(
        y_true.sum()
    )

    cutoffs = {
        "top_20": 20,
        "top_50": 50,
        "top_100": 100,
        "top_1_percent": int(
            np.ceil(
                number_of_rows * 0.01
            )
        ),
        "top_5_percent": int(
            np.ceil(
                number_of_rows * 0.05
            )
        ),
    }

    ranking = np.argsort(
        -risk_score,
        kind="stable",
    )

    rows = []

    for cutoff_name, requested_k in (
        cutoffs.items()
    ):
        k = min(
            requested_k,
            number_of_rows,
        )

        selected_indices = ranking[
            :k
        ]

        captured_positives = int(
            y_true[
                selected_indices
            ].sum()
        )

        recall_at_k = (
            captured_positives
            / total_positives
            if total_positives > 0
            else 0.0
        )

        precision_at_k = (
            captured_positives
            / k
            if k > 0
            else 0.0
        )

        rows.append(
            {
                "model": model_name,
                "cutoff": cutoff_name,
                "k_rows": int(k),
                "captured_onsets": (
                    captured_positives
                ),
                "total_validation_onsets": (
                    total_positives
                ),
                "recall_at_k": float(
                    recall_at_k
                ),
                "precision_at_k": float(
                    precision_at_k
                ),
            }
        )

    return rows


def build_time_only_model() -> Pipeline:
    """
    Create a strong schedule-learning baseline.

    Hour and day are one-hot encoded so the model can learn
    distinct risks for specific hours rather than assuming a
    simple linear relationship.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical_time",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
                TIME_CATEGORICAL_FEATURES,
            ),
            (
                "numeric_time",
                StandardScaler(),
                TIME_NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=LOGISTIC_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=LOGISTIC_MAX_ITERATIONS,
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def build_qoe_history_model() -> Pipeline:
    """
    Create the current/historical QoE logistic baseline.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=LOGISTIC_C,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=(
                        LOGISTIC_MAX_ITERATIONS
                    ),
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def extract_coefficients(
    model_name: str,
    model: Pipeline,
) -> list[dict]:
    """Extract standardized logistic coefficients."""

    classifier = model.named_steps[
        "classifier"
    ]

    if model_name == "time_only":
        feature_names = (
            model
            .named_steps[
                "preprocessor"
            ]
            .get_feature_names_out()
        )
    elif model_name == "qoe_history_only":
        feature_names = np.array(
            QOE_HISTORY_ONLY_FEATURES,
            dtype=object,
        )
    else:
        raise ValueError(
            f"Unsupported model name: {model_name}"
        )

    coefficients = classifier.coef_[
        0
    ]

    rows = []

    for feature, coefficient in zip(
        feature_names,
        coefficients,
    ):
        rows.append(
            {
                "model": model_name,
                "feature": str(
                    feature
                ),
                "coefficient": float(
                    coefficient
                ),
                "absolute_coefficient": float(
                    abs(
                        coefficient
                    )
                ),
            }
        )

    return rows


def print_model_results(
    model_name: str,
    ranking_metrics: dict,
    threshold_metrics: dict,
    training_seconds: float,
    iterations: int,
) -> None:
    """Print one model's validation results."""

    print(f"\n{model_name}")

    print(
        "Validation PR-AUC:",
        f"{ranking_metrics['pr_auc']:.6f}",
    )

    print(
        "Validation ROC-AUC:",
        f"{ranking_metrics['roc_auc']:.6f}",
    )

    print(
        "Selected F1 threshold:",
        f"{threshold_metrics['threshold']:.8f}",
    )

    print(
        "Precision:",
        f"{threshold_metrics['precision']:.6f}",
    )

    print(
        "Recall:",
        f"{threshold_metrics['recall']:.6f}",
    )

    print(
        "F1:",
        f"{threshold_metrics['f1']:.6f}",
    )

    print(
        "True positives:",
        threshold_metrics[
            "true_positives"
        ],
    )

    print(
        "False positives:",
        threshold_metrics[
            "false_positives"
        ],
    )

    print(
        "Predicted-positive rows:",
        threshold_metrics[
            "predicted_positive_rows"
        ],
    )

    print(
        "Training time:",
        f"{training_seconds:.4f} seconds",
    )

    print(
        "Logistic iterations:",
        iterations,
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Onset Logistic Baselines")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found:\n"
            f"{DATA_PATH}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_columns = (
        unique_preserving_order(
            CONTROL_COLUMNS
            + TIME_ONLY_FEATURES
            + QOE_HISTORY_ONLY_FEATURES
        )
    )

    print(
        f"\nLoading required hourly columns from:\n"
        f"{DATA_PATH}"
    )

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    df[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        df[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    current_state = pd.to_numeric(
        df[
            CURRENT_STATE_COLUMN
        ],
        errors="coerce",
    )

    next_state = pd.to_numeric(
        df[
            NEXT_STATE_COLUMN
        ],
        errors="coerce",
    )

    onset_candidate_mask = (
        next_state.notna()
        & ~df[
            WARMUP_COLUMN
        ].astype(bool)
        & df[
            ANCHOR_COLUMN
        ].astype(bool)
        & (
            current_state == 0
        )
    )

    onset_df = df[
        onset_candidate_mask
    ].copy()

    onset_df[
        TARGET_COLUMN
    ] = (
        pd.to_numeric(
            onset_df[
                NEXT_STATE_COLUMN
            ],
            errors="raise",
        )
        .astype("int8")
    )

    onset_df[
        "temporal_split"
    ] = assign_temporal_split(
        onset_df
    )

    train_df = onset_df[
        onset_df[
            "temporal_split"
        ] == "train"
    ].copy()

    validation_df = onset_df[
        onset_df[
            "temporal_split"
        ] == "validation"
    ].copy()

    test_df = onset_df[
        onset_df[
            "temporal_split"
        ] == "test"
    ].copy()

    print("\n--- Locked split summary ---")

    for split_name, split_df in [
        (
            "TRAIN",
            train_df,
        ),
        (
            "VALIDATION",
            validation_df,
        ),
        (
            "TEST — WITHHELD",
            test_df,
        ),
    ]:
        positives = int(
            split_df[
                TARGET_COLUMN
            ].sum()
        )

        print(f"\n{split_name}")

        print(
            "Rows:",
            f"{len(split_df):,}",
        )

        print(
            "Positive onsets:",
            f"{positives:,}",
        )

        print(
            "Positive rate:",
            f"{positives / len(split_df) * 100:.4f}%",
        )

    print(
        "\nThe test rows are counted for reconciliation only."
    )

    print(
        "No test score, prediction, threshold, or metric "
        "will be calculated in this script."
    )

    y_train = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype="int8"
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype="int8"
    )

    training_positive_count = int(
        y_train.sum()
    )

    training_negative_count = int(
        len(y_train)
        - training_positive_count
    )

    training_prevalence = float(
        y_train.mean()
    )

    validation_prevalence = float(
        y_validation.mean()
    )

    positive_balanced_weight = (
        len(y_train)
        / (
            2
            * training_positive_count
        )
    )

    negative_balanced_weight = (
        len(y_train)
        / (
            2
            * training_negative_count
        )
    )

    print("\n--- Class imbalance ---")

    print(
        "Training prevalence:",
        f"{training_prevalence:.8f}",
    )

    print(
        "Validation prevalence:",
        f"{validation_prevalence:.8f}",
    )

    print(
        "Balanced positive-class weight:",
        f"{positive_balanced_weight:.4f}",
    )

    print(
        "Balanced negative-class weight:",
        f"{negative_balanced_weight:.4f}",
    )

    print(
        "\nBecause class weighting is used, model outputs "
        "must be described as risk scores rather than "
        "calibrated probabilities."
    )

    # ========================================================
    # BASELINE 0: CONSTANT TRAINING PREVALENCE
    # ========================================================

    constant_validation_score = np.full(
        shape=len(
            validation_df
        ),
        fill_value=training_prevalence,
        dtype="float64",
    )

    constant_metrics = (
        calculate_ranking_metrics(
            y_true=y_validation,
            risk_score=(
                constant_validation_score
            ),
        )
    )

    constant_threshold_metrics = (
        calculate_threshold_metrics(
            y_true=y_validation,
            risk_score=(
                constant_validation_score
            ),
            threshold=0.5,
        )
    )

    print("\n" + "-" * 70)
    print("CONSTANT-PREVALENCE BASELINE")
    print("-" * 70)

    print(
        "Validation PR-AUC:",
        f"{constant_metrics['pr_auc']:.6f}",
    )

    print(
        "Validation prevalence reference:",
        f"{validation_prevalence:.6f}",
    )

    print(
        "Validation ROC-AUC:",
        f"{constant_metrics['roc_auc']:.6f}",
    )

    # ========================================================
    # MODEL 1: TIME-ONLY LOGISTIC
    # ========================================================

    print("\n" + "-" * 70)
    print("TRAINING TIME-ONLY LOGISTIC BASELINE")
    print("-" * 70)

    time_model = build_time_only_model()

    time_start = time.perf_counter()

    time_model.fit(
        train_df[
            TIME_ONLY_FEATURES
        ],
        y_train,
    )

    time_training_seconds = (
        time.perf_counter()
        - time_start
    )

    time_validation_score = (
        time_model.predict_proba(
            validation_df[
                TIME_ONLY_FEATURES
            ]
        )[:, 1]
    )

    time_ranking_metrics = (
        calculate_ranking_metrics(
            y_true=y_validation,
            risk_score=(
                time_validation_score
            ),
        )
    )

    (
        time_best_threshold,
        time_threshold_metrics,
    ) = find_best_f1_threshold(
        y_true=y_validation,
        risk_score=(
            time_validation_score
        ),
    )

    time_iterations = int(
        time_model
        .named_steps[
            "classifier"
        ]
        .n_iter_[0]
    )

    print_model_results(
        model_name=(
            "Time-only logistic"
        ),
        ranking_metrics=(
            time_ranking_metrics
        ),
        threshold_metrics=(
            time_threshold_metrics
        ),
        training_seconds=(
            time_training_seconds
        ),
        iterations=(
            time_iterations
        ),
    )

    # ========================================================
    # MODEL 2: QOE-STATE/HISTORY LOGISTIC
    # ========================================================

    print("\n" + "-" * 70)
    print("TRAINING QOE-STATE/HISTORY LOGISTIC BASELINE")
    print("-" * 70)

    qoe_model = build_qoe_history_model()

    qoe_start = time.perf_counter()

    qoe_model.fit(
        train_df[
            QOE_HISTORY_ONLY_FEATURES
        ],
        y_train,
    )

    qoe_training_seconds = (
        time.perf_counter()
        - qoe_start
    )

    qoe_validation_score = (
        qoe_model.predict_proba(
            validation_df[
                QOE_HISTORY_ONLY_FEATURES
            ]
        )[:, 1]
    )

    qoe_ranking_metrics = (
        calculate_ranking_metrics(
            y_true=y_validation,
            risk_score=(
                qoe_validation_score
            ),
        )
    )

    (
        qoe_best_threshold,
        qoe_threshold_metrics,
    ) = find_best_f1_threshold(
        y_true=y_validation,
        risk_score=(
            qoe_validation_score
        ),
    )

    qoe_iterations = int(
        qoe_model
        .named_steps[
            "classifier"
        ]
        .n_iter_[0]
    )

    print_model_results(
        model_name=(
            "QoE-state/history logistic"
        ),
        ranking_metrics=(
            qoe_ranking_metrics
        ),
        threshold_metrics=(
            qoe_threshold_metrics
        ),
        training_seconds=(
            qoe_training_seconds
        ),
        iterations=(
            qoe_iterations
        ),
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON")
    print("=" * 70)

    print(
        "Constant PR-AUC:",
        f"{constant_metrics['pr_auc']:.6f}",
    )

    print(
        "Time-only PR-AUC:",
        f"{time_ranking_metrics['pr_auc']:.6f}",
    )

    print(
        "QoE-history PR-AUC:",
        f"{qoe_ranking_metrics['pr_auc']:.6f}",
    )

    time_improvement_over_constant = (
        time_ranking_metrics[
            "pr_auc"
        ]
        - constant_metrics[
            "pr_auc"
        ]
    )

    qoe_improvement_over_constant = (
        qoe_ranking_metrics[
            "pr_auc"
        ]
        - constant_metrics[
            "pr_auc"
        ]
    )

    qoe_improvement_over_time = (
        qoe_ranking_metrics[
            "pr_auc"
        ]
        - time_ranking_metrics[
            "pr_auc"
        ]
    )

    print(
        "\nTime-only improvement over constant:",
        f"{time_improvement_over_constant:.6f}",
    )

    print(
        "QoE-history improvement over constant:",
        f"{qoe_improvement_over_constant:.6f}",
    )

    print(
        "QoE-history improvement over time-only:",
        f"{qoe_improvement_over_time:.6f}",
    )

    if (
        time_ranking_metrics[
            "pr_auc"
        ]
        >= qoe_ranking_metrics[
            "pr_auc"
        ]
    ):
        schedule_warning = True

        print(
            "\nWARNING: The time-only model matches or "
            "outperforms the QoE-history model."
        )

        print(
            "This indicates that the simulation schedule may "
            "be a stronger predictor than network-condition "
            "history."
        )
    else:
        schedule_warning = False

        print(
            "\nQoE history outperforms the time-only baseline."
        )

    # ========================================================
    # TOP-K METRICS
    # ========================================================

    top_k_rows = []

    top_k_rows.extend(
        calculate_top_k_metrics(
            y_true=y_validation,
            risk_score=(
                time_validation_score
            ),
            model_name="time_only",
        )
    )

    top_k_rows.extend(
        calculate_top_k_metrics(
            y_true=y_validation,
            risk_score=(
                qoe_validation_score
            ),
            model_name=(
                "qoe_history_only"
            ),
        )
    )

    top_k_df = pd.DataFrame(
        top_k_rows
    )

    print("\n--- Operational Top-K metrics ---")

    print(
        top_k_df.to_string(
            index=False
        )
    )

    # ========================================================
    # COEFFICIENTS
    # ========================================================

    coefficient_rows = []

    coefficient_rows.extend(
        extract_coefficients(
            model_name="time_only",
            model=time_model,
        )
    )

    coefficient_rows.extend(
        extract_coefficients(
            model_name=(
                "qoe_history_only"
            ),
            model=qoe_model,
        )
    )

    coefficient_df = pd.DataFrame(
        coefficient_rows
    )

    coefficient_df = (
        coefficient_df
        .sort_values(
            [
                "model",
                "absolute_coefficient",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print("\n--- Largest absolute coefficients ---")

    for model_name in [
        "time_only",
        "qoe_history_only",
    ]:
        print(f"\n{model_name}")

        subset = coefficient_df[
            coefficient_df[
                "model"
            ] == model_name
        ].head(10)

        for _, row in subset.iterrows():
            print(
                f"{row['feature']:<60}"
                f"{row['coefficient']:>10.4f}"
            )

    # ========================================================
    # VALIDATION SCORE BY HOUR
    # ========================================================

    hourly_score_df = (
        pd.DataFrame(
            {
                "hour_of_day": (
                    validation_df[
                        "hour_of_day"
                    ].to_numpy()
                ),
                "onset_target": (
                    y_validation
                ),
                "time_only_score": (
                    time_validation_score
                ),
                "qoe_history_score": (
                    qoe_validation_score
                ),
            }
        )
        .groupby(
            "hour_of_day",
            as_index=False,
        )
        .agg(
            rows=(
                "onset_target",
                "size",
            ),
            positive_onsets=(
                "onset_target",
                "sum",
            ),
            mean_time_only_score=(
                "time_only_score",
                "mean",
            ),
            mean_qoe_history_score=(
                "qoe_history_score",
                "mean",
            ),
            max_time_only_score=(
                "time_only_score",
                "max",
            ),
            max_qoe_history_score=(
                "qoe_history_score",
                "max",
            ),
        )
        .sort_values(
            "positive_onsets",
            ascending=False,
        )
    )

    print("\n--- Validation score summary by hour ---")

    print(
        hourly_score_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    time_model_path = (
        MODEL_DIRECTORY
        / "time_only_logistic.joblib"
    )

    qoe_model_path = (
        MODEL_DIRECTORY
        / "qoe_history_logistic.joblib"
    )

    joblib.dump(
        time_model,
        time_model_path,
    )

    joblib.dump(
        qoe_model,
        qoe_model_path,
    )

    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    validation_output = (
        validation_df[
            [
                ID_COLUMN,
                SPATIAL_GROUP_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
                "hour_of_day",
            ]
        ]
        .copy()
    )

    validation_output[
        "constant_prevalence_score"
    ] = constant_validation_score

    validation_output[
        "time_only_risk_score"
    ] = time_validation_score

    validation_output[
        "qoe_history_risk_score"
    ] = qoe_validation_score

    validation_output[
        "time_only_prediction_at_selected_threshold"
    ] = (
        time_validation_score
        >= time_best_threshold
    ).astype("int8")

    validation_output[
        "qoe_history_prediction_at_selected_threshold"
    ] = (
        qoe_validation_score
        >= qoe_best_threshold
    ).astype("int8")

    validation_output.to_parquet(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metric_rows = [
        {
            "model": (
                "constant_prevalence"
            ),
            "pr_auc": constant_metrics[
                "pr_auc"
            ],
            "roc_auc": constant_metrics[
                "roc_auc"
            ],
            "selected_threshold": 0.5,
            "precision": (
                constant_threshold_metrics[
                    "precision"
                ]
            ),
            "recall": (
                constant_threshold_metrics[
                    "recall"
                ]
            ),
            "f1": (
                constant_threshold_metrics[
                    "f1"
                ]
            ),
            "training_seconds": 0.0,
        },
        {
            "model": "time_only",
            "pr_auc": (
                time_ranking_metrics[
                    "pr_auc"
                ]
            ),
            "roc_auc": (
                time_ranking_metrics[
                    "roc_auc"
                ]
            ),
            "selected_threshold": (
                time_best_threshold
            ),
            "precision": (
                time_threshold_metrics[
                    "precision"
                ]
            ),
            "recall": (
                time_threshold_metrics[
                    "recall"
                ]
            ),
            "f1": (
                time_threshold_metrics[
                    "f1"
                ]
            ),
            "training_seconds": (
                time_training_seconds
            ),
        },
        {
            "model": (
                "qoe_history_only"
            ),
            "pr_auc": (
                qoe_ranking_metrics[
                    "pr_auc"
                ]
            ),
            "roc_auc": (
                qoe_ranking_metrics[
                    "roc_auc"
                ]
            ),
            "selected_threshold": (
                qoe_best_threshold
            ),
            "precision": (
                qoe_threshold_metrics[
                    "precision"
                ]
            ),
            "recall": (
                qoe_threshold_metrics[
                    "recall"
                ]
            ),
            "f1": (
                qoe_threshold_metrics[
                    "f1"
                ]
            ),
            "training_seconds": (
                qoe_training_seconds
            ),
        },
    ]

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        METRICS_TABLE_PATH,
        index=False,
    )

    top_k_df.to_csv(
        TOP_K_METRICS_PATH,
        index=False,
    )

    coefficient_df.to_csv(
        COEFFICIENTS_PATH,
        index=False,
    )

    hourly_score_df.to_csv(
        HOURLY_SCORE_PATH,
        index=False,
    )

    thresholds = {
        "selected_using": (
            "validation split only"
        ),
        "time_only": {
            "threshold": float(
                time_best_threshold
            ),
            "validation_metrics": (
                time_threshold_metrics
            ),
        },
        "qoe_history_only": {
            "threshold": float(
                qoe_best_threshold
            ),
            "validation_metrics": (
                qoe_threshold_metrics
            ),
        },
        "test_usage": (
            "No test prediction or metric was calculated"
        ),
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            thresholds,
            file,
            indent=4,
        )

    metrics_report = {
        "dataset": (
            "cell_time_dataset.parquet"
        ),
        "data_status": (
            "Ookla-anchored simulated hourly scenario"
        ),
        "task": (
            "Among currently normal anchored cells, "
            "rank next-hour degradation onset risk"
        ),
        "training": {
            "rows": len(
                train_df
            ),
            "positive_onsets": (
                training_positive_count
            ),
            "prevalence": (
                training_prevalence
            ),
        },
        "validation": {
            "rows": len(
                validation_df
            ),
            "positive_onsets": int(
                y_validation.sum()
            ),
            "prevalence": (
                validation_prevalence
            ),
        },
        "test": {
            "rows": len(
                test_df
            ),
            "positive_onsets": int(
                test_df[
                    TARGET_COLUMN
                ].sum()
            ),
            "evaluated": False,
        },
        "logistic_configuration": {
            "C": LOGISTIC_C,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iterations": (
                LOGISTIC_MAX_ITERATIONS
            ),
            "output_interpretation": (
                "risk score, not calibrated probability"
            ),
        },
        "constant_prevalence": {
            "ranking_metrics": (
                constant_metrics
            ),
        },
        "time_only": {
            "features": (
                TIME_ONLY_FEATURES
            ),
            "categorical_encoding": (
                "one-hot encoding for hour, day, "
                "weekend and peak-period fields"
            ),
            "ranking_metrics": (
                time_ranking_metrics
            ),
            "selected_threshold_metrics": (
                time_threshold_metrics
            ),
            "training_seconds": (
                time_training_seconds
            ),
            "iterations": (
                time_iterations
            ),
        },
        "qoe_history_only": {
            "features": (
                QOE_HISTORY_ONLY_FEATURES
            ),
            "ranking_metrics": (
                qoe_ranking_metrics
            ),
            "selected_threshold_metrics": (
                qoe_threshold_metrics
            ),
            "training_seconds": (
                qoe_training_seconds
            ),
            "iterations": (
                qoe_iterations
            ),
        },
        "comparison": {
            "time_pr_auc_improvement_over_constant": float(
                time_improvement_over_constant
            ),
            "qoe_pr_auc_improvement_over_constant": float(
                qoe_improvement_over_constant
            ),
            "qoe_pr_auc_improvement_over_time": float(
                qoe_improvement_over_time
            ),
            "schedule_warning": (
                schedule_warning
            ),
        },
        "test_usage_rule": (
            "The test period remains untouched. "
            "It will be evaluated only after the full "
            "model and threshold are frozen."
        ),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_report,
            file,
            indent=4,
        )

    print("\n--- Saved outputs ---")

    print(
        f"Validation metrics JSON:\n"
        f"{METRICS_PATH}"
    )

    print(
        f"\nValidation metrics CSV:\n"
        f"{METRICS_TABLE_PATH}"
    )

    print(
        f"\nValidation predictions:\n"
        f"{VALIDATION_PREDICTIONS_PATH}"
    )

    print(
        f"\nTop-K metrics:\n"
        f"{TOP_K_METRICS_PATH}"
    )

    print(
        f"\nCoefficients:\n"
        f"{COEFFICIENTS_PATH}"
    )

    print(
        f"\nScores by hour:\n"
        f"{HOURLY_SCORE_PATH}"
    )

    print(
        f"\nSelected thresholds:\n"
        f"{THRESHOLD_PATH}"
    )

    print(
        f"\nSaved models:\n"
        f"{MODEL_DIRECTORY}"
    )

    print(
        "\nOnset logistic baseline training completed."
    )

    print(
        "The final test period remains unevaluated."
    )


if __name__ == "__main__":
    main()