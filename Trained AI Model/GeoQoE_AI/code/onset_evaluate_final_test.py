from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from onset_feature_registry import (
    ANCHOR_COLUMN,
    CONTROL_COLUMNS,
    CURRENT_STATE_COLUMN,
    FULL_CAUSAL_FEATURES,
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
    / "onset_xgboost"
)

VALIDATION_METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_validation_metrics.json"
)

THRESHOLDS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_selected_thresholds.json"
)

TEST_METRICS_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "onset_final_test_metrics.json"
)

TEST_METRICS_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "onset_final_test_metrics.csv"
)

TEST_PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_final_test_predictions.parquet"
)

TEST_PREDICTIONS_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "onset_final_test_predictions.csv"
)

TEST_TOP_K_PATH = (
    OUTPUT_DIRECTORY
    / "onset_final_test_top_k_per_timestamp.csv"
)


# This model was selected using validation PR-AUC before
# any final-test prediction was calculated.
SELECTED_MODEL = "full_causal_xgboost"


MODEL_FEATURE_SETS = {
    "time_only_xgboost": TIME_ONLY_FEATURES,
    "qoe_history_xgboost": (
        QOE_HISTORY_ONLY_FEATURES
    ),
    "full_causal_xgboost": (
        FULL_CAUSAL_FEATURES
    ),
}


def calculate_ranking_metrics(
    y_true: np.ndarray,
    risk_score: np.ndarray,
) -> dict:
    """Calculate threshold-independent final-test metrics."""

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
    """
    Apply the threshold selected previously on validation.

    No threshold is selected or changed using test labels.
    """

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


def calculate_top_k_per_timestamp(
    prediction_df: pd.DataFrame,
    score_column: str,
    model_name: str,
) -> list[dict]:
    """
    Rank cells separately at every test timestamp.

    This represents the operational workflow where an operator
    inspects the highest-risk cells at each hour.
    """

    total_onsets = int(
        prediction_df[
            TARGET_COLUMN
        ].sum()
    )

    cutoffs = {
        "top_20_per_hour": (
            "fixed",
            20,
        ),
        "top_50_per_hour": (
            "fixed",
            50,
        ),
        "top_100_per_hour": (
            "fixed",
            100,
        ),
        "top_1_percent_per_hour": (
            "fraction",
            0.01,
        ),
        "top_5_percent_per_hour": (
            "fraction",
            0.05,
        ),
    }

    result_rows = []

    for (
        cutoff_name,
        (
            cutoff_type,
            cutoff_value,
        ),
    ) in cutoffs.items():
        alerts_issued = 0
        captured_onsets = 0
        timestamps_with_onset = 0
        timestamps_with_captured_onset = 0

        for _, timestamp_df in (
            prediction_df.groupby(
                TIMESTAMP_COLUMN,
                sort=True,
            )
        ):
            timestamp_onsets = int(
                timestamp_df[
                    TARGET_COLUMN
                ].sum()
            )

            if timestamp_onsets > 0:
                timestamps_with_onset += 1

            if cutoff_type == "fixed":
                k = min(
                    int(cutoff_value),
                    len(timestamp_df),
                )
            else:
                k = max(
                    1,
                    int(
                        np.ceil(
                            len(timestamp_df)
                            * float(
                                cutoff_value
                            )
                        )
                    ),
                )

            selected_df = (
                timestamp_df.nlargest(
                    k,
                    score_column,
                )
            )

            selected_onsets = int(
                selected_df[
                    TARGET_COLUMN
                ].sum()
            )

            alerts_issued += len(
                selected_df
            )

            captured_onsets += (
                selected_onsets
            )

            if selected_onsets > 0:
                (
                    timestamps_with_captured_onset
                ) += 1

        recall_at_k = (
            captured_onsets
            / total_onsets
            if total_onsets > 0
            else 0.0
        )

        precision_at_k = (
            captured_onsets
            / alerts_issued
            if alerts_issued > 0
            else 0.0
        )

        result_rows.append(
            {
                "model": model_name,
                "cutoff": cutoff_name,
                "alerts_issued": int(
                    alerts_issued
                ),
                "captured_onsets": int(
                    captured_onsets
                ),
                "total_test_onsets": int(
                    total_onsets
                ),
                "recall": float(
                    recall_at_k
                ),
                "precision": float(
                    precision_at_k
                ),
                "timestamps_with_onset": int(
                    timestamps_with_onset
                ),
                "timestamps_with_captured_onset": int(
                    timestamps_with_captured_onset
                ),
            }
        )

    return result_rows


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — One-Time Final Onset Test Evaluation")
    print("=" * 70)

    # Prevent accidental repeated use of the final test set.
    if TEST_METRICS_JSON_PATH.exists():
        raise FileExistsError(
            "The final test report already exists:\n"
            f"{TEST_METRICS_JSON_PATH}\n\n"
            "Do not repeatedly evaluate the final test set "
            "for model or threshold selection."
        )

    required_paths = [
        DATA_PATH,
        VALIDATION_METRICS_PATH,
        THRESHOLDS_PATH,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{required_path}"
            )

    for model_name in MODEL_FEATURE_SETS:
        model_path = (
            MODEL_DIRECTORY
            / f"{model_name}.json"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found:\n"
                f"{model_path}"
            )

    with open(
        VALIDATION_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        validation_report = json.load(
            file
        )

    with open(
        THRESHOLDS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        threshold_report = json.load(
            file
        )

    recorded_selected_model = (
        validation_report[
            "provisional_highest_pr_auc_model"
        ]
    )

    print("\n--- Frozen model-selection check ---")

    print(
        "Model selected before test:",
        SELECTED_MODEL,
    )

    print(
        "Highest validation PR-AUC model:",
        recorded_selected_model,
    )

    if recorded_selected_model != SELECTED_MODEL:
        raise ValueError(
            "The manually frozen model does not match "
            "the saved validation winner."
        )

    all_features = unique_preserving_order(
        TIME_ONLY_FEATURES
        + QOE_HISTORY_ONLY_FEATURES
        + FULL_CAUSAL_FEATURES
    )

    required_columns = (
        unique_preserving_order(
            CONTROL_COLUMNS
            + all_features
        )
    )

    print(
        f"\nLoading final-test source columns from:\n"
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

    test_df = onset_df[
        onset_df[
            "temporal_split"
        ] == "test"
    ].copy()

    test_df = test_df.reset_index(
        drop=True
    )

    y_test = test_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype="int8"
    )

    test_prevalence = float(
        y_test.mean()
    )

    print("\n--- Final test dataset ---")

    print(
        "Rows:",
        f"{len(test_df):,}",
    )

    print(
        "Positive onsets:",
        f"{int(y_test.sum()):,}",
    )

    print(
        "Positive rate:",
        f"{test_prevalence * 100:.4f}%",
    )

    print(
        "Unique timestamps:",
        test_df[
            TIMESTAMP_COLUMN
        ].nunique(),
    )

    print(
        "Period:",
        test_df[
            TIMESTAMP_COLUMN
        ].min(),
        "to",
        test_df[
            TIMESTAMP_COLUMN
        ].max(),
    )

    prediction_output = test_df[
        [
            ID_COLUMN,
            SPATIAL_GROUP_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            "hour_of_day",
        ]
    ].copy()

    metric_rows = []
    top_k_rows = []
    detailed_results = {}

    for model_name, features in (
        MODEL_FEATURE_SETS.items()
    ):
        print("\n" + "-" * 70)
        print(
            f"EVALUATING FROZEN {model_name.upper()}"
        )
        print("-" * 70)

        model_path = (
            MODEL_DIRECTORY
            / f"{model_name}.json"
        )

        model = xgb.XGBClassifier()

        model.load_model(
            str(model_path)
        )

        best_iteration = int(
            validation_report[
                "xgboost_results"
            ][
                model_name
            ][
                "best_iteration"
            ]
        )

        frozen_threshold = float(
            threshold_report[
                model_name
            ][
                "threshold"
            ]
        )

        X_test = test_df[
            features
        ].astype("float32")

        risk_score = model.predict_proba(
            X_test,
            iteration_range=(
                0,
                best_iteration + 1,
            ),
        )[:, 1]

        ranking_metrics = (
            calculate_ranking_metrics(
                y_true=y_test,
                risk_score=risk_score,
            )
        )

        threshold_metrics = (
            calculate_threshold_metrics(
                y_true=y_test,
                risk_score=risk_score,
                threshold=(
                    frozen_threshold
                ),
            )
        )

        validation_pr_auc = float(
            validation_report[
                "xgboost_results"
            ][
                model_name
            ][
                "ranking_metrics"
            ][
                "pr_auc"
            ]
        )

        pr_auc_difference = (
            ranking_metrics[
                "pr_auc"
            ]
            - validation_pr_auc
        )

        print(
            "Frozen best iteration:",
            best_iteration,
        )

        print(
            "Frozen validation threshold:",
            f"{frozen_threshold:.8f}",
        )

        print(
            "Test PR-AUC:",
            f"{ranking_metrics['pr_auc']:.6f}",
        )

        print(
            "Test PR-AUC lift over prevalence:",
            f"{ranking_metrics['pr_auc'] / test_prevalence:.2f}x",
        )

        print(
            "Test ROC-AUC:",
            f"{ranking_metrics['roc_auc']:.6f}",
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
            "Test minus validation PR-AUC:",
            f"{pr_auc_difference:.6f}",
        )

        score_column = (
            f"{model_name}_risk_score"
        )

        prediction_column = (
            f"{model_name}_prediction_"
            f"at_frozen_threshold"
        )

        prediction_output[
            score_column
        ] = risk_score

        prediction_output[
            prediction_column
        ] = (
            risk_score
            >= frozen_threshold
        ).astype("int8")

        model_top_k_rows = (
            calculate_top_k_per_timestamp(
                prediction_df=(
                    prediction_output
                ),
                score_column=score_column,
                model_name=model_name,
            )
        )

        top_k_rows.extend(
            model_top_k_rows
        )

        metric_rows.append(
            {
                "model": model_name,
                "selected_model": (
                    model_name
                    == SELECTED_MODEL
                ),
                "feature_count": len(
                    features
                ),
                "best_iteration": (
                    best_iteration
                ),
                "frozen_validation_threshold": (
                    frozen_threshold
                ),
                "validation_pr_auc": (
                    validation_pr_auc
                ),
                "test_prevalence": (
                    test_prevalence
                ),
                "test_pr_auc": (
                    ranking_metrics[
                        "pr_auc"
                    ]
                ),
                "test_pr_auc_lift_over_prevalence": (
                    ranking_metrics[
                        "pr_auc"
                    ]
                    / test_prevalence
                ),
                "test_minus_validation_pr_auc": (
                    pr_auc_difference
                ),
                "test_roc_auc": (
                    ranking_metrics[
                        "roc_auc"
                    ]
                ),
                "test_precision": (
                    threshold_metrics[
                        "precision"
                    ]
                ),
                "test_recall": (
                    threshold_metrics[
                        "recall"
                    ]
                ),
                "test_f1": (
                    threshold_metrics[
                        "f1"
                    ]
                ),
                "test_true_positives": (
                    threshold_metrics[
                        "true_positives"
                    ]
                ),
                "test_false_positives": (
                    threshold_metrics[
                        "false_positives"
                    ]
                ),
                "test_false_negatives": (
                    threshold_metrics[
                        "false_negatives"
                    ]
                ),
                "test_predicted_positive_rows": (
                    threshold_metrics[
                        "predicted_positive_rows"
                    ]
                ),
            }
        )

        detailed_results[
            model_name
        ] = {
            "selected_model": (
                model_name
                == SELECTED_MODEL
            ),
            "features": features,
            "best_iteration": (
                best_iteration
            ),
            "frozen_validation_threshold": (
                frozen_threshold
            ),
            "validation_pr_auc": (
                validation_pr_auc
            ),
            "test_ranking_metrics": (
                ranking_metrics
            ),
            "test_threshold_metrics": (
                threshold_metrics
            ),
            "test_minus_validation_pr_auc": float(
                pr_auc_difference
            ),
        }

    metrics_df = pd.DataFrame(
        metric_rows
    )

    top_k_df = pd.DataFrame(
        top_k_rows
    )

    print("\n" + "=" * 70)
    print("FINAL TEST MODEL COMPARISON")
    print("=" * 70)

    print(
        metrics_df[
            [
                "model",
                "selected_model",
                "test_pr_auc",
                "test_roc_auc",
                "test_precision",
                "test_recall",
                "test_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FINAL TEST PER-TIMESTAMP TOP-K")
    print("=" * 70)

    print(
        top_k_df.to_string(
            index=False
        )
    )

    selected_row = metrics_df[
        metrics_df[
            "model"
        ] == SELECTED_MODEL
    ].iloc[0]

    selected_top_k = (
        top_k_df[
            top_k_df[
                "model"
            ] == SELECTED_MODEL
        ]
        .copy()
    )

    print("\n--- Frozen selected-model result ---")

    print(
        "Selected model:",
        SELECTED_MODEL,
    )

    print(
        "Test PR-AUC:",
        f"{selected_row['test_pr_auc']:.6f}",
    )

    print(
        "Test PR-AUC lift over prevalence:",
        f"{selected_row['test_pr_auc_lift_over_prevalence']:.2f}x",
    )

    print(
        "Test precision:",
        f"{selected_row['test_precision']:.6f}",
    )

    print(
        "Test recall:",
        f"{selected_row['test_recall']:.6f}",
    )

    print(
        "Test F1:",
        f"{selected_row['test_f1']:.6f}",
    )

    print("\nSelected-model Top-K:")

    print(
        selected_top_k.to_string(
            index=False
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        TEST_METRICS_CSV_PATH,
        index=False,
    )

    top_k_df.to_csv(
        TEST_TOP_K_PATH,
        index=False,
    )

    prediction_output.to_parquet(
        TEST_PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    prediction_output.to_csv(
        TEST_PREDICTIONS_CSV_PATH,
        index=False,
    )

    final_report = {
        "dataset": (
            "cell_time_dataset.parquet"
        ),
        "data_status": (
            "Ookla-anchored simulated hourly scenario"
        ),
        "task": (
            "Among anchored currently normal cells, "
            "rank next-hour degradation onset risk"
        ),
        "selection_protocol": {
            "selected_model": (
                SELECTED_MODEL
            ),
            "selection_basis": (
                "Highest validation PR-AUC before "
                "final-test evaluation"
            ),
            "model_switching_after_test_allowed": False,
            "threshold_selection": (
                "Frozen validation maximum-F1 threshold"
            ),
        },
        "test_dataset": {
            "rows": len(
                test_df
            ),
            "positive_onsets": int(
                y_test.sum()
            ),
            "prevalence": (
                test_prevalence
            ),
            "timestamps": int(
                test_df[
                    TIMESTAMP_COLUMN
                ].nunique()
            ),
            "first_timestamp": str(
                test_df[
                    TIMESTAMP_COLUMN
                ].min()
            ),
            "last_timestamp": str(
                test_df[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
        "model_results": (
            detailed_results
        ),
        "selected_model_result": (
            detailed_results[
                SELECTED_MODEL
            ]
        ),
        "selected_model_top_k": (
            selected_top_k.to_dict(
                orient="records"
            )
        ),
        "score_interpretation": (
            "Class-weighted model outputs are relative "
            "onset-risk scores, not calibrated probabilities."
        ),
        "important_limitation": (
            "The hourly QoE panel and onset labels are "
            "simulated. Final-test performance measures "
            "generalisation within the simulation and does "
            "not validate performance on observed hourly "
            "telecom telemetry."
        ),
    }

    with open(
        TEST_METRICS_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_report,
            file,
            indent=4,
        )

    print("\n--- Saved final-test outputs ---")

    print(
        f"Metrics JSON:\n"
        f"{TEST_METRICS_JSON_PATH}"
    )

    print(
        f"\nMetrics CSV:\n"
        f"{TEST_METRICS_CSV_PATH}"
    )

    print(
        f"\nTest predictions Parquet:\n"
        f"{TEST_PREDICTIONS_PATH}"
    )

    print(
        f"\nTest predictions CSV:\n"
        f"{TEST_PREDICTIONS_CSV_PATH}"
    )

    print(
        f"\nPer-timestamp Top-K metrics:\n"
        f"{TEST_TOP_K_PATH}"
    )

    print(
        "\nOne-time final test evaluation completed."
    )

    print(
        "Do not use these test results to change the "
        "selected model, features, hyperparameters, or threshold."
    )


if __name__ == "__main__":
    main()