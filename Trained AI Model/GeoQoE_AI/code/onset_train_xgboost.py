from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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

LOGISTIC_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
    / "onset_logistic_validation_metrics.json"
)

LOGISTIC_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "onset_model"
    / "onset_logistic_validation_predictions.parquet"
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

METRICS_JSON_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_validation_metrics.json"
)

METRICS_CSV_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_validation_metrics.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_validation_predictions.parquet"
)

FEATURE_IMPORTANCE_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_feature_importance.csv"
)

TOP_K_PER_TIMESTAMP_PATH = (
    OUTPUT_DIRECTORY
    / "onset_validation_top_k_per_timestamp.csv"
)

THRESHOLDS_PATH = (
    OUTPUT_DIRECTORY
    / "onset_xgboost_selected_thresholds.json"
)


RANDOM_SEED = 42


MODEL_FEATURE_SETS = {
    "time_only_xgboost": TIME_ONLY_FEATURES,
    "qoe_history_xgboost": QOE_HISTORY_ONLY_FEATURES,
    "full_causal_xgboost": FULL_CAUSAL_FEATURES,
}


BASE_XGBOOST_PARAMETERS = {
    "n_estimators": 1000,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "gamma": 0.10,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "max_delta_step": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "importance_type": "gain",
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def calculate_ranking_metrics(
    y_true: np.ndarray,
    risk_score: np.ndarray,
) -> dict:
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
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            risk_score,
        )
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No thresholds were produced."
        )

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_values = np.divide(
        2 * precision * recall,
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

    threshold = float(
        thresholds[best_index]
    )

    metrics = calculate_threshold_metrics(
        y_true=y_true,
        risk_score=risk_score,
        threshold=threshold,
    )

    return threshold, metrics


def calculate_top_k_per_timestamp(
    prediction_df: pd.DataFrame,
    score_column: str,
    model_name: str,
) -> list[dict]:
    """
    Rank cells separately within every predictor timestamp.

    This is closer to the operational question:
    Which cells should the operator inspect at this hour?
    """

    total_onsets = int(
        prediction_df[
            TARGET_COLUMN
        ].sum()
    )

    cutoff_definitions = {
        "top_20_per_hour": ("fixed", 20),
        "top_50_per_hour": ("fixed", 50),
        "top_100_per_hour": ("fixed", 100),
        "top_1_percent_per_hour": (
            "fraction",
            0.01,
        ),
        "top_5_percent_per_hour": (
            "fraction",
            0.05,
        ),
    }

    output_rows = []

    grouped = prediction_df.groupby(
        TIMESTAMP_COLUMN,
        sort=True,
    )

    for (
        cutoff_name,
        (cutoff_type, cutoff_value),
    ) in cutoff_definitions.items():
        alerts_issued = 0
        captured_onsets = 0
        timestamps_with_onset = 0
        timestamps_with_captured_onset = 0

        for _, timestamp_df in grouped:
            timestamp_positive_count = int(
                timestamp_df[
                    TARGET_COLUMN
                ].sum()
            )

            if timestamp_positive_count > 0:
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
                            * float(cutoff_value)
                        )
                    ),
                )

            selected_df = timestamp_df.nlargest(
                k,
                score_column,
            )

            selected_positive_count = int(
                selected_df[
                    TARGET_COLUMN
                ].sum()
            )

            alerts_issued += len(
                selected_df
            )

            captured_onsets += (
                selected_positive_count
            )

            if selected_positive_count > 0:
                timestamps_with_captured_onset += 1

        recall = (
            captured_onsets / total_onsets
            if total_onsets > 0
            else 0.0
        )

        precision = (
            captured_onsets / alerts_issued
            if alerts_issued > 0
            else 0.0
        )

        output_rows.append(
            {
                "model": model_name,
                "cutoff": cutoff_name,
                "alerts_issued": int(
                    alerts_issued
                ),
                "captured_onsets": int(
                    captured_onsets
                ),
                "total_validation_onsets": (
                    total_onsets
                ),
                "recall": float(recall),
                "precision": float(
                    precision
                ),
                "timestamps_with_onset": int(
                    timestamps_with_onset
                ),
                "timestamps_with_captured_onset": int(
                    timestamps_with_captured_onset
                ),
            }
        )

    return output_rows


def print_model_results(
    model_name: str,
    ranking_metrics: dict,
    threshold_metrics: dict,
    best_iteration: int,
    training_seconds: float,
) -> None:
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
        "Best boosting iteration:",
        best_iteration,
    )

    print(
        "Training time:",
        f"{training_seconds:.4f} seconds",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Onset XGBoost Validation")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Hourly dataset not found:\n"
            f"{DATA_PATH}"
        )

    if not LOGISTIC_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Logistic metrics not found. Run "
            "onset_train_logistic_baselines.py first."
        )

    if not LOGISTIC_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Logistic validation predictions not found."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_model_features = (
        unique_preserving_order(
            TIME_ONLY_FEATURES
            + QOE_HISTORY_ONLY_FEATURES
            + FULL_CAUSAL_FEATURES
        )
    )

    required_columns = (
        unique_preserving_order(
            CONTROL_COLUMNS
            + all_model_features
        )
    )

    print(
        f"\nLoading selected hourly columns from:\n"
        f"{DATA_PATH}"
    )

    df = pd.read_parquet(
        DATA_PATH,
        columns=required_columns,
    )

    df[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="raise",
    )

    current_state = pd.to_numeric(
        df[CURRENT_STATE_COLUMN],
        errors="coerce",
    )

    next_state = pd.to_numeric(
        df[NEXT_STATE_COLUMN],
        errors="coerce",
    )

    onset_mask = (
        next_state.notna()
        & ~df[WARMUP_COLUMN].astype(bool)
        & df[ANCHOR_COLUMN].astype(bool)
        & (current_state == 0)
    )

    onset_df = df[
        onset_mask
    ].copy()

    onset_df[
        TARGET_COLUMN
    ] = (
        pd.to_numeric(
            onset_df[NEXT_STATE_COLUMN],
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
        onset_df["temporal_split"]
        == "train"
    ].copy()

    validation_df = onset_df[
        onset_df["temporal_split"]
        == "validation"
    ].copy()

    test_df = onset_df[
        onset_df["temporal_split"]
        == "test"
    ].copy()

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

    training_positives = int(
        y_train.sum()
    )

    training_negatives = int(
        len(y_train)
        - training_positives
    )

    scale_pos_weight = (
        training_negatives
        / training_positives
    )

    validation_prevalence = float(
        y_validation.mean()
    )

    print("\n--- Locked split summary ---")

    print(
        "Training rows:",
        f"{len(train_df):,}",
    )

    print(
        "Training onsets:",
        f"{training_positives:,}",
    )

    print(
        "Validation rows:",
        f"{len(validation_df):,}",
    )

    print(
        "Validation onsets:",
        f"{int(y_validation.sum()):,}",
    )

    print(
        "Withheld test rows:",
        f"{len(test_df):,}",
    )

    print(
        "Withheld test onsets:",
        f"{int(test_df[TARGET_COLUMN].sum()):,}",
    )

    print(
        "XGBoost scale_pos_weight:",
        f"{scale_pos_weight:.4f}",
    )

    print(
        "Validation prevalence:",
        f"{validation_prevalence:.8f}",
    )

    print(
        "\nThe test split will not be predicted or evaluated."
    )

    validation_output = validation_df[
        [
            ID_COLUMN,
            SPATIAL_GROUP_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            "hour_of_day",
        ]
    ].copy()

    metrics_rows = []
    feature_importance_rows = []
    threshold_output = {}
    xgboost_results = {}

    total_start_time = time.perf_counter()

    for model_name, features in (
        MODEL_FEATURE_SETS.items()
    ):
        print("\n" + "-" * 70)
        print(
            f"TRAINING {model_name.upper()}"
        )
        print("-" * 70)

        X_train = train_df[
            features
        ].astype("float32")

        X_validation = validation_df[
            features
        ].astype("float32")

        parameters = {
            **BASE_XGBOOST_PARAMETERS,
            "scale_pos_weight": (
                scale_pos_weight
            ),
        }

        model = xgb.XGBClassifier(
            **parameters
        )

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],
            verbose=False,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        risk_score = model.predict_proba(
            X_validation
        )[:, 1]

        ranking_metrics = (
            calculate_ranking_metrics(
                y_true=y_validation,
                risk_score=risk_score,
            )
        )

        (
            selected_threshold,
            threshold_metrics,
        ) = find_best_f1_threshold(
            y_true=y_validation,
            risk_score=risk_score,
        )

        best_iteration = int(
            getattr(
                model,
                "best_iteration",
                BASE_XGBOOST_PARAMETERS[
                    "n_estimators"
                ] - 1,
            )
        )

        print_model_results(
            model_name=model_name,
            ranking_metrics=ranking_metrics,
            threshold_metrics=(
                threshold_metrics
            ),
            best_iteration=best_iteration,
            training_seconds=training_seconds,
        )

        score_column = (
            model_name
            + "_risk_score"
        )

        prediction_column = (
            model_name
            + "_prediction_at_selected_threshold"
        )

        validation_output[
            score_column
        ] = risk_score

        validation_output[
            prediction_column
        ] = (
            risk_score
            >= selected_threshold
        ).astype("int8")

        metrics_rows.append(
            {
                "model": model_name,
                "feature_count": len(
                    features
                ),
                "pr_auc": ranking_metrics[
                    "pr_auc"
                ],
                "pr_auc_lift_over_prevalence": (
                    ranking_metrics[
                        "pr_auc"
                    ]
                    / validation_prevalence
                ),
                "roc_auc": ranking_metrics[
                    "roc_auc"
                ],
                "selected_threshold": (
                    selected_threshold
                ),
                "precision": (
                    threshold_metrics[
                        "precision"
                    ]
                ),
                "recall": (
                    threshold_metrics[
                        "recall"
                    ]
                ),
                "f1": (
                    threshold_metrics[
                        "f1"
                    ]
                ),
                "true_positives": (
                    threshold_metrics[
                        "true_positives"
                    ]
                ),
                "false_positives": (
                    threshold_metrics[
                        "false_positives"
                    ]
                ),
                "predicted_positive_rows": (
                    threshold_metrics[
                        "predicted_positive_rows"
                    ]
                ),
                "best_iteration": (
                    best_iteration
                ),
                "training_seconds": (
                    training_seconds
                ),
            }
        )

        threshold_output[
            model_name
        ] = {
            "threshold": float(
                selected_threshold
            ),
            "selected_using": (
                "validation maximum F1"
            ),
            "validation_metrics": (
                threshold_metrics
            ),
        }

        xgboost_results[
            model_name
        ] = {
            "features": features,
            "ranking_metrics": (
                ranking_metrics
            ),
            "threshold_metrics": (
                threshold_metrics
            ),
            "best_iteration": (
                best_iteration
            ),
            "training_seconds": (
                training_seconds
            ),
        }

        importances = (
            model.feature_importances_
        )

        for feature, importance in zip(
            features,
            importances,
        ):
            feature_importance_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "gain_importance": float(
                        importance
                    ),
                }
            )

        model_path = (
            MODEL_DIRECTORY
            / f"{model_name}.json"
        )

        model.save_model(
            str(model_path)
        )

        print(
            "Saved model:",
            model_path.name,
        )

    total_training_seconds = (
        time.perf_counter()
        - total_start_time
    )

    # --------------------------------------------------------
    # Add the earlier logistic scores for direct comparison.
    # --------------------------------------------------------

    logistic_predictions = (
        pd.read_parquet(
            LOGISTIC_PREDICTIONS_PATH
        )
    )

    logistic_subset = (
        logistic_predictions[
            [
                ID_COLUMN,
                TIMESTAMP_COLUMN,
                "time_only_risk_score",
                "qoe_history_risk_score",
            ]
        ]
        .copy()
    )

    validation_output = (
        validation_output.merge(
            logistic_subset,
            on=[
                ID_COLUMN,
                TIMESTAMP_COLUMN,
            ],
            how="left",
            validate="one_to_one",
        )
    )

    if validation_output[
        [
            "time_only_risk_score",
            "qoe_history_risk_score",
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing logistic scores after merge."
        )

    # --------------------------------------------------------
    # Operational Top-K ranking, separately per timestamp.
    # --------------------------------------------------------

    top_k_rows = []

    score_columns = {
        "time_only_logistic": (
            "time_only_risk_score"
        ),
        "qoe_history_logistic": (
            "qoe_history_risk_score"
        ),
        "time_only_xgboost": (
            "time_only_xgboost_risk_score"
        ),
        "qoe_history_xgboost": (
            "qoe_history_xgboost_risk_score"
        ),
        "full_causal_xgboost": (
            "full_causal_xgboost_risk_score"
        ),
    }

    for model_name, score_column in (
        score_columns.items()
    ):
        top_k_rows.extend(
            calculate_top_k_per_timestamp(
                prediction_df=(
                    validation_output
                ),
                score_column=score_column,
                model_name=model_name,
            )
        )

    top_k_df = pd.DataFrame(
        top_k_rows
    )

    print("\n" + "=" * 70)
    print("PER-TIMESTAMP TOP-K VALIDATION RESULTS")
    print("=" * 70)

    print(
        top_k_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compare all logistic and XGBoost candidates.
    # --------------------------------------------------------

    with open(
        LOGISTIC_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        logistic_metrics = json.load(
            file
        )

    candidate_pr_auc = {
        "constant_prevalence": float(
            logistic_metrics[
                "constant_prevalence"
            ][
                "ranking_metrics"
            ][
                "pr_auc"
            ]
        ),
        "time_only_logistic": float(
            logistic_metrics[
                "time_only"
            ][
                "ranking_metrics"
            ][
                "pr_auc"
            ]
        ),
        "qoe_history_logistic": float(
            logistic_metrics[
                "qoe_history_only"
            ][
                "ranking_metrics"
            ][
                "pr_auc"
            ]
        ),
    }

    for model_name, result in (
        xgboost_results.items()
    ):
        candidate_pr_auc[
            model_name
        ] = float(
            result[
                "ranking_metrics"
            ][
                "pr_auc"
            ]
        )

    ranked_candidates = sorted(
        candidate_pr_auc.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    provisional_winner = (
        ranked_candidates[0][0]
    )

    print("\n" + "=" * 70)
    print("VALIDATION MODEL RANKING")
    print("=" * 70)

    for position, (
        model_name,
        pr_auc,
    ) in enumerate(
        ranked_candidates,
        start=1,
    ):
        print(
            f"{position:>2}. "
            f"{model_name:<28}"
            f"PR-AUC={pr_auc:.6f}"
        )

    print(
        "\nProvisional highest validation PR-AUC model:",
        provisional_winner,
    )

    full_vs_time_difference = (
        xgboost_results[
            "full_causal_xgboost"
        ][
            "ranking_metrics"
        ][
            "pr_auc"
        ]
        - xgboost_results[
            "time_only_xgboost"
        ][
            "ranking_metrics"
        ][
            "pr_auc"
        ]
    )

    full_vs_qoe_difference = (
        xgboost_results[
            "full_causal_xgboost"
        ][
            "ranking_metrics"
        ][
            "pr_auc"
        ]
        - xgboost_results[
            "qoe_history_xgboost"
        ][
            "ranking_metrics"
        ][
            "pr_auc"
        ]
    )

    print(
        "Full causal PR-AUC minus time-only XGBoost:",
        f"{full_vs_time_difference:.6f}",
    )

    print(
        "Full causal PR-AUC minus QoE-history XGBoost:",
        f"{full_vs_qoe_difference:.6f}",
    )

    # --------------------------------------------------------
    # Print full-model feature importance.
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        feature_importance_rows
    )

    importance_df = (
        importance_df.sort_values(
            [
                "model",
                "gain_importance",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        "\n--- Full causal XGBoost top features ---"
    )

    full_importance = (
        importance_df[
            importance_df["model"]
            == "full_causal_xgboost"
        ]
        .head(15)
    )

    for position, row in enumerate(
        full_importance.itertuples(
            index=False
        ),
        start=1,
    ):
        print(
            f"{position:>2}. "
            f"{row.feature:<42}"
            f"{row.gain_importance:>10.6f}"
        )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        METRICS_CSV_PATH,
        index=False,
    )

    validation_output.to_parquet(
        PREDICTIONS_PATH,
        index=False,
        compression="zstd",
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    top_k_df.to_csv(
        TOP_K_PER_TIMESTAMP_PATH,
        index=False,
    )

    threshold_output[
        "test_usage"
    ] = (
        "No test score or metric was calculated"
    )

    with open(
        THRESHOLDS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_output,
            file,
            indent=4,
        )

    report = {
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
        "training_rows": len(
            train_df
        ),
        "training_onsets": (
            training_positives
        ),
        "validation_rows": len(
            validation_df
        ),
        "validation_onsets": int(
            y_validation.sum()
        ),
        "validation_prevalence": (
            validation_prevalence
        ),
        "test_rows": len(
            test_df
        ),
        "test_onsets": int(
            test_df[
                TARGET_COLUMN
            ].sum()
        ),
        "test_evaluated": False,
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "base_parameters": (
            BASE_XGBOOST_PARAMETERS
        ),
        "xgboost_results": (
            xgboost_results
        ),
        "all_candidate_validation_pr_auc": (
            candidate_pr_auc
        ),
        "provisional_highest_pr_auc_model": (
            provisional_winner
        ),
        "full_causal_pr_auc_minus_time_only_xgboost": float(
            full_vs_time_difference
        ),
        "full_causal_pr_auc_minus_qoe_history_xgboost": float(
            full_vs_qoe_difference
        ),
        "total_training_seconds": float(
            total_training_seconds
        ),
        "score_interpretation": (
            "Class-weighted model output is an onset "
            "risk score, not a calibrated probability."
        ),
        "test_usage_rule": (
            "The test split remains untouched until "
            "model and threshold selection are frozen."
        ),
    }

    with open(
        METRICS_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print("\n--- Saved outputs ---")

    print(
        f"Validation metrics JSON:\n"
        f"{METRICS_JSON_PATH}"
    )

    print(
        f"\nValidation metrics CSV:\n"
        f"{METRICS_CSV_PATH}"
    )

    print(
        f"\nValidation predictions:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"\nFeature importance:\n"
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    print(
        f"\nPer-timestamp Top-K metrics:\n"
        f"{TOP_K_PER_TIMESTAMP_PATH}"
    )

    print(
        f"\nSelected thresholds:\n"
        f"{THRESHOLDS_PATH}"
    )

    print(
        f"\nSaved XGBoost models:\n"
        f"{MODEL_DIRECTORY}"
    )

    print(
        "\nOnset XGBoost validation completed."
    )

    print(
        "The final test period remains unevaluated."
    )


if __name__ == "__main__":
    main()