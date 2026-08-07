from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from feature_registry import TARGET_COLUMN
from temporal_split import create_temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "forecast_dataset_development.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "forecast_readiness_report.json"
)

TRANSITION_PATH = (
    OUTPUT_DIRECTORY
    / "forecast_readiness_transitions.csv"
)


def analyse_transitions(
    name: str,
    split_df: pd.DataFrame,
) -> dict:
    """
    Analyse changes between the current-hour degradation state
    and the next-hour degradation state.
    """

    current_state = split_df[
        "target_is_degraded"
    ].astype("int8")

    next_state = split_df[
        TARGET_COLUMN
    ].astype("int8")

    normal_to_normal = int(
        (
            (current_state == 0)
            & (next_state == 0)
        ).sum()
    )

    normal_to_degraded = int(
        (
            (current_state == 0)
            & (next_state == 1)
        ).sum()
    )

    degraded_to_normal = int(
        (
            (current_state == 1)
            & (next_state == 0)
        ).sum()
    )

    degraded_to_degraded = int(
        (
            (current_state == 1)
            & (next_state == 1)
        ).sum()
    )

    same_state_rate = float(
        (current_state == next_state).mean()
    )

    # Persistence baseline:
    # predict that the next hour will have the same state
    # as the current hour.
    persistence_prediction = current_state

    persistence_metrics = {
        "accuracy": float(
            accuracy_score(
                next_state,
                persistence_prediction,
            )
        ),
        "precision": float(
            precision_score(
                next_state,
                persistence_prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                next_state,
                persistence_prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                next_state,
                persistence_prediction,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                next_state,
                persistence_prediction,
            )
        ),
    }

    return {
        "split": name,
        "rows": int(len(split_df)),
        "positive_rate": float(
            next_state.mean()
        ),
        "normal_to_normal": normal_to_normal,
        "normal_to_degraded": normal_to_degraded,
        "degraded_to_normal": degraded_to_normal,
        "degraded_to_degraded": degraded_to_degraded,
        "same_state_rate": same_state_rate,
        "persistence_baseline": persistence_metrics,
    }


def print_analysis(
    result: dict,
) -> None:
    print(f"\n--- {result['split'].upper()} ---")

    print(
        "Rows:",
        f"{result['rows']:,}",
    )

    print(
        "Next-hour degradation rate:",
        f"{result['positive_rate'] * 100:.4f}%",
    )

    print("\nState transitions:")

    print(
        "Normal → Normal:",
        f"{result['normal_to_normal']:,}",
    )

    print(
        "Normal → Degraded:",
        f"{result['normal_to_degraded']:,}",
    )

    print(
        "Degraded → Normal:",
        f"{result['degraded_to_normal']:,}",
    )

    print(
        "Degraded → Degraded:",
        f"{result['degraded_to_degraded']:,}",
    )

    print(
        "Same-state rate:",
        f"{result['same_state_rate'] * 100:.4f}%",
    )

    persistence = result[
        "persistence_baseline"
    ]

    print("\nPersistence baseline:")

    print(
        "Precision:",
        f"{persistence['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{persistence['recall']:.4f}",
    )

    print(
        "F1:",
        f"{persistence['f1']:.4f}",
    )

    print(
        "PR-AUC:",
        f"{persistence['pr_auc']:.4f}",
    )


def main() -> None:
    print("=" * 70)
    print("GeoQoE Pulse — Forecasting Readiness Check")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Forecast dataset not found:\n{DATA_PATH}"
        )

    columns = [
        "h3_index",
        "timestamp",
        "forecast_timestamp",
        "in_warmup_window",
        "target_is_degraded",
        TARGET_COLUMN,
        "dl_median",
        "rtt_median",
        "qoe_is_synthetic",
    ]

    print(f"\nLoading data from:\n{DATA_PATH}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=columns,
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
        "Training rows analysed:",
        f"{len(train_df):,}",
    )

    print(
        "Validation rows analysed:",
        f"{len(validation_df):,}",
    )

    print(
        "Test rows kept untouched:",
        f"{len(test_df):,}",
    )

    training_result = analyse_transitions(
        "training",
        train_df,
    )

    validation_result = analyse_transitions(
        "validation",
        validation_df,
    )

    print_analysis(training_result)
    print_analysis(validation_result)

    # Check whether the current target exactly matches
    # the documented threshold rule.
    validation_threshold_rule = (
        (validation_df["dl_median"] < 10.0)
        | (validation_df["rtt_median"] > 100.0)
    ).astype("int8")

    validation_rule_match_rate = float(
        (
            validation_threshold_rule
            == validation_df["target_is_degraded"]
        ).mean()
    )

    synthetic_rate = float(
        validation_df["qoe_is_synthetic"].mean()
    )

    print("\n--- Additional data checks ---")

    print(
        "Current-label threshold-rule match:",
        f"{validation_rule_match_rate * 100:.4f}%",
    )

    print(
        "Validation rows marked as synthetic QoE:",
        f"{synthetic_rate * 100:.4f}%",
    )

    failures = []
    warnings = []

    # A forecasting validation period should contain both:
    # 1. degradation onsets; and
    # 2. recoveries.
    if validation_result[
        "normal_to_degraded"
    ] == 0:
        failures.append(
            "Validation data contain no normal-to-degraded transitions."
        )

    if validation_result[
        "degraded_to_normal"
    ] == 0:
        failures.append(
            "Validation data contain no degraded-to-normal transitions."
        )

    if validation_result[
        "persistence_baseline"
    ]["f1"] >= 0.9999:
        failures.append(
            "The current-state persistence baseline predicts "
            "the validation target perfectly."
        )

    if validation_result[
        "same_state_rate"
    ] >= 0.99:
        warnings.append(
            "At least 99% of validation rows retain the same "
            "degradation state in the next hour."
        )

    if validation_rule_match_rate >= 0.9999:
        warnings.append(
            "The current degradation label is completely "
            "determined by the download-speed and RTT rule."
        )

    if synthetic_rate >= 0.99:
        warnings.append(
            "At least 99% of validation QoE rows are marked synthetic."
        )

    readiness_passed = len(failures) == 0

    print("\n--- Forecasting readiness decision ---")

    if readiness_passed:
        print("READINESS CHECK PASSED.")
        print(
            "The validation period contains both degradation "
            "onsets and recoveries."
        )
    else:
        print("READINESS CHECK FAILED.")

    if failures:
        print("\nFailure reasons:")

        for reason in failures:
            print(f"  - {reason}")

    if warnings:
        print("\nWarnings:")

        for warning in warnings:
            print(f"  - {warning}")

    report = {
        "dataset": str(DATA_PATH),
        "training": training_result,
        "validation": validation_result,
        "validation_current_rule_match_rate": (
            validation_rule_match_rate
        ),
        "validation_synthetic_qoe_rate": (
            synthetic_rate
        ),
        "readiness_passed": readiness_passed,
        "failure_reasons": failures,
        "warnings": warnings,
        "test_rows": int(len(test_df)),
        "test_set_analysed": False,
    }

    transition_rows = [
        {
            "split": result["split"],
            "normal_to_normal": result[
                "normal_to_normal"
            ],
            "normal_to_degraded": result[
                "normal_to_degraded"
            ],
            "degraded_to_normal": result[
                "degraded_to_normal"
            ],
            "degraded_to_degraded": result[
                "degraded_to_degraded"
            ],
            "same_state_rate": result[
                "same_state_rate"
            ],
        }
        for result in [
            training_result,
            validation_result,
        ]
    ]

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    pd.DataFrame(
        transition_rows
    ).to_csv(
        TRANSITION_PATH,
        index=False,
    )

    print(f"\nReadiness report saved to:\n{REPORT_PATH}")

    print(f"\nTransition summary saved to:\n{TRANSITION_PATH}")

    print("\nForecasting-readiness check completed.")
    print("The test set was not analysed.")


if __name__ == "__main__":
    main()