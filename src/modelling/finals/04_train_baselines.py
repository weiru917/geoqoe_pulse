from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# 1. Locate repository
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None
for candidate in [SCRIPT_PATH.parent, *SCRIPT_PATH.parents]:
    if (candidate / "data" / "outputs").exists():
        ROOT = candidate
        break

if ROOT is None:
    raise RuntimeError("Could not locate repository root.")

DATA_PATH = (
    ROOT
    / "data"
    / "outputs"
    / "cell_quarter_dataset.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "finals_quarterly"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REGISTRY_PATH = (
    OUTPUT_DIR
    / "03_feature_registry.json"
)


# ============================================================
# 2. Load data + locked feature registry
# ============================================================

df = pd.read_parquet(DATA_PATH)

with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8",
) as f:
    registry = json.load(f)

TARGET = registry["target"]

QOE_FEATURES = (
    registry["feature_groups"]["qoe_history"]
)

CONTEXT_FEATURES = (
    registry["feature_groups"]["context_only"]
)

FULL_FEATURES = (
    registry["feature_groups"]["full"]
)


# ============================================================
# 3. Construct the agreed modelling population
#
# ONLY:
#   currently normal
#   +
#   real next-quarter outcome exists
# ============================================================

data = df[
    (df["target_next_q_valid"] == 1)
    & (df["is_below_area_norm"] == 0)
].copy()

data["onset_target"] = (
    data[TARGET]
    .astype(int)
)


# ============================================================
# 4. Recreate LOCKED chronological split
# ============================================================

TRAIN_QUARTERS = list(
    range(8092, 8102)
)
# 2023Q1 -> 2025Q2

VALIDATION_QUARTERS = [
    8102,  # 2025Q3
    8103,  # 2025Q4
]

TEST_QUARTER = 8104
# 2026Q1 -> predicts real 2026Q2


train = data[
    data["quarter_index"].isin(
        TRAIN_QUARTERS
    )
].copy()

validation = data[
    data["quarter_index"].isin(
        VALIDATION_QUARTERS
    )
].copy()

test = data[
    data["quarter_index"] == TEST_QUARTER
].copy()


print("=" * 80)
print("GeoQoE Pulse Finals — Baseline Training")
print("=" * 80)

print("\nLOCKED SPLIT")

print(
    f"Train:      {len(train):,} rows | "
    f"{train['onset_target'].sum():,} events | "
    f"{train['onset_target'].mean():.4%}"
)

print(
    f"Validation: {len(validation):,} rows | "
    f"{validation['onset_target'].sum():,} events | "
    f"{validation['onset_target'].mean():.4%}"
)

print(
    f"Final test: {len(test):,} rows | "
    f"NOT EVALUATED"
)


# ============================================================
# 5. Helper: overall ranking metrics
# ============================================================

def ranking_metrics(
    y_true,
    scores,
):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    ap = average_precision_score(
        y_true,
        scores,
    )

    try:
        roc = roc_auc_score(
            y_true,
            scores,
        )
    except ValueError:
        roc = np.nan

    return {
        "pr_auc": float(ap),
        "roc_auc": float(roc),
    }


# ============================================================
# 6. Helper: operational Top-K metrics PER QUARTER
#
# This simulates:
# "If the regulator can investigate only the top K
# flagged cells each quarter, how useful is the list?"
# ============================================================

def top_k_by_quarter(
    frame,
    score_col,
    ks=(50, 100, 200),
):

    rows = []

    for (
        quarter_index,
        quarter_label
    ), group in frame.groupby(
        ["quarter_index", "quarter_label"]
    ):

        group = (
            group
            .sort_values(
                score_col,
                ascending=False
            )
            .reset_index(drop=True)
        )

        total_events = int(
            group["onset_target"].sum()
        )

        prevalence = float(
            group["onset_target"].mean()
        )

        for k in ks:

            actual_k = min(
                k,
                len(group),
            )

            selected = group.head(actual_k)

            captured = int(
                selected["onset_target"].sum()
            )

            precision = (
                captured / actual_k
                if actual_k > 0
                else np.nan
            )

            recall = (
                captured / total_events
                if total_events > 0
                else np.nan
            )

            lift = (
                precision / prevalence
                if prevalence > 0
                else np.nan
            )

            rows.append({
                "quarter_index": quarter_index,
                "quarter_label": quarter_label,
                "k": actual_k,
                "captured_events": captured,
                "total_events": total_events,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_at_k": lift,
            })

    return pd.DataFrame(rows)


# ============================================================
# 7. Simple non-ML baselines
# ============================================================

validation_predictions = validation[
    [
        "h3_index",
        "quarter_index",
        "quarter_label",
        "onset_target",
    ]
].copy()


# ------------------------------------------------------------
# Baseline A:
# Constant train prevalence
# ------------------------------------------------------------

train_prevalence = float(
    train["onset_target"].mean()
)

validation_predictions[
    "constant"
] = train_prevalence


# ------------------------------------------------------------
# Baseline B:
# Current DL percentile
#
# Lower percentile = closer to bottom 25%
# Therefore higher risk score = 100 - percentile
# ------------------------------------------------------------

validation_predictions[
    "current_percentile"
] = (
    100.0
    - validation["dl_percentile_in_quarter"]
)


# ------------------------------------------------------------
# Baseline C:
# Current download speed
#
# Lower speed -> larger risk score
# ------------------------------------------------------------

validation_predictions[
    "current_dl"
] = (
    -validation["ookla_dl_mbps"]
)


# ------------------------------------------------------------
# Baseline D:
# Historical underperformance rate
#
# Higher historical rate -> higher risk
#
# Missing history receives train median.
# ------------------------------------------------------------

history_median = (
    train[
        "below_area_norm_rate_to_date"
    ]
    .median()
)

validation_predictions[
    "historical_recurrence"
] = (
    validation[
        "below_area_norm_rate_to_date"
    ]
    .fillna(history_median)
)


# ============================================================
# 8. Logistic Regression models
# ============================================================

def build_logistic_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                C=1.0,
                max_iter=5000,
                solver="lbfgs",
                random_state=42,
            )
        ),
    ])


LOGISTIC_GROUPS = {
    "logistic_qoe_history": QOE_FEATURES,
    "logistic_context_only": CONTEXT_FEATURES,
    "logistic_full": FULL_FEATURES,
}


trained_models = {}

for model_name, features in LOGISTIC_GROUPS.items():

    print(
        "\nTraining:",
        model_name,
        f"({len(features)} features)"
    )

    model = build_logistic_model()

    model.fit(
        train[features],
        train["onset_target"],
    )

    scores = model.predict_proba(
        validation[features]
    )[:, 1]

    validation_predictions[
        model_name
    ] = scores

    trained_models[
        model_name
    ] = model


# ============================================================
# 9. Overall validation metrics
# ============================================================

MODEL_COLUMNS = [
    "constant",
    "current_percentile",
    "current_dl",
    "historical_recurrence",
    "logistic_qoe_history",
    "logistic_context_only",
    "logistic_full",
]


metric_rows = []

y_val = validation_predictions[
    "onset_target"
].values


print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

for model_name in MODEL_COLUMNS:

    result = ranking_metrics(
        y_val,
        validation_predictions[
            model_name
        ].values,
    )

    result["model"] = model_name

    metric_rows.append(result)

metrics_df = (
    pd.DataFrame(metric_rows)
    [
        [
            "model",
            "pr_auc",
            "roc_auc",
        ]
    ]
    .sort_values(
        "pr_auc",
        ascending=False
    )
)

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 10. Per-quarter Top-K operational evaluation
# ============================================================

all_topk = []

for model_name in MODEL_COLUMNS:

    temp = (
        validation_predictions
        .copy()
    )

    result = top_k_by_quarter(
        temp,
        model_name,
        ks=(50, 100, 200),
    )

    result[
        "model"
    ] = model_name

    all_topk.append(result)


topk_df = pd.concat(
    all_topk,
    ignore_index=True
)


summary_topk = (
    topk_df
    .groupby(
        ["model", "k"]
    )
    .agg(
        mean_precision_at_k=(
            "precision_at_k",
            "mean"
        ),
        mean_recall_at_k=(
            "recall_at_k",
            "mean"
        ),
        mean_lift_at_k=(
            "lift_at_k",
            "mean"
        ),
        total_captured=(
            "captured_events",
            "sum"
        ),
        total_events=(
            "total_events",
            "sum"
        ),
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("OPERATIONAL TOP-K — VALIDATION")
print("=" * 80)

for k in [50, 100, 200]:

    print(f"\nTOP {k}")

    display = (
        summary_topk[
            summary_topk["k"] == k
        ]
        .sort_values(
            "mean_precision_at_k",
            ascending=False
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# 11. Save logistic coefficients
# ============================================================

coefficient_rows = []

for model_name, features in LOGISTIC_GROUPS.items():

    pipeline = trained_models[
        model_name
    ]

    coef = (
        pipeline
        .named_steps["model"]
        .coef_[0]
    )

    for feature, value in zip(
        features,
        coef,
    ):
        coefficient_rows.append({
            "model": model_name,
            "feature": feature,
            "standardized_coefficient": float(
                value
            ),
            "absolute_coefficient": float(
                abs(value)
            ),
        })


coefficients_df = pd.DataFrame(
    coefficient_rows
)


# ============================================================
# 12. Save outputs
# ============================================================

metrics_df.to_csv(
    OUTPUT_DIR
    / "04_baseline_validation_metrics.csv",
    index=False,
)

topk_df.to_csv(
    OUTPUT_DIR
    / "04_baseline_topk_by_quarter.csv",
    index=False,
)

summary_topk.to_csv(
    OUTPUT_DIR
    / "04_baseline_topk_summary.csv",
    index=False,
)

coefficients_df.to_csv(
    OUTPUT_DIR
    / "04_logistic_coefficients.csv",
    index=False,
)

validation_predictions.to_parquet(
    OUTPUT_DIR
    / "04_baseline_validation_predictions.parquet",
    index=False,
)


print("\n" + "=" * 80)
print("CHECKPOINT 4 COMPLETE")
print("=" * 80)

print(
    "\nFinal test was NOT evaluated."
)

print("\nSaved baseline outputs to:")
print(OUTPUT_DIR)