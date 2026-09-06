from pathlib import Path
import json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# 1. Locate repository
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()

ROOT = None

for candidate in [
    SCRIPT_PATH.parent,
    *SCRIPT_PATH.parents,
]:
    if (candidate / "data" / "outputs").exists():
        ROOT = candidate
        break

if ROOT is None:
    raise RuntimeError(
        "Could not locate repository root."
    )


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

MODEL_DIR = (
    ROOT
    / "models"
    / "finals_quarterly"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Load locked registry + freeze record
# ============================================================

df = pd.read_parquet(DATA_PATH)

with open(
    OUTPUT_DIR / "03_feature_registry.json",
    "r",
    encoding="utf-8",
) as f:
    registry = json.load(f)


with open(
    OUTPUT_DIR / "08_model_freeze.json",
    "r",
    encoding="utf-8",
) as f:
    freeze = json.load(f)


if freeze["status"] != "FROZEN BEFORE FINAL TEST":
    raise RuntimeError(
        "Model freeze record is not valid."
    )


TARGET = registry["target"]

FULL_FEATURES = (
    registry["feature_groups"]["full"]
)

QOE_FEATURES = (
    registry["feature_groups"]["qoe_history"]
)


# ============================================================
# 3. Modelling population
# ============================================================

data = df[
    (df["target_next_q_valid"] == 1)
    & (df["is_below_area_norm"] == 0)
].copy()

data["onset_target"] = (
    data[TARGET].astype(int)
)


# ============================================================
# 4. FINAL split
#
# Pre-test:
# 2023Q1 -> 2025Q4
#
# Sealed final test:
# 2026Q1 -> 2026Q2
# ============================================================

PRETEST_QUARTERS = list(
    range(8092, 8104)
)

TEST_QUARTER = 8104


pretest = data[
    data["quarter_index"].isin(
        PRETEST_QUARTERS
    )
].copy()

test = data[
    data["quarter_index"]
    == TEST_QUARTER
].copy()


print("=" * 80)
print(
    "GeoQoE Pulse Finals — "
    "SEALED FINAL TEST"
)
print("=" * 80)

print(
    f"\nFinal training rows: "
    f"{len(pretest):,}"
)

print(
    f"Final training events: "
    f"{pretest['onset_target'].sum():,}"
)

print(
    f"Final training event rate: "
    f"{pretest['onset_target'].mean():.4%}"
)

print(
    f"\nSEALED test rows: "
    f"{len(test):,}"
)

print(
    "\nTest predictor quarter: 2026Q1"
)

print(
    "Actual outcome quarter: 2026Q2"
)


# ============================================================
# 5. Helper metrics
# ============================================================

def metrics(
    y_true,
    scores,
):

    return {

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
    }


def top_k_metrics(
    frame,
    score_col,
    ks=(50, 100, 200),
):

    ranked = (
        frame
        .sort_values(
            score_col,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    total_events = int(
        ranked["onset_target"].sum()
    )

    prevalence = float(
        ranked["onset_target"].mean()
    )

    rows = []

    for k in ks:

        selected = ranked.head(k)

        captured = int(
            selected[
                "onset_target"
            ].sum()
        )

        precision = (
            captured / len(selected)
        )

        recall = (
            captured / total_events
        )

        lift = (
            precision / prevalence
        )

        rows.append({

            "k":
                k,

            "captured_events":
                captured,

            "total_events":
                total_events,

            "precision_at_k":
                precision,

            "recall_at_k":
                recall,

            "lift_at_k":
                lift,
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. Train FINAL frozen full model
#
# Same model configuration chosen BEFORE seeing test.
# No early stopping.
# No hyperparameter changes.
# ============================================================

FROZEN_CONFIG = (
    freeze["hyperparameters"]
)


final_model = XGBClassifier(
    **FROZEN_CONFIG
)


final_model.fit(
    pretest[FULL_FEATURES],
    pretest["onset_target"],
)


full_scores = (
    final_model
    .predict_proba(
        test[FULL_FEATURES]
    )[:, 1]
)


# ============================================================
# 7. QoE-history comparator
#
# This configuration was already evaluated in Checkpoint 6.
# It is NOT selected using the final test.
# ============================================================

QOE_COMPARATOR_CONFIG = {

    "n_estimators": 28,

    "max_depth": 3,

    "learning_rate": 0.03,

    "min_child_weight": 5,

    "gamma": 0.05,

    "subsample": 0.80,

    "colsample_bytree": 0.80,

    "reg_alpha": 0.5,

    "reg_lambda": 5.0,

    "objective":
        "binary:logistic",

    "eval_metric":
        "aucpr",

    "tree_method":
        "hist",

    "random_state": 42,

    "n_jobs": -1,
}


qoe_model = XGBClassifier(
    **QOE_COMPARATOR_CONFIG
)


qoe_model.fit(
    pretest[QOE_FEATURES],
    pretest["onset_target"],
)


qoe_scores = (
    qoe_model
    .predict_proba(
        test[QOE_FEATURES]
    )[:, 1]
)


# ============================================================
# 8. Already-defined simple baselines
# ============================================================

constant_score = np.full(
    len(test),
    pretest[
        "onset_target"
    ].mean(),
)


current_percentile_score = (
    100.0
    - test[
        "dl_percentile_in_quarter"
    ].values
)


current_dl_score = (
    -test[
        "ookla_dl_mbps"
    ].values
)


history_median = (
    pretest[
        "below_area_norm_rate_to_date"
    ]
    .median()
)


historical_recurrence_score = (
    test[
        "below_area_norm_rate_to_date"
    ]
    .fillna(
        history_median
    )
    .values
)


# ============================================================
# 9. Build test prediction table
# ============================================================

predictions = test[
    [
        "h3_index",
        "quarter_index",
        "quarter_label",
        "onset_target",
        "ookla_tests",
        "ookla_dl_mbps",
        "ookla_ul_mbps",
        "ookla_rtt_ms",
        "dl_percentile_in_quarter",
    ]
].copy()


predictions[
    "constant"
] = constant_score

predictions[
    "current_percentile"
] = current_percentile_score

predictions[
    "current_dl"
] = current_dl_score

predictions[
    "historical_recurrence"
] = historical_recurrence_score

predictions[
    "xgb_qoe_history"
] = qoe_scores

predictions[
    "xgb_full"
] = full_scores


# ============================================================
# 10. FINAL TEST metrics
# ============================================================

MODEL_COLUMNS = [

    "constant",

    "current_percentile",

    "current_dl",

    "historical_recurrence",

    "xgb_qoe_history",

    "xgb_full",
]


metric_rows = []


for model_name in MODEL_COLUMNS:

    result = metrics(

        predictions[
            "onset_target"
        ],

        predictions[
            model_name
        ],
    )

    result[
        "model"
    ] = model_name

    metric_rows.append(
        result
    )


metrics_df = (
    pd.DataFrame(
        metric_rows
    )
    [
        [
            "model",
            "pr_auc",
            "roc_auc",
        ]
    ]
    .sort_values(
        "pr_auc",
        ascending=False,
    )
)


print(
    "\n" + "=" * 80
)

print(
    "FINAL TEST RESULTS"
)

print("=" * 80)


print(
    f"\nActual emerging events: "
    f"{predictions['onset_target'].sum():,}"
)

print(
    f"Actual event rate: "
    f"{predictions['onset_target'].mean():.4%}"
)


print("\nModel ranking:")

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 11. Top-K FINAL evaluation
# ============================================================

all_topk = []


for model_name in MODEL_COLUMNS:

    topk = top_k_metrics(
        predictions,
        model_name,
        ks=(50, 100, 200),
    )

    topk[
        "model"
    ] = model_name

    all_topk.append(
        topk
    )


topk_df = pd.concat(
    all_topk,
    ignore_index=True,
)


print(
    "\n" + "=" * 80
)

print(
    "FINAL TEST TOP-K"
)

print("=" * 80)


for k in [
    50,
    100,
    200,
]:

    print(
        f"\nTOP {k}"
    )

    display = (
        topk_df[
            topk_df["k"] == k
        ]
        .sort_values(
            "precision_at_k",
            ascending=False,
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# 12. Apply FROZEN dashboard risk policy
# ============================================================

predictions[
    "emerging_risk_score"
] = predictions[
    "xgb_full"
]


predictions[
    "risk_rank"
] = (
    predictions[
        "emerging_risk_score"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)


def assign_band(rank):

    if rank <= 50:
        return "high"

    elif rank <= 200:
        return "elevated"

    else:
        return "lower"


predictions[
    "risk_band"
] = (
    predictions[
        "risk_rank"
    ]
    .apply(
        assign_band
    )
)


band_rows = []


total_events = int(
    predictions[
        "onset_target"
    ].sum()
)


for band in [
    "high",
    "elevated",
    "lower",
]:

    group = predictions[
        predictions[
            "risk_band"
        ] == band
    ]

    rows = len(group)

    events = int(
        group[
            "onset_target"
        ].sum()
    )

    band_rows.append({

        "risk_band":
            band,

        "rows":
            rows,

        "events":
            events,

        "event_rate":
            events / rows,

        "share_of_all_events":
            events / total_events,
    })


band_df = pd.DataFrame(
    band_rows
)


print(
    "\n" + "=" * 80
)

print(
    "FROZEN RISK BAND — FINAL TEST"
)

print("=" * 80)


print(
    band_df.to_string(
        index=False
    )
)


# ============================================================
# 13. Save FINAL model + results
# ============================================================

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "final_xgb_full_v1.0.0.json"
)


final_model.save_model(
    FINAL_MODEL_PATH
)


QOE_MODEL_PATH = (
    MODEL_DIR
    / "final_xgb_qoe_comparator.json"
)


qoe_model.save_model(
    QOE_MODEL_PATH
)


metrics_df.to_csv(

    OUTPUT_DIR
    / "09_final_test_metrics.csv",

    index=False,
)


topk_df.to_csv(

    OUTPUT_DIR
    / "09_final_test_topk.csv",

    index=False,
)


band_df.to_csv(

    OUTPUT_DIR
    / "09_final_test_risk_bands.csv",

    index=False,
)


predictions.to_parquet(

    OUTPUT_DIR
    / "09_final_test_predictions.parquet",

    index=False,
)


# ============================================================
# 14. Save final evaluation record
# ============================================================

evaluation_record = {

    "model_version":
        "finals_quarterly_onset_v1.0.0",

    "model_selection":
        "frozen before final test",

    "final_training_period":
        "2023Q1 through 2025Q4",

    "test_predictor_period":
        "2026Q1",

    "test_outcome_period":
        "2026Q2",

    "test_rows":
        len(test),

    "test_events":
        int(
            test[
                "onset_target"
            ].sum()
        ),

    "no_post_test_tuning":
        True,
}


with open(

    OUTPUT_DIR
    / "09_final_test_record.json",

    "w",
    encoding="utf-8",

) as f:

    json.dump(
        evaluation_record,
        f,
        indent=2,
    )


print(
    "\n" + "=" * 80
)

print(
    "CHECKPOINT 9 COMPLETE"
)

print("=" * 80)


print(
    "\nTHE SEALED FINAL TEST "
    "HAS NOW BEEN OPENED."
)

print(
    "\nDo NOT tune model selection, "
    "hyperparameters or risk bands "
    "using these results."
)

print(
    "\nFinal model:"
)

print(
    FINAL_MODEL_PATH
)