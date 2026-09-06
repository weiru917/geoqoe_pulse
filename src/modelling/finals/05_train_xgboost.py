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

MODEL_DIR = (
    ROOT
    / "models"
    / "finals_quarterly"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
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
# 3. Agreed modelling population
#
# ONLY currently normal cells with a real future label
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
# 4. LOCKED temporal split
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
# 2026Q1 predicts 2026Q2


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
print("GeoQoE Pulse Finals — XGBoost Training")
print("=" * 80)

print(
    f"\nTrain:      {len(train):,} rows | "
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
# 5. Metrics
# ============================================================

def ranking_metrics(y_true, scores):

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


# ============================================================
# 6. Operational Top-K per quarter
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
                ascending=False,
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
                selected[
                    "onset_target"
                ].sum()
            )

            precision = (
                captured / actual_k
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
# 7. Fixed XGBoost configuration
#
# We deliberately use ONE reasonable configuration first.
# No hyperparameter fishing yet.
# ============================================================

def build_xgb():

    return XGBClassifier(

        n_estimators=2000,

        max_depth=3,

        learning_rate=0.03,

        min_child_weight=5,

        gamma=0.05,

        subsample=0.80,

        colsample_bytree=0.80,

        reg_alpha=0.5,

        reg_lambda=5.0,

        objective="binary:logistic",

        eval_metric="aucpr",

        tree_method="hist",

        random_state=42,

        n_jobs=-1,

        early_stopping_rounds=75,
    )


# ============================================================
# 8. Train the three agreed variants
# ============================================================

MODEL_GROUPS = {

    "xgb_qoe_history":
        QOE_FEATURES,

    "xgb_context_only":
        CONTEXT_FEATURES,

    "xgb_full":
        FULL_FEATURES,
}


validation_predictions = validation[
    [
        "h3_index",
        "quarter_index",
        "quarter_label",
        "onset_target",
    ]
].copy()


metric_rows = []
per_quarter_rows = []
importance_rows = []
topk_outputs = []


for model_name, features in MODEL_GROUPS.items():

    print("\n" + "=" * 80)
    print(
        f"TRAINING {model_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Features: {len(features)}"
    )

    model = build_xgb()

    model.fit(

        train[features],
        train["onset_target"],

        eval_set=[
            (
                validation[features],
                validation["onset_target"],
            )
        ],

        verbose=False,
    )


    # --------------------------------------------------------
    # Validation scores
    # --------------------------------------------------------

    scores = model.predict_proba(
        validation[features]
    )[:, 1]


    validation_predictions[
        model_name
    ] = scores


    overall = ranking_metrics(
        validation["onset_target"],
        scores,
    )


    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )


    metric_rows.append({

        "model": model_name,

        "feature_count":
            len(features),

        "best_iteration":
            best_iteration,

        "validation_pr_auc":
            overall["pr_auc"],

        "validation_roc_auc":
            overall["roc_auc"],
    })


    print(
        f"Best iteration: {best_iteration}"
    )

    print(
        f"Validation PR-AUC: "
        f"{overall['pr_auc']:.6f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{overall['roc_auc']:.6f}"
    )


    # --------------------------------------------------------
    # Performance separately for each validation quarter
    # --------------------------------------------------------

    temp = validation_predictions[
        [
            "quarter_index",
            "quarter_label",
            "onset_target",
            model_name,
        ]
    ].copy()


    print("\nPer-quarter validation:")

    for (
        quarter_index,
        quarter_label
    ), group in temp.groupby(
        ["quarter_index", "quarter_label"]
    ):

        q_metrics = ranking_metrics(
            group["onset_target"],
            group[model_name],
        )

        print(
            f"{quarter_label}: "
            f"PR-AUC={q_metrics['pr_auc']:.6f} | "
            f"ROC-AUC={q_metrics['roc_auc']:.6f}"
        )

        per_quarter_rows.append({

            "model":
                model_name,

            "quarter_index":
                quarter_index,

            "quarter_label":
                quarter_label,

            "pr_auc":
                q_metrics["pr_auc"],

            "roc_auc":
                q_metrics["roc_auc"],
        })


    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    topk = top_k_by_quarter(
        validation_predictions,
        model_name,
        ks=(50, 100, 200),
    )

    topk["model"] = model_name

    topk_outputs.append(topk)


    # --------------------------------------------------------
    # Feature importance — gain
    # --------------------------------------------------------

    booster = model.get_booster()

    gain_dict = booster.get_score(
        importance_type="gain"
    )

    total_gain = sum(
        gain_dict.values()
    )

    for feature in features:

        gain = float(
            gain_dict.get(feature, 0.0)
        )

        importance_rows.append({

            "model":
                model_name,

            "feature":
                feature,

            "gain":
                gain,

            "gain_pct":
                (
                    100.0 * gain / total_gain
                    if total_gain > 0
                    else 0.0
                ),
        })


    # --------------------------------------------------------
    # Save each model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name}.json"
    )

    model.save_model(
        model_path
    )

    print(
        f"Saved model: {model_path}"
    )


# ============================================================
# 9. Overall model comparison
# ============================================================

metrics_df = (
    pd.DataFrame(metric_rows)
    .sort_values(
        "validation_pr_auc",
        ascending=False,
    )
)


print("\n" + "=" * 80)
print("XGBOOST VALIDATION RANKING")
print("=" * 80)

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 10. Compare against previous logistic baselines
# ============================================================

BASELINE_PATH = (
    OUTPUT_DIR
    / "04_baseline_validation_metrics.csv"
)

if BASELINE_PATH.exists():

    baselines = pd.read_csv(
        BASELINE_PATH
    )

    print("\n" + "=" * 80)
    print("COMPARISON WITH LOGISTIC BASELINES")
    print("=" * 80)

    baseline_best = (
        baselines
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        f"Best previous baseline: "
        f"{baseline_best['model']} | "
        f"PR-AUC={baseline_best['pr_auc']:.6f}"
    )

    best_xgb = metrics_df.iloc[0]

    absolute_gain = (
        best_xgb["validation_pr_auc"]
        - baseline_best["pr_auc"]
    )

    relative_gain = (
        absolute_gain
        / baseline_best["pr_auc"]
        * 100.0
    )

    print(
        f"Best XGBoost: "
        f"{best_xgb['model']} | "
        f"PR-AUC="
        f"{best_xgb['validation_pr_auc']:.6f}"
    )

    print(
        f"Absolute PR-AUC gain: "
        f"{absolute_gain:.6f}"
    )

    print(
        f"Relative improvement: "
        f"{relative_gain:.2f}%"
    )


# ============================================================
# 11. Top-K summary
# ============================================================

topk_df = pd.concat(
    topk_outputs,
    ignore_index=True,
)


topk_summary = (
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
print("XGBOOST OPERATIONAL TOP-K")
print("=" * 80)

for k in [50, 100, 200]:

    print(f"\nTOP {k}")

    display = (
        topk_summary[
            topk_summary["k"] == k
        ]
        .sort_values(
            "mean_precision_at_k",
            ascending=False,
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# 12. Feature importance
# ============================================================

importance_df = pd.DataFrame(
    importance_rows
)


print("\n" + "=" * 80)
print("TOP FEATURES")
print("=" * 80)

for model_name in MODEL_GROUPS:

    print(
        f"\n{model_name}:"
    )

    top_features = (
        importance_df[
            importance_df["model"]
            == model_name
        ]
        .sort_values(
            "gain_pct",
            ascending=False,
        )
        .head(15)
    )

    print(
        top_features[
            [
                "feature",
                "gain_pct",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 13. Save everything
# ============================================================

metrics_df.to_csv(

    OUTPUT_DIR
    / "05_xgboost_validation_metrics.csv",

    index=False,
)


pd.DataFrame(
    per_quarter_rows
).to_csv(

    OUTPUT_DIR
    / "05_xgboost_per_quarter_metrics.csv",

    index=False,
)


topk_df.to_csv(

    OUTPUT_DIR
    / "05_xgboost_topk_by_quarter.csv",

    index=False,
)


topk_summary.to_csv(

    OUTPUT_DIR
    / "05_xgboost_topk_summary.csv",

    index=False,
)


importance_df.to_csv(

    OUTPUT_DIR
    / "05_xgboost_feature_importance.csv",

    index=False,
)


validation_predictions.to_parquet(

    OUTPUT_DIR
    / "05_xgboost_validation_predictions.parquet",

    index=False,
)


print("\n" + "=" * 80)
print("CHECKPOINT 5 COMPLETE")
print("=" * 80)

print(
    "\nIMPORTANT: Final test was NOT evaluated."
)

print(
    "\nDo not change the final test or inspect "
    "its model performance yet."
)