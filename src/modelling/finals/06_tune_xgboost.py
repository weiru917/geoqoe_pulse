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
    *SCRIPT_PATH.parents
]:
    if (
        candidate
        / "data"
        / "outputs"
    ).exists():

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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Load data + locked registry
# ============================================================

df = pd.read_parquet(
    DATA_PATH
)

with open(
    OUTPUT_DIR
    / "03_feature_registry.json",
    "r",
    encoding="utf-8",
) as f:

    registry = json.load(f)


TARGET = registry["target"]

FEATURE_GROUPS = {

    "qoe_history":
        registry[
            "feature_groups"
        ]["qoe_history"],

    "full":
        registry[
            "feature_groups"
        ]["full"],
}


# ============================================================
# 3. Locked modelling population
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
# 4. Locked temporal split
# ============================================================

TRAIN_QUARTERS = list(
    range(8092, 8102)
)

VALIDATION_QUARTERS = [
    8102,
    8103,
]

TEST_QUARTER = 8104


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
    data["quarter_index"]
    == TEST_QUARTER
].copy()


print("=" * 80)
print(
    "GeoQoE Pulse Finals — "
    "Controlled XGBoost Tuning"
)
print("=" * 80)

print(
    f"\nTrain:      {len(train):,}"
)

print(
    f"Validation: {len(validation):,}"
)

print(
    f"Final test: {len(test):,} "
    f"| NOT EVALUATED"
)


# ============================================================
# 5. Small, pre-defined tuning set
# ============================================================

CONFIGS = [

    {
        "name": "baseline",
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "gamma": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },

    {
        "name": "shallow",
        "max_depth": 2,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "gamma": 0.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
    },

    {
        "name": "less_regularized",
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "gamma": 0.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
    },

    {
        "name": "deeper",
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "gamma": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 8.0,
    },

    {
        "name": "faster_learning",
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "gamma": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },

    {
        "name": "simple_fast",
        "max_depth": 2,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "gamma": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 3.0,
    },

    {
        "name": "strong_regularization",
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 10,
        "gamma": 0.10,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    },

    {
        "name": "deep_conservative",
        "max_depth": 4,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "gamma": 0.10,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
    },
]


# ============================================================
# 6. Helpers
# ============================================================

def pr_auc(
    y_true,
    scores,
):

    return float(
        average_precision_score(
            y_true,
            scores,
        )
    )


def top50_precision(
    frame,
    scores,
):

    temp = frame[
        [
            "quarter_index",
            "quarter_label",
            "onset_target",
        ]
    ].copy()

    temp["score"] = scores

    values = []

    for _, group in temp.groupby(
        [
            "quarter_index",
            "quarter_label",
        ]
    ):

        top = (
            group
            .sort_values(
                "score",
                ascending=False,
            )
            .head(50)
        )

        values.append(
            top[
                "onset_target"
            ].mean()
        )

    return float(
        np.mean(values)
    )


# ============================================================
# 7. Train all controlled candidates
# ============================================================

rows = []


for feature_group, features in (
    FEATURE_GROUPS.items()
):

    print(
        "\n" + "=" * 80
    )

    print(
        f"FEATURE GROUP: "
        f"{feature_group.upper()}"
    )

    print("=" * 80)

    for config in CONFIGS:

        print(
            f"\nTraining "
            f"{feature_group} / "
            f"{config['name']}"
        )

        model = XGBClassifier(

            n_estimators=2500,

            max_depth=
                config["max_depth"],

            learning_rate=
                config["learning_rate"],

            min_child_weight=
                config["min_child_weight"],

            gamma=
                config["gamma"],

            subsample=
                config["subsample"],

            colsample_bytree=
                config[
                    "colsample_bytree"
                ],

            reg_alpha=
                config["reg_alpha"],

            reg_lambda=
                config["reg_lambda"],

            objective=
                "binary:logistic",

            eval_metric=
                "aucpr",

            tree_method=
                "hist",

            random_state=42,

            n_jobs=-1,

            early_stopping_rounds=100,
        )


        model.fit(

            train[features],

            train[
                "onset_target"
            ],

            eval_set=[
                (
                    validation[
                        features
                    ],

                    validation[
                        "onset_target"
                    ],
                )
            ],

            verbose=False,
        )


        scores = (
            model
            .predict_proba(
                validation[
                    features
                ]
            )[:, 1]
        )


        overall_pr = pr_auc(
            validation[
                "onset_target"
            ],
            scores,
        )


        roc = float(
            roc_auc_score(
                validation[
                    "onset_target"
                ],
                scores,
            )
        )


        p50 = top50_precision(
            validation,
            scores,
        )


        q_scores = (
            validation[
                [
                    "quarter_index",
                    "quarter_label",
                    "onset_target",
                ]
            ]
            .copy()
        )

        q_scores[
            "score"
        ] = scores


        quarter_prs = {}

        for (
            quarter_index,
            quarter_label
        ), group in q_scores.groupby(
            [
                "quarter_index",
                "quarter_label",
            ]
        ):

            quarter_prs[
                quarter_label
            ] = pr_auc(
                group[
                    "onset_target"
                ],
                group[
                    "score"
                ],
            )


        row = {

            "feature_group":
                feature_group,

            "config":
                config["name"],

            "feature_count":
                len(features),

            "best_iteration":
                int(
                    model.best_iteration
                ),

            "validation_pr_auc":
                overall_pr,

            "validation_roc_auc":
                roc,

            "precision_at_50":
                p50,

            "pr_auc_2025Q3":
                quarter_prs.get(
                    "2025Q3",
                    np.nan,
                ),

            "pr_auc_2025Q4":
                quarter_prs.get(
                    "2025Q4",
                    np.nan,
                ),
        }


        row[
            "min_quarter_pr_auc"
        ] = min(
            row[
                "pr_auc_2025Q3"
            ],
            row[
                "pr_auc_2025Q4"
            ],
        )


        for key, value in (
            config.items()
        ):

            if key != "name":
                row[key] = value


        rows.append(row)


        print(
            f"PR-AUC="
            f"{overall_pr:.6f} | "
            f"P@50="
            f"{p50:.3f} | "
            f"best_iter="
            f"{model.best_iteration}"
        )


# ============================================================
# 8. Rank candidates
# ============================================================

results = pd.DataFrame(
    rows
)

results = results.sort_values(
    [
        "validation_pr_auc",
        "precision_at_50",
    ],
    ascending=[
        False,
        False,
    ],
)


print(
    "\n" + "=" * 80
)

print(
    "TUNING RESULTS — "
    "SORTED BY VALIDATION PR-AUC"
)

print("=" * 80)


display_cols = [

    "feature_group",

    "config",

    "validation_pr_auc",

    "precision_at_50",

    "pr_auc_2025Q3",

    "pr_auc_2025Q4",

    "min_quarter_pr_auc",

    "best_iteration",
]


print(
    results[
        display_cols
    ].to_string(
        index=False
    )
)


# ============================================================
# 9. Save
# ============================================================

results.to_csv(

    OUTPUT_DIR
    / "06_xgboost_tuning_results.csv",

    index=False,
)


print(
    "\n" + "=" * 80
)

print(
    "CHECKPOINT 6 COMPLETE"
)

print("=" * 80)


print(
    "\nFinal test was NOT evaluated."
)

print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
    / "06_xgboost_tuning_results.csv"
)