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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Load data and locked features
# ============================================================

df = pd.read_parquet(DATA_PATH)

with open(
    OUTPUT_DIR / "03_feature_registry.json",
    "r",
    encoding="utf-8",
) as f:
    registry = json.load(f)


TARGET = registry["target"]

FEATURES = (
    registry["feature_groups"]["full"]
)


# ============================================================
# 3. Agreed modelling population
# ============================================================

data = df[
    (df["target_next_q_valid"] == 1)
    & (df["is_below_area_norm"] == 0)
].copy()

data["onset_target"] = (
    data[TARGET].astype(int)
)


# ============================================================
# 4. Locked split
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
    data["quarter_index"] == TEST_QUARTER
].copy()


print("=" * 80)
print(
    "GeoQoE Pulse Finals — "
    "Measurement Reliability Check"
)
print("=" * 80)

print(
    f"\nTrain:      {len(train):,}"
)

print(
    f"Validation: {len(validation):,}"
)

print(
    f"Final test: {len(test):,} | NOT EVALUATED"
)


# ============================================================
# 5. Inspect Ookla measurement support
# ============================================================

print("\n" + "=" * 80)
print("OOKLA TEST-COUNT SUPPORT")
print("=" * 80)

print(
    train["ookla_tests"]
    .describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
        ]
    )
)


# Use training-set 25th percentile as a
# sensitivity threshold.
support_threshold = float(
    train["ookla_tests"].quantile(0.25)
)

print(
    f"\nHigher-support threshold "
    f"(training 25th percentile): "
    f"{support_threshold:.1f} tests"
)


# ============================================================
# 6. Candidate configurations from Checkpoint 6
# ============================================================

CONFIGS = {

    "less_regularized": {
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "gamma": 0.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.2,
        "reg_lambda": 3.0,
    },

    "deeper": {
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "gamma": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 8.0,
    },
}


# ============================================================
# 7. Measurement-reliability weights
#
# log1p prevents a few cells with huge test counts
# from dominating training.
#
# Normalize mean weight to 1.
# ============================================================

raw_weights = np.log1p(
    train["ookla_tests"].clip(lower=1)
)

reliability_weights = (
    raw_weights
    / raw_weights.mean()
)


print("\nTraining weight summary:")

print(
    pd.Series(
        reliability_weights
    ).describe()
)


# ============================================================
# 8. Helper functions
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

    precisions = []

    for _, group in temp.groupby(
        [
            "quarter_index",
            "quarter_label",
        ]
    ):

        top50 = (
            group
            .sort_values(
                "score",
                ascending=False,
            )
            .head(50)
        )

        precisions.append(
            top50[
                "onset_target"
            ].mean()
        )

    return float(
        np.mean(precisions)
    )


# ============================================================
# 9. Train weighted and unweighted versions
# ============================================================

results = []


for config_name, config in CONFIGS.items():

    for weighting in [
        "unweighted",
        "ookla_reliability_weighted",
    ]:

        print(
            "\nTraining "
            f"{config_name} / {weighting}"
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
                config["colsample_bytree"],

            reg_alpha=
                config["reg_alpha"],

            reg_lambda=
                config["reg_lambda"],

            objective="binary:logistic",

            eval_metric="aucpr",

            tree_method="hist",

            random_state=42,

            n_jobs=-1,

            early_stopping_rounds=100,
        )


        fit_kwargs = {

            "X":
                train[FEATURES],

            "y":
                train[
                    "onset_target"
                ],

            "eval_set": [
                (
                    validation[FEATURES],

                    validation[
                        "onset_target"
                    ],
                )
            ],

            "verbose":
                False,
        }


        if (
            weighting
            == "ookla_reliability_weighted"
        ):
            fit_kwargs[
                "sample_weight"
            ] = reliability_weights


        model.fit(
            **fit_kwargs
        )


        scores = model.predict_proba(
            validation[FEATURES]
        )[:, 1]


        overall_pr = pr_auc(
            validation[
                "onset_target"
            ],
            scores,
        )


        overall_roc = float(
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


        # ----------------------------------------------------
        # Per-quarter stability
        # ----------------------------------------------------

        temp = validation[
            [
                "quarter_index",
                "quarter_label",
                "onset_target",
                "ookla_tests",
            ]
        ].copy()

        temp["score"] = scores


        quarter_scores = {}

        for (
            quarter_index,
            quarter_label
        ), group in temp.groupby(
            [
                "quarter_index",
                "quarter_label",
            ]
        ):

            quarter_scores[
                quarter_label
            ] = pr_auc(
                group[
                    "onset_target"
                ],
                group["score"],
            )


        # ----------------------------------------------------
        # Higher-support sensitivity check
        # ----------------------------------------------------

        supported = temp[
            temp["ookla_tests"]
            >= support_threshold
        ].copy()


        supported_pr = pr_auc(
            supported[
                "onset_target"
            ],
            supported[
                "score"
            ],
        )


        row = {

            "config":
                config_name,

            "weighting":
                weighting,

            "best_iteration":
                int(
                    model.best_iteration
                ),

            "validation_pr_auc":
                overall_pr,

            "validation_roc_auc":
                overall_roc,

            "precision_at_50":
                p50,

            "pr_auc_2025Q3":
                quarter_scores.get(
                    "2025Q3",
                    np.nan,
                ),

            "pr_auc_2025Q4":
                quarter_scores.get(
                    "2025Q4",
                    np.nan,
                ),

            "higher_support_pr_auc":
                supported_pr,

            "higher_support_rows":
                len(supported),
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


        results.append(row)


        print(
            f"PR-AUC="
            f"{overall_pr:.6f} | "
            f"P@50={p50:.3f} | "
            f"supported PR-AUC="
            f"{supported_pr:.6f}"
        )


# ============================================================
# 10. Compare
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "validation_pr_auc",
        "min_quarter_pr_auc",
    ],
    ascending=[
        False,
        False,
    ],
)


print("\n" + "=" * 80)
print("MEASUREMENT RELIABILITY RESULTS")
print("=" * 80)


display_columns = [

    "config",

    "weighting",

    "validation_pr_auc",

    "precision_at_50",

    "pr_auc_2025Q3",

    "pr_auc_2025Q4",

    "min_quarter_pr_auc",

    "higher_support_pr_auc",

    "best_iteration",
]


print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. Save
# ============================================================

results_df.to_csv(

    OUTPUT_DIR
    / "07_measurement_weighting_results.csv",

    index=False,
)


print("\n" + "=" * 80)
print("CHECKPOINT 7 COMPLETE")
print("=" * 80)

print(
    "\nFinal test was NOT evaluated."
)

print(
    "\nAfter this checkpoint, "
    "freeze model selection."
)