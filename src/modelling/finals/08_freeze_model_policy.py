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
# 2. Load dataset + locked feature registry
# ============================================================

df = pd.read_parquet(DATA_PATH)

with open(
    OUTPUT_DIR
    / "03_feature_registry.json",
    "r",
    encoding="utf-8",
) as f:

    registry = json.load(f)


TARGET = registry["target"]

FEATURES = (
    registry[
        "feature_groups"
    ]["full"]
)


# ============================================================
# 3. Agreed modelling population
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
    8102,  # 2025Q3
    8103,  # 2025Q4
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
    "Model Freeze & Operating Policy"
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
# 5. FROZEN model configuration
#
# Selected from Checkpoints 6 and 7.
# ============================================================

FROZEN_CONFIG = {

    "n_estimators": 23,

    "max_depth": 3,

    "learning_rate": 0.03,

    "min_child_weight": 3,

    "gamma": 0.0,

    "subsample": 0.90,

    "colsample_bytree": 0.90,

    "reg_alpha": 0.2,

    "reg_lambda": 3.0,

    "objective":
        "binary:logistic",

    "eval_metric":
        "aucpr",

    "tree_method":
        "hist",

    "random_state": 42,

    "n_jobs": -1,
}


# ============================================================
# 6. Train frozen pre-test model
#
# IMPORTANT:
# Only TRAIN data is used here.
# Validation is still used only for evaluation.
# ============================================================

model = XGBClassifier(
    **FROZEN_CONFIG
)

model.fit(
    train[FEATURES],
    train["onset_target"],
)


# ============================================================
# 7. Validation risk scores
# ============================================================

scores = model.predict_proba(
    validation[FEATURES]
)[:, 1]


validation_output = validation[
    [
        "h3_index",
        "quarter_index",
        "quarter_label",
        "onset_target",
        "ookla_tests",
    ]
].copy()

validation_output[
    "emerging_risk_score"
] = scores


# ============================================================
# 8. Rank cells WITHIN each quarter
# ============================================================

validation_output[
    "risk_rank"
] = (
    validation_output
    .groupby(
        "quarter_index"
    )[
        "emerging_risk_score"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)


# ============================================================
# 9. Freeze operating risk bands
#
# HIGH:
#   Top 50 highest-risk cells
#
# ELEVATED:
#   ranks 51–200
#
# LOWER:
#   everything else
#
# These are ranking tiers,
# NOT calibrated probabilities.
# ============================================================

def assign_risk_band(rank):

    if rank <= 50:
        return "high"

    elif rank <= 200:
        return "elevated"

    else:
        return "lower"


validation_output[
    "risk_band"
] = (
    validation_output[
        "risk_rank"
    ]
    .apply(
        assign_risk_band
    )
)


# ============================================================
# 10. Overall validation performance
# ============================================================

pr_auc = float(
    average_precision_score(
        validation_output[
            "onset_target"
        ],
        validation_output[
            "emerging_risk_score"
        ],
    )
)

roc_auc = float(
    roc_auc_score(
        validation_output[
            "onset_target"
        ],
        validation_output[
            "emerging_risk_score"
        ],
    )
)


print(
    "\n" + "=" * 80
)

print(
    "FROZEN MODEL VALIDATION"
)

print("=" * 80)

print(
    f"PR-AUC:  {pr_auc:.6f}"
)

print(
    f"ROC-AUC: {roc_auc:.6f}"
)


# ============================================================
# 11. Risk-band behaviour
# ============================================================

band_rows = []


for (
    quarter_index,
    quarter_label
), quarter_df in validation_output.groupby(
    [
        "quarter_index",
        "quarter_label",
    ]
):

    total_events = int(
        quarter_df[
            "onset_target"
        ].sum()
    )

    for band in [
        "high",
        "elevated",
        "lower",
    ]:

        group = quarter_df[
            quarter_df[
                "risk_band"
            ] == band
        ]

        rows = len(group)

        events = int(
            group[
                "onset_target"
            ].sum()
        )

        event_rate = (
            events / rows
            if rows > 0
            else np.nan
        )

        recall = (
            events / total_events
            if total_events > 0
            else np.nan
        )

        band_rows.append({

            "quarter_index":
                quarter_index,

            "quarter_label":
                quarter_label,

            "risk_band":
                band,

            "rows":
                rows,

            "events":
                events,

            "event_rate":
                event_rate,

            "share_of_all_events":
                recall,
        })


band_df = pd.DataFrame(
    band_rows
)


print(
    "\n" + "=" * 80
)

print(
    "RISK BAND PERFORMANCE"
)

print("=" * 80)


print(
    band_df.to_string(
        index=False
    )
)


# ============================================================
# 12. Combined band summary
# ============================================================

combined = (
    band_df
    .groupby(
        "risk_band"
    )
    .agg(

        rows=(
            "rows",
            "sum"
        ),

        events=(
            "events",
            "sum"
        ),
    )
    .reset_index()
)


combined[
    "event_rate"
] = (
    combined["events"]
    / combined["rows"]
)


combined[
    "share_of_all_events"
] = (
    combined["events"]
    / combined["events"].sum()
)


band_order = {
    "high": 0,
    "elevated": 1,
    "lower": 2,
}

combined[
    "_order"
] = (
    combined[
        "risk_band"
    ]
    .map(
        band_order
    )
)

combined = (
    combined
    .sort_values(
        "_order"
    )
    .drop(
        columns="_order"
    )
)


print(
    "\nCombined validation:"
)

print(
    combined.to_string(
        index=False
    )
)


# ============================================================
# 13. Save pre-test model
# ============================================================

PRETEST_MODEL_PATH = (
    MODEL_DIR
    / "frozen_pretest_xgb_full.json"
)

model.save_model(
    PRETEST_MODEL_PATH
)


# ============================================================
# 14. Save formal freeze record
# ============================================================

freeze_record = {

    "model_version":
        "finals_quarterly_onset_v1.0.0",

    "status":
        "FROZEN BEFORE FINAL TEST",

    "model_family":
        "XGBClassifier",

    "feature_group":
        "full",

    "feature_count":
        len(FEATURES),

    "features":
        FEATURES,

    "target":
        TARGET,

    "task":
        (
            "Among currently normal cells, "
            "rank which cells are most likely "
            "to fall into the bottom 25% of "
            "Klang Valley download performance "
            "next quarter."
        ),

    "training_weighting":
        "none",

    "selection_primary_metric":
        "validation PR-AUC",

    "selected_configuration":
        "less_regularized",

    "hyperparameters":
        FROZEN_CONFIG,

    "validation_periods": [
        "2025Q3",
        "2025Q4",
    ],

    "sealed_final_test":
        "2026Q1 -> 2026Q2",

    "score_interpretation":
        (
            "Model output is an emerging-risk "
            "ranking score. It is not presented "
            "as a calibrated probability."
        ),

    "risk_policy": {

        "high":
            "Top 50 cells by emerging-risk score",

        "elevated":
            "Ranks 51 to 200",

        "lower":
            "Ranks 201 and below",
    },

    "measurement_confidence":
        (
            "Ookla test count is retained as "
            "a separate confidence indicator "
            "and is not used as a training weight."
        ),
}


FREEZE_PATH = (
    OUTPUT_DIR
    / "08_model_freeze.json"
)


with open(
    FREEZE_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        freeze_record,
        f,
        indent=2,
    )


# ============================================================
# 15. Save validation outputs
# ============================================================

validation_output.to_parquet(

    OUTPUT_DIR
    / "08_frozen_validation_predictions.parquet",

    index=False,
)


band_df.to_csv(

    OUTPUT_DIR
    / "08_risk_band_by_quarter.csv",

    index=False,
)


combined.to_csv(

    OUTPUT_DIR
    / "08_risk_band_summary.csv",

    index=False,
)


# ============================================================
# 16. Finish
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "CHECKPOINT 8 COMPLETE"
)

print("=" * 80)


print(
    "\nMODEL SELECTION IS NOW FROZEN."
)

print(
    "\nFinal test was NOT evaluated."
)

print(
    "\nFreeze record:"
)

print(
    FREEZE_PATH
)

print(
    "\nPre-test model:"
)

print(
    PRETEST_MODEL_PATH
)