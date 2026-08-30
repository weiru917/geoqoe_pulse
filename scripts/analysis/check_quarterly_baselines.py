"""
Is the quarterly dataset actually learnable?

SCOPE: this is a DATA-READINESS check, not a modelling deliverable. The
question is not "which model is best" — that belongs to the modelling
team. The question is whether the data supports ANY improvement over
trivial rules. If a gradient-boosted model cannot beat "assume next
quarter looks like this one", the dataset has no exploitable signal and
the modelling team needs to know that before they invest in it.

Held out honestly: trained on 2023Q1-2025Q4, tested on 2026Q1 — a real
future quarter the model never saw.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUT_DIR

TARGET = "target_is_below_area_norm_next_q"
OUT = OUTPUT_DIR / "quarterly_baseline_check.json"


def metrics(y, s, k=200):
    from sklearn.metrics import average_precision_score, roc_auc_score
    order = np.argsort(-s)
    return {
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        f"precision_at_top_{k}": round(float(y[order[:k]].mean()), 4),
    }


def main():
    df = pd.read_parquet(OUTPUT_DIR / "cell_quarter_dataset.parquet")
    d = pd.read_excel(OUTPUT_DIR / "feature_dictionary.xlsx")
    feats = d[(d.file == "cell_quarter_dataset") & (d.is_feature_for_next_hour_target)]["column"].tolist()
    feats = [c for c in feats if c in df.columns and df[c].dtype.kind in "fiub"]

    v = df[df.target_next_q_valid == 1].copy()
    tr, te = v[v.quarter_index < 2026 * 4], v[v.quarter_index >= 2026 * 4]
    y_tr, y_te = tr[TARGET].astype(int).values, te[TARGET].astype(int).values

    res = {
        "scope": "Data-readiness check, not a model deliverable.",
        "split": {"train": "2023Q1-2025Q4", "test": "2026Q1 (real held-out quarter)",
                   "n_train": int(len(tr)), "n_test": int(len(te)),
                   "test_positive_rate": round(float(y_te.mean()), 4)},
        "n_features": len(feats),
        "baselines": {},
    }

    # 1. base rate — no information at all
    res["baselines"]["prior_only"] = metrics(y_te, np.full(len(te), y_te.mean()))
    # 2. persistence — assume next quarter looks like this one
    res["baselines"]["persistence"] = metrics(y_te, te["is_below_area_norm"].values.astype(float))
    # 3. the cell's own history
    res["baselines"]["historical_rate"] = metrics(
        y_te, te["below_area_norm_rate_to_date"].fillna(y_tr.mean()).values)
    # 4. current percentile — a smarter single number
    res["baselines"]["current_percentile"] = metrics(y_te, -te["dl_percentile_in_quarter"].values)

    # 5. trained model
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                      random_state=42, n_jobs=4)
    m.fit(tr[feats], y_tr)
    res["model_xgboost"] = metrics(y_te, m.predict_proba(te[feats])[:, 1])

    best_base = max(res["baselines"].values(), key=lambda x: x["pr_auc"])["pr_auc"]
    lift = res["model_xgboost"]["pr_auc"] / best_base
    res["verdict"] = {
        "best_baseline_pr_auc": best_base,
        "model_pr_auc": res["model_xgboost"]["pr_auc"],
        "lift_over_best_baseline": round(float(lift), 3),
        "learnable": bool(lift > 1.10),
        "reading": ("The data carries signal a model can exploit beyond trivial rules."
                    if lift > 1.10 else
                    "The model does NOT meaningfully beat a trivial rule. Report the baseline, "
                    "not the model — and say so."),
    }

    imp = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:12]
    res["top_features"] = [{"feature": f, "gain": round(float(g), 4)} for f, g in imp]

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
