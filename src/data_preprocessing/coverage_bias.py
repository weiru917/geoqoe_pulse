"""
Ookla coverage-bias tooling.

THE PROBLEM THIS EXISTS FOR
---------------------------
Ookla covers ~60% of the study grid, and that coverage is NOT random. The
modelling team's own diagnostic found ROC-AUC 0.976 / PR-AUC 0.985 for
predicting `has_real_ookla_label` from the GIS features alone — i.e. you
can tell almost perfectly which cells Ookla sees just by looking at
population, buildings, roads and POIs.

That has a hard consequence for the README's Option A workflow (train on
labelled cells, predict the rest): those predictions are EXTRAPOLATION
outside the training distribution, not interpolation. It is the best
available explanation for the static model's near-zero spatial
out-of-fold R^2, and it is a property of the DATA, not of the model.

Telling the modelling team about it in a README is not enough — they need
columns they can weight and filter with. This module produces three:

  coverage_propensity
      Out-of-fold predicted probability that a cell has a real Ookla
      label, from GIS features only. Use as an inverse-propensity weight
      when training on labelled cells, so cells resembling the unlabelled
      population count for more.

  in_common_support
      True where a cell's propensity falls inside the range actually
      spanned by the labelled cells (5th-95th percentile). Predictions
      for unlabelled cells OUTSIDE common support are extrapolation and
      should be flagged in the dashboard rather than shown at the same
      confidence as the rest.

  ookla_label_confidence
      0-1 confidence in the label itself, from the real test count
      behind it. The median labelled cell rests on only ~34 tests and the
      p25 on ~8, so treating every label as equally reliable is wrong.

NONE OF THESE ARE PREDICTORS. They are weighting/diagnostic columns and
are excluded from the default feature lists — training on
coverage_propensity would just teach a model to recognise which rows have
labels.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features used to model coverage. Deliberately GIS-only: no QoE-derived
# column may enter, or the propensity becomes circular with the target it
# is meant to help correct for.
COVERAGE_FEATURE_PREFIXES = (
    "population_", "building_", "built_up_", "avg_building_",
    "poi_", "road_", "intersection_", "dist_to_", "tower_",
    "opencellid_", "viirs_", "landcover_", "demand_pressure_base",
)

COMMON_SUPPORT_LOW_PCT = 5.0
COMMON_SUPPORT_HIGH_PCT = 95.0


def _coverage_features(df: pd.DataFrame) -> list:
    import pandas.api.types as ptypes

    cols = []
    for c in df.columns:
        if not c.startswith(COVERAGE_FEATURE_PREFIXES):
            continue
        if c.endswith(("_is_synthetic", "_source")):
            continue
        if not ptypes.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def compute_coverage_bias_columns(baseline_df: pd.DataFrame,
                                   group_col: str = "h3_parent_res6",
                                   seed: int = 42) -> pd.DataFrame:
    """
    Add coverage_propensity / in_common_support / ookla_label_confidence
    to the static QoE baseline.

    The propensity is fitted OUT OF FOLD with a spatial grouping
    (StratifiedGroupKFold on `group_col`), so a cell's own neighbourhood
    never trains the model that scores it — the same discipline the
    README demands for the QoE model itself. An in-sample propensity
    would be optimistic and would defeat the purpose.
    """
    df = baseline_df.copy()
    if "has_real_ookla_label" not in df.columns:
        logger.warning("has_real_ookla_label missing — skipping coverage-bias columns.")
        return df

    features = _coverage_features(df)
    if not features:
        logger.warning("No GIS coverage features found — skipping coverage-bias columns.")
        return df

    X = df[features].fillna(0.0).values.astype(float)
    y = df["has_real_ookla_label"].astype(int).values
    groups = df[group_col].values if group_col in df.columns else np.arange(len(df))

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        logger.warning("scikit-learn unavailable — skipping coverage-bias columns.")
        return df

    oof = np.full(len(df), np.nan)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    df["coverage_propensity"] = oof

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)

    # Common support: the propensity range the LABELLED cells actually
    # span. An unlabelled cell outside it has no comparable training row.
    labelled_prop = oof[y == 1]
    low = float(np.nanpercentile(labelled_prop, COMMON_SUPPORT_LOW_PCT))
    high = float(np.nanpercentile(labelled_prop, COMMON_SUPPORT_HIGH_PCT))
    df["in_common_support"] = (df["coverage_propensity"] >= low) & (df["coverage_propensity"] <= high)

    # Label confidence from the real test count behind each label.
    if "ookla_tests" in df.columns:
        tests = df["ookla_tests"].fillna(0.0)
        # log-scaled and capped: 1 test is weak, ~100 tests is solid, and
        # beyond that extra tests add little. NaN where unlabelled — a
        # cell with no label has no label confidence, and 0 would wrongly
        # read as "confidently bad".
        conf = np.log1p(tests) / np.log1p(100.0)
        df["ookla_label_confidence"] = np.where(
            df["has_real_ookla_label"], np.clip(conf, 0.0, 1.0), np.nan
        )
    else:
        df["ookla_label_confidence"] = np.nan

    # A three-way tier the dashboard can render directly and the modelling
    # team can filter on. The point is to make "we cannot say" a deliberate,
    # visible answer rather than something that happens by default when an
    # extrapolated estimate is displayed with the same confidence as a
    # measurement.
    #
    #   measured   — a real Ookla label exists. Show the number.
    #   estimable  — no label, but the cell resembles labelled cells
    #                (inside common support), so a model prediction is
    #                interpolation and can be shown WITH a confidence caveat.
    #   unknown    — no label and outside common support. A prediction here
    #                is extrapolation into a region the model never saw.
    #                Show it as unknown, not as a low estimate.
    df["coverage_tier"] = np.select(
        [df["has_real_ookla_label"],
         (~df["has_real_ookla_label"]) & df["in_common_support"]],
        ["measured", "estimable"],
        default="unknown",
    )

    n_unlabelled = int((~df["has_real_ookla_label"]).sum())
    n_unlabelled_outside = int(((~df["has_real_ookla_label"]) & (~df["in_common_support"])).sum())

    logger.info(
        "Coverage bias: GIS features predict Ookla coverage at ROC-AUC %.3f / PR-AUC %.3f — the "
        "labelled cells are NOT a random sample. Common support (labelled p%.0f-p%.0f propensity) "
        "= [%.3f, %.3f]; %d of %d unlabelled cells (%.1f%%) fall OUTSIDE it and are extrapolation, "
        "not interpolation. coverage_propensity is exported for inverse-propensity weighting.",
        auc, ap, COMMON_SUPPORT_LOW_PCT, COMMON_SUPPORT_HIGH_PCT, low, high,
        n_unlabelled_outside, n_unlabelled,
        100 * n_unlabelled_outside / max(n_unlabelled, 1),
    )

    tiers = df["coverage_tier"].value_counts().to_dict()
    logger.info(
        "Coverage tiers: %d measured, %d estimable (prediction is interpolation), "
        "%d unknown (extrapolation — present as 'insufficient data', not as an estimate).",
        tiers.get("measured", 0), tiers.get("estimable", 0), tiers.get("unknown", 0),
    )

    df.attrs["coverage_bias_diagnostics"] = {
        "coverage_tiers": tiers,
        "tier_guidance": {
            "measured": "Real Ookla label. Show the value.",
            "estimable": "No label, but inside the labelled cells' feature range. A model "
                          "prediction is interpolation — show it with a confidence caveat.",
            "unknown": "No label and outside common support. A prediction is extrapolation "
                        "into unseen territory. Present as 'insufficient data' rather than as "
                        "a number — an honest gap reads better than a confident guess.",
        },
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "common_support_low": low,
        "common_support_high": high,
        "n_unlabelled": n_unlabelled,
        "n_unlabelled_outside_common_support": n_unlabelled_outside,
        "n_features_used": len(features),
        "interpretation": (
            "An AUC well above 0.5 means Ookla coverage is predictable from GIS features, so "
            "training on labelled cells and predicting unlabelled ones is extrapolation. Weight "
            "by coverage_propensity and flag predictions outside common support."
        ),
    }
    return df
