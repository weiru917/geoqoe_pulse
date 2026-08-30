"""
GeoQoE Pulse Target Variable Generator.

Computes a custom composite QoE index, a derived 1-5 quality score, and
binary/3-class degradation labels — plus, critically, the 1-hour-ahead
FORECAST versions of the degradation labels, which is the actual MVP target
per the project proposal ("will the same area experience degradation again
in the next time period?").

Honesty note (see also config.SIMULATION_DISCLAIMER): this module computes
targets from whatever dl_median/rtt_median/etc. values it is given. On the
current pipeline those values come from the SIMULATED hourly panel
(data_downloader.build_dynamic_panel) unless you swap in real repeated QoE
measurements. The formulas below are correct; the input data determines
whether the output targets mean anything about real network behaviour.
"""
import numpy as np
import pandas as pd
from config import QOE_THRESHOLDS, QOE_INDEX_WEIGHTS


def compute_qoe_targets(df):
    """
    Derive a continuous composite QoE index (0-100), a derived 1-5 quality
    score, and binary/3-class degradation labels, all from a SINGLE shared
    threshold source (config.QOE_THRESHOLDS) so the binary and 3-class
    labels can never disagree with each other on the same row.

    IMPORTANT — naming and scope, corrected from an earlier version:
    - This is a CUSTOM composite index (logarithmic download/upload utility
      minus exponential latency/loss penalties). It is *inspired by* ideas
      from the Weber-Fechner perceptual-scaling literature and by the shape
      of MOS-style 1-5 scores, but it is NOT a validated ITU-T G.107 E-model
      implementation and must not be presented as one. If your team needs a
      real E-model MOS, implement ITU-T G.107 directly or run a subjective
      (ACR) test panel — this index is not a substitute for either.
    - Renamed from target_qoe_score/target_mos to target_qoe_index/
      target_service_quality_1to5 to stop implying a standards-validated
      metric. Both names still start with "target_" so the pipeline's
      existing "exclude every target_* column from the feature matrix"
      logic continues to work unchanged.
    """
    df = df.copy()

    dl = df["dl_median"]
    ul = df["ul_median"]
    rtt = df["rtt_median"]
    loss = df["loss_median"] if "loss_median" in df.columns else pd.Series(0.0, index=df.index)

    s_dl = np.minimum(100.0, (np.log1p(dl) / np.log1p(50.0)) * 100.0)
    s_ul = np.minimum(100.0, (np.log1p(ul) / np.log1p(20.0)) * 100.0)

    penalty_rtt = 100.0 * (1.0 - np.exp(-np.maximum(0.0, rtt - 20.0) / 50.0))
    penalty_loss = 100.0 * (1.0 - np.exp(-np.maximum(0.0, loss) / 3.0))

    w = QOE_INDEX_WEIGHTS
    # w["download"] + w["upload"] MUST sum to 1.0 (config enforces 0.6+0.4)
    # so the index can actually reach 100 before any penalty is applied —
    # previously these were 0.5+0.3=0.8, silently capping the index at 80
    # and the derived 1-5 score at 4.2 regardless of how good dl/ul were.
    assert abs((w["download"] + w["upload"]) - 1.0) < 1e-9, (
        "QOE_INDEX_WEIGHTS download+upload must sum to 1.0 or the index's "
        "documented 0-100 range becomes unreachable — see config.py."
    )
    raw_qoe = w["download"] * s_dl + w["upload"] * s_ul \
        - w["rtt_penalty"] * penalty_rtt - w["loss_penalty"] * penalty_loss
    qoe_index = np.clip(raw_qoe, 0.0, 100.0)

    service_quality_1to5 = 1.0 + 4.0 * (qoe_index / 100.0)

    # --- Single shared threshold source for BOTH degradation labels ---
    t = QOE_THRESHOLDS
    severity_class = pd.Series(
        np.select(
            [
                (dl < t["dl_severe_mbps"]) | (rtt > t["rtt_severe_ms"]),
                (dl < t["dl_degraded_mbps"]) | (rtt > t["rtt_degraded_ms"]),
            ],
            [2, 1],
            default=0,
        ),
        index=df.index,
    )
    # is_degraded is now DERIVED from severity_class, not computed from a
    # separately hardcoded threshold pair — the two labels are therefore
    # consistent with each other by construction, not just by convention.
    is_degraded = (severity_class >= 1).astype(int)

    sample_counts = df["test_sample_count"] if "test_sample_count" in df.columns else pd.Series(10, index=df.index)
    confidence_score = np.minimum(1.0, sample_counts / 10.0)

    df["target_qoe_index"] = qoe_index
    df["target_service_quality_1to5"] = service_quality_1to5
    df["target_is_degraded"] = is_degraded
    df["target_degradation_class"] = severity_class
    df["qoe_confidence_score"] = confidence_score

    return df


# Backward-compatible alias — DO NOT use in new code. The old name implied a
# score that could reach 100 with the old (buggy) 0.5+0.3 weights; the
# function it now points to fixes that, but keep using compute_qoe_targets()
# directly so the naming stops being a source of confusion.
compute_composite_qoe_targets = compute_qoe_targets


def compute_relative_targets(df, percentile_cutoff: float = None):
    """
    A RELATIVE degradation target, alongside the absolute one.

    WHY THIS EXISTS
    ---------------
    config.QOE_THRESHOLDS marks a cell degraded below 10 Mbps download.
    The real Ookla distribution for this study area has a median of
    175 Mbps and a p25 of 108 Mbps, so the absolute threshold fires on
    only ~2.3% of labelled cells. That makes "degraded" a rare tail event
    defined by a constant that was never calibrated against the data, and
    it cannot express the project's actual question — recurring LOCALISED
    degradation that aggregate statistics hide. A cell running at 40 Mbps
    in a city where its neighbours run at 200 is invisible to a 10 Mbps
    cutoff, yet it is exactly the kind of place an operator should look at.

    WHAT IS COMPUTED
    ----------------
    At each timestamp, every cell is ranked against ALL OTHER CELLS in the
    study area at that same hour:

      target_qoe_percentile_in_area : 0-100 percentile of dl_median among
          all cells at this timestamp. 50 = exactly typical for the city
          right now.
      target_is_below_area_norm     : 1 if that percentile is below
          `percentile_cutoff` (default config.RELATIVE_DEGRADATION_PERCENTILE).

    This is a SPATIALLY relative measure evaluated per timestamp, not a
    per-cell historical one. That choice is deliberate: "worse than the
    rest of the city right now" is the question a prioritisation dashboard
    actually asks, and unlike a per-cell historical threshold it stays
    meaningful for a cell that is consistently bad (such a cell would
    never trip a threshold defined against its own history).

    HONEST LIMITATION ON THE CURRENT PANEL — READ THIS
    --------------------------------------------------
    While the hourly panel is built from ONE shared diurnal curve, an
    anchored cell's value is level(cell) x shape(hour). The cross-cell
    ranking at time t is then determined entirely by level(cell), because
    shape(hour) multiplies every cell identically and a common positive
    factor does not change an ordering. So on the CURRENT panel these
    columns are effectively TIME-INVARIANT: a cell below the area norm at
    one hour is below it at every hour.

    That is a property of the panel, not of the definition. Once real
    per-district curves exist (see mlab_diurnal_shape.load_district_shape_
    lookups), the ranking genuinely moves through the week and these
    columns start carrying spatio-temporal information. They are added now
    so the target contract is stable and the modelling team can build
    against the definition that will still be correct when the data
    improves. data_validator.validate_relative_targets reports the
    measured time-variation each run so nobody has to take this on trust.
    """
    from config import RELATIVE_DEGRADATION_PERCENTILE

    if percentile_cutoff is None:
        percentile_cutoff = RELATIVE_DEGRADATION_PERCENTILE

    df = df.copy()
    if "dl_median" not in df.columns or "timestamp" not in df.columns:
        return df

    # Cross-cell percentile within each timestamp. `rank(pct=True)` gives
    # 0-1; scale to 0-100 for readability in the dashboard.
    df["target_qoe_percentile_in_area"] = (
        df.groupby("timestamp")["dl_median"].rank(pct=True) * 100.0
    )
    df["target_is_below_area_norm"] = (
        df["target_qoe_percentile_in_area"] < percentile_cutoff
    ).astype(int)

    return df


def compute_next_hour_targets(df, horizon_hours: int = 1):
    """
    Shift the degradation labels forward within each cell to create genuine
    FORECAST targets: "will THIS cell be degraded `horizon_hours` from now",
    using only information available at or before the current hour as
    features. This was entirely missing from the previous version, which
    only supported same-hour classification ("is it degraded right now,
    using features measured in that same hour") — not a forecast at all,
    despite forecasting being the proposal's actual MVP question.

    Requires df to already contain target_is_degraded, target_degradation_
    class, and target_qoe_index (i.e. call compute_qoe_targets() first), and
    to be sorted by (h3_index, timestamp) — the caller is responsible for
    that ordering (run_pipeline.py sorts once, upstream, and relies on it).

    Adds:
      target_is_degraded_next_{h}h        — the label to predict
      target_degradation_class_next_{h}h  — 3-class version, same horizon
      target_qoe_index_next_{h}h          — continuous version, same horizon
      target_next_{h}h_valid              — 0 for each cell's final
          `horizon_hours` rows, where no future observation exists to shift
          in (these rows must be dropped before training, not treated as a
          real "not degraded" case — pd.shift() would otherwise leave NaN,
          which is correct, but this explicit flag makes the reason obvious
          without requiring every downstream user to remember why).

    The target built here is config.PANEL_DEMO_TARGET, not the primary one.
    It drives the dashboard's hourly animation and is reconstructable from
    (cell, hour-of-week) — see config.PANEL_STRUCTURE_DISCLOSURE. The primary
    ML target is config.PRIMARY_ML_TARGET on config.PRIMARY_ML_TABLE.
    """
    df = df.sort_values(["h3_index", "timestamp"]).copy()
    suffix = f"next_{horizon_hours}h"

    grouped = df.groupby("h3_index", sort=False)
    df[f"target_is_degraded_{suffix}"] = grouped["target_is_degraded"].shift(-horizon_hours)
    df[f"target_degradation_class_{suffix}"] = grouped["target_degradation_class"].shift(-horizon_hours)
    df[f"target_qoe_index_{suffix}"] = grouped["target_qoe_index"].shift(-horizon_hours)
    if "target_is_below_area_norm" in df.columns:
        df[f"target_is_below_area_norm_{suffix}"] = grouped["target_is_below_area_norm"].shift(-horizon_hours)
        df[f"target_qoe_percentile_in_area_{suffix}"] = grouped["target_qoe_percentile_in_area"].shift(-horizon_hours)
    df[f"target_{suffix}_valid"] = df[f"target_is_degraded_{suffix}"].notna().astype(int)

    return df
