"""
Per-application QoE profiles — the proposal's "application profiles"
function, finally built.

WHY THIS MATTERS: a single "degraded" flag hides that a cell can be
perfectly fine for one thing and unusable for another. A cell at 8 Mbps
with 30 ms latency streams video badly but carries voice calls
flawlessly; a cell at 200 Mbps with 180 ms latency is the reverse. An
operator deciding where to intervene needs to know WHICH experience is
broken, because the fix differs.

TWO SEPARATE SOURCES, DELIBERATELY NOT CONFLATED
------------------------------------------------
1. THRESHOLDS — what each application technically needs — come from
   established engineering guidance, not from any survey. Latency limits
   follow ITU-T G.114's one-way delay guidance for interactive speech;
   throughput floors follow typical codec and streaming requirements.
   These say what the network must deliver.

2. WEIGHTS — how much Malaysian users care — come from MCMC's Broadband
   Quality of Experience Survey 2024 (mobile broadband results). These
   say which failures matter most HERE. Voice calls score highest on
   importance (4.31/5) and show the largest satisfaction gap (-0.34),
   so a cell failing voice is weighted more heavily than one failing
   gaming (importance 3.47, gap +0.33 — expectations already exceeded).

Keeping these apart matters: the thresholds are engineering facts, the
weights are a survey of opinion. Presenting a survey as if it set
technical requirements would be a category error.

WHAT CANNOT BE EVALUATED FROM REAL DATA — READ THIS
---------------------------------------------------
Ookla provides download, upload and latency. It does NOT provide jitter
or packet loss. Voice and video calls depend heavily on both, so their
assessment here is PARTIAL and OPTIMISTIC: a cell can pass on latency
and throughput while still delivering choppy audio. Every profile
therefore carries an `_evaluable` count recording how many of its
criteria could actually be checked, and the module never claims a cell
"supports" an application — only that it meets the criteria that were
measurable.

The simulated hourly panel does contain jitter and loss columns, but
those are generated, so evaluating voice quality against them would be
measuring the formula, not the network. Profiles are computed on real
tables only.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold sources are engineering guidance. Importance/gap are MCMC
# BQoES 2024 mobile-broadband results, used only for weighting.
APPLICATION_PROFILES = {
    "voice_call": {
        "label": "Online voice calls",
        "min_dl_mbps": 0.1, "min_ul_mbps": 0.1, "max_rtt_ms": 150.0,
        "bqoes_importance": 4.31, "bqoes_gap": -0.34,
        "driver": "latency, jitter and loss — throughput is almost irrelevant",
        "unmeasurable": ["jitter", "packet_loss"],
        "top_complaint": "choppy or broken voice (49.0% of dissatisfied users)",
    },
    "web_browsing": {
        "label": "Web browsing",
        "min_dl_mbps": 5.0, "min_ul_mbps": 0.5, "max_rtt_ms": 100.0,
        "bqoes_importance": 4.24, "bqoes_gap": -0.31,
        "driver": "latency first — a page load is many round trips — then throughput",
        "unmeasurable": [],
        "top_complaint": "waiting for pages to load (24.0%)",
    },
    "video_call": {
        "label": "Online video calls",
        "min_dl_mbps": 2.0, "min_ul_mbps": 1.5, "max_rtt_ms": 150.0,
        "bqoes_importance": 4.09, "bqoes_gap": -0.19,
        "driver": "throughput and jitter together, in both directions",
        "unmeasurable": ["jitter", "packet_loss"],
        "top_complaint": "video freezing or disconnecting (38.9%)",
    },
    "video_streaming": {
        "label": "Video streaming (HD)",
        "min_dl_mbps": 5.0, "min_ul_mbps": 0.0, "max_rtt_ms": 400.0,
        "bqoes_importance": 3.88, "bqoes_gap": 0.04,
        "driver": "sustained download; buffering absorbs latency",
        "unmeasurable": [],
        "top_complaint": "video stalls and buffering (52.4%)",
    },
    "online_gaming": {
        "label": "Online gaming",
        "min_dl_mbps": 1.0, "min_ul_mbps": 0.5, "max_rtt_ms": 100.0,
        "bqoes_importance": 3.47, "bqoes_gap": 0.33,
        "driver": "latency above all; bandwidth needs are tiny",
        "unmeasurable": ["jitter"],
        "top_complaint": "animation stuttering (42.9%)",
    },
}


def compute_application_profiles(df: pd.DataFrame,
                                  dl_col: str = "ookla_dl_mbps",
                                  ul_col: str = "ookla_ul_mbps",
                                  rtt_col: str = "ookla_rtt_ms") -> pd.DataFrame:
    """
    Add one `app_<name>_ok` column per application, plus a weighted
    user-impact score.

    A cell is marked 1 only when it meets EVERY measurable criterion,
    0 when it fails at least one, and NaN when the inputs needed are
    missing — never 0 by default, since "we don't know" and "it fails"
    are different answers and only one of them justifies sending an
    engineer.
    """
    out = df.copy()
    has_ul = ul_col in out.columns

    weight_total = sum(p["bqoes_importance"] for p in APPLICATION_PROFILES.values())
    impact = pd.Series(0.0, index=out.index)
    weight_used = pd.Series(0.0, index=out.index)

    for name, p in APPLICATION_PROFILES.items():
        dl, rtt = out[dl_col], out[rtt_col]
        known = dl.notna() & rtt.notna()
        ok = (dl >= p["min_dl_mbps"]) & (rtt <= p["max_rtt_ms"])
        if has_ul and p["min_ul_mbps"] > 0:
            known &= out[ul_col].notna()
            ok &= out[ul_col] >= p["min_ul_mbps"]

        col = f"app_{name}_ok"
        out[col] = np.where(known, ok.astype(float), np.nan)
        # Impact accumulates only where the answer is known, and the
        # denominator tracks it, so a cell missing one application is not
        # silently scored as if that application passed.
        impact += np.where(known & ~ok, p["bqoes_importance"], 0.0)
        weight_used += np.where(known, p["bqoes_importance"], 0.0)

    # 0 = everything measurable works, 1 = every measurable application fails,
    # weighted by how much Malaysian users say each one matters.
    out["app_user_impact_score"] = np.where(weight_used > 0, impact / weight_used, np.nan)
    app_cols = [f"app_{n}_ok" for n in APPLICATION_PROFILES]
    out["app_n_evaluable"] = out[app_cols].notna().sum(axis=1)
    out["app_n_failing"] = (out[app_cols] == 0).sum(axis=1)

    # The single most important application this cell fails, for the
    # dashboard: "what is actually broken here".
    order = sorted(APPLICATION_PROFILES.items(), key=lambda kv: -kv[1]["bqoes_importance"])
    worst = pd.Series(pd.NA, index=out.index, dtype="object")
    for name, p in reversed(order):
        worst = worst.mask(out[f"app_{name}_ok"] == 0, p["label"])
    out["app_worst_failing"] = worst

    n = int(out["app_n_evaluable"].gt(0).sum())
    logger.info(
        "Application profiles computed for %d rows with usable QoE. Failure rates: %s. "
        "Voice and video-call assessments are PARTIAL — jitter and packet loss are not in "
        "Ookla, so a cell can pass here and still sound broken.",
        n, {k: f"{100 * (out[f'app_{k}_ok'] == 0).sum() / max(n, 1):.1f}%"
            for k in APPLICATION_PROFILES},
    )
    return out


def profile_metadata() -> dict:
    """Provenance block for metadata.json."""
    return {
        "threshold_source": (
            "Engineering guidance — ITU-T G.114 one-way delay for interactive speech, "
            "typical codec and streaming throughput requirements. NOT derived from any survey."
        ),
        "weight_source": (
            "MCMC Broadband Quality of Experience Survey 2024, mobile broadband results. "
            "Used ONLY to weight how much each application failure matters to Malaysian "
            "users — never to set a technical threshold."
        ),
        "known_gap": (
            "Ookla provides download, upload and latency but NOT jitter or packet loss. "
            "Voice and video-call profiles are therefore partial and optimistic: a cell can "
            "meet every measurable criterion and still deliver choppy audio. app_n_evaluable "
            "records how many criteria were actually checkable per row."
        ),
        "applications": {
            k: {"label": v["label"], "min_dl_mbps": v["min_dl_mbps"],
                "max_rtt_ms": v["max_rtt_ms"], "bqoes_importance": v["bqoes_importance"],
                "bqoes_gap": v["bqoes_gap"], "driver": v["driver"],
                "not_measurable": v["unmeasurable"], "top_user_complaint": v["top_complaint"]}
            for k, v in APPLICATION_PROFILES.items()
        },
    }
