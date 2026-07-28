"""
GeoQoE Pulse Target Variable Generator.
Computes ITU-T G.107 & Weber-Fechner composite QoE score, MOS, binary degradation, and severity class targets.
"""
import numpy as np
import pandas as pd
from config import QOE_THRESHOLDS

def compute_composite_qoe_targets(df):
    """
    Derive continuous QoE Score (0-100), MOS (1.0-5.0), and categorical degradation targets.
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
    
    raw_qoe = 0.5 * s_dl + 0.3 * s_ul - 0.4 * penalty_rtt - 0.4 * penalty_loss
    qoe_score = np.clip(raw_qoe, 0.0, 100.0)
    
    mos = 1.0 + 4.0 * (qoe_score / 100.0)
    
    t = QOE_THRESHOLDS
    is_degraded = (
        (dl < t["dl_degraded_mbps"]) | 
        (rtt > t["rtt_degraded_ms"])
    ).astype(int)
    
    def classify_severity(row):
        if row["dl_median"] < 5.0 or row["rtt_median"] > 150.0:
            return 2
        elif row["dl_median"] < 15.0 or row["rtt_median"] > 80.0:
            return 1
        else:
            return 0

    severity_class = df.apply(classify_severity, axis=1)
    
    sample_counts = df["test_sample_count"] if "test_sample_count" in df.columns else pd.Series(10, index=df.index)
    confidence_score = np.minimum(1.0, sample_counts / 10.0)
    
    df["target_qoe_score"] = qoe_score
    df["target_mos"] = mos
    df["target_is_degraded"] = is_degraded
    df["target_degradation_class"] = severity_class
    df["qoe_confidence_score"] = confidence_score
    
    return df
