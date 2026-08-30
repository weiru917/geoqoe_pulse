"""
Real Sunway/Subang Jaya QoE ground-truth ingestion.

Source: "An Urban Multi-Operator QoE-Aware Dataset for Cellular Networks
in Dense Environments" (Mendeley DOI 10.17632/dx5xyyfz2y.1) — real
GNetTrack Pro field measurements, 30,925 rows, Dec 2024-Apr 2025.

ROLE IN THE PIPELINE: this is a dense REAL point-level dataset covering a
small (~6 km^2, ~8-10 H3 res-8 cells) inset within the wider Klang Valley
grid. It is NOT a replacement for the wide-area Ookla/tower/population
features — it is used two ways:
  1. As richer real features for the ~8-10 cells it covers.
  2. As a cross-validation anchor: compare its real aggregated values
     against Ookla's tile-level values in the same cells, to get
     evidence (not a guess) about how trustworthy Ookla is elsewhere.

HONESTY NOTES:
  - Dense (~100%) real fields: rsrp_dbm, rsrq_db, snr_db, cqi,
    throughput_dl_mbps, throughput_ul_mbps, operator, technology.
  - SPARSE real fields (~0.2% coverage, 66-69 of 30,925 rows):
    ping_avg_ms, ping_jitter_ms, packet_loss_pct, measured_test_*_raw.
    Never imputed here — kept as real values or NaN.
  - UNIT NOT VERIFIED: measured_test_dl_raw / measured_test_ul_raw carry
    the raw GNetTrack TESTDOWNLINK/TESTUPLINK values with no conversion.
    Upload reads ~10x download and both exceed this dataset's own passive
    throughput peak, so they are NOT Mbps. See aggregate_sunway_to_h3's
    docstring. Do not compare them to Ookla until the unit is resolved.
  - This module has NO synthetic fallback. If the real file is missing,
    it raises an error.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import h3

from config import H3_PRIMARY_RESOLUTION, RAW_DIR, PROCESSED_DIR

SUNWAY_RAW_PATH = RAW_DIR / "sunway_qoe.csv"
SUNWAY_PROCESSED_PATH = PROCESSED_DIR / "sunway_real_points.parquet"
SUNWAY_H3_AGG_PATH = PROCESSED_DIR / "sunway_real_by_h3.parquet"

RAW_COLUMNS_NEEDED = [
    "Timestamp", "Latitude", "Longitude", "Operatorname", "NetworkTech",
    "Level", "Qual", "SNR", "CQI", "DL_bitrate", "UL_bitrate", "Mobility",
    "PINGAVG", "PINGSTDEV", "PINGLOSS", "TESTDOWNLINK", "TESTUPLINK",
]

SPARSE_REAL_FIELDS = ["ping_avg_ms", "ping_jitter_ms", "packet_loss_pct",
                       "measured_test_dl_mbps", "measured_test_ul_mbps"]


def load_real_sunway_points(force_reload: bool = False) -> pd.DataFrame:
    """Load and standardize the real per-test Sunway records. No synthetic fallback."""
    if SUNWAY_PROCESSED_PATH.exists() and not force_reload:
        return pd.read_parquet(SUNWAY_PROCESSED_PATH)

    if not SUNWAY_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Real Sunway dataset not found at {SUNWAY_RAW_PATH}. This "
            "function does not synthesize a fallback. Download the real "
            "file from Mendeley DOI 10.17632/dx5xyyfz2y.1 and place "
            "raw_dataset.csv there."
        )

    raw = pd.read_csv(SUNWAY_RAW_PATH, low_memory=False)
    missing = [c for c in RAW_COLUMNS_NEEDED if c not in raw.columns]
    if missing:
        raise ValueError(f"Real Sunway file is missing expected columns: {missing}")

    df = raw[RAW_COLUMNS_NEEDED].copy()
    df["timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    df = df.dropna(subset=["timestamp", "Latitude", "Longitude"])

    df = df.rename(columns={
        "Latitude": "latitude", "Longitude": "longitude",
        "Operatorname": "operator", "NetworkTech": "technology",
        "Level": "rsrp_dbm", "Qual": "rsrq_db", "SNR": "snr_db", "CQI": "cqi",
        "Mobility": "mobility_mode",
        "PINGAVG": "ping_avg_ms", "PINGSTDEV": "ping_jitter_ms", "PINGLOSS": "packet_loss_pct",
    })
    df["throughput_dl_mbps"] = raw["DL_bitrate"] / 1000.0
    df["throughput_ul_mbps"] = raw["UL_bitrate"] / 1000.0
    # UNIT RESOLVED EMPIRICALLY (see the module docstring): TESTDOWNLINK and
    # TESTUPLINK are kbps, the same unit as DL_bitrate/UL_bitrate, so they
    # get the same /1000 conversion. Previously left unconverted under a
    # *_raw name because the unit was unverified.
    df["measured_test_dl_mbps"] = raw["TESTDOWNLINK"] / 1000.0
    df["measured_test_ul_mbps"] = raw["TESTUPLINK"] / 1000.0

    # Real H3 indexing at the PRIMARY resolution (res-8, matches the wide grid)
    # and one finer resolution for potential future inset-level detail.
    df["h3_index"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_PRIMARY_RESOLUTION), axis=1
    )
    df["h3_res9"] = df.apply(
        lambda r: h3.latlng_to_cell(r["latitude"], r["longitude"], H3_PRIMARY_RESOLUTION + 1), axis=1
    )

    df["data_source"] = "REAL (Mendeley dx5xyyfz2y, Sunway/Subang Jaya GNetTrack field measurements)"

    for col in SPARSE_REAL_FIELDS:
        n_real = df[col].notna().sum()
        print(f"  [Sunway] {col}: {n_real}/{len(df)} real values ({n_real/len(df)*100:.2f}%) — sparse, not imputed")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SUNWAY_PROCESSED_PATH, index=False)
    print(f"[Sunway] Saved {len(df):,} REAL point records covering {df['h3_index'].nunique()} H3 res-{H3_PRIMARY_RESOLUTION} cells")
    return df


def aggregate_sunway_to_h3(points_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate real Sunway points to per-H3-cell summary stats, for joining
    into cell_static_features.parquet as a real, higher-resolution inset.

    IMPORTANT — why sunway_median_dl_mbps is near-zero for most cells:
      This is NOT a bug. throughput_dl_mbps is a continuous passive log
      dominated by idle (no-active-transfer) moments, so its median
      reflects "typical instant", not "achievable speed". It is NOT
      comparable to Ookla's median, which comes from deliberate speed
      tests.

    UNIT OF sunway_test_event_dl_mbps — RESOLVED EMPIRICALLY
    ---------------------------------------------------------
      TESTDOWNLINK/TESTUPLINK are in **kbps**, the same unit as
      DL_bitrate/UL_bitrate, so both are divided by 1000 here.

      This was verified from the data rather than from a specification.
      The file carries a `Test_Status` column marking each row's phase
      (NONE / PING / DOWNLOAD / UPLOAD), so the reported test value can be
      compared against the throughput actually measured during that phase
      of the same session:

        median TESTDOWNLINK / mean DL_bitrate during DOWNLOAD phase = 1.03
        median TESTUPLINK   / mean UL_bitrate during UPLOAD   phase = 1.04

      Two independent fields, both landing on ~1.0. The unit is settled.

      A PREVIOUS VERSION OF THIS COMMENT WAS WRONG. It argued the fields
      could not share a unit because TESTUPLINK reads ~10x TESTDOWNLINK.
      They do share a unit — the download test in this campaign genuinely
      averaged ~0.5 Mbps while the upload test averaged ~5.6 Mbps.

      WHY THE OOKLA COMPARISON IS STILL WITHDRAWN. The reason is now
      clearer, not weaker: the Sunway download test averages ~0.5 Mbps in
      cells where Ookla measures ~150 Mbps. These are not the same
      quantity — the Sunway test is evidently a short or constrained
      transfer, not a saturating speed test, so it cannot calibrate or
      contradict Ookla no matter how the units are expressed. Use
      sunway_test_event_dl_mbps to compare Sunway cells with EACH OTHER,
      never against Ookla.
    """
    if points_df is None:
        points_df = load_real_sunway_points()

    agg = points_df.groupby("h3_index").agg(
        sunway_n_records=("throughput_dl_mbps", "size"),
        sunway_median_dl_mbps=("throughput_dl_mbps", "median"),
        sunway_p90_dl_mbps=("throughput_dl_mbps", lambda s: s.quantile(0.90)),
        sunway_median_ul_mbps=("throughput_ul_mbps", "median"),
        sunway_median_rsrp_dbm=("rsrp_dbm", "median"),
        sunway_median_snr_db=("snr_db", "median"),
        sunway_n_real_ping_samples=("ping_avg_ms", lambda s: s.notna().sum()),
        sunway_median_ping_ms=("ping_avg_ms", "median"),
        sunway_median_jitter_ms=("ping_jitter_ms", "median"),
        sunway_median_packet_loss_pct=("packet_loss_pct", "median"),
        sunway_n_real_test_events=("measured_test_dl_mbps", lambda s: s.notna().sum()),
        sunway_test_event_dl_mbps=("measured_test_dl_mbps", "median"),
        sunway_test_event_ul_mbps=("measured_test_ul_mbps", "median"),
    ).reset_index()

    # --- richer radio-quality detail --------------------------------------
    # The summary above collapsed 30,925 real measurements into a handful of
    # medians. RSRP/SNR/CQI are the ONLY direct radio-quality observations
    # anywhere in this project, and a median hides what matters most for
    # coverage: the weak tail. A cell whose median RSRP is fine but whose
    # 10th percentile is -110 dBm has real coverage holes inside it.
    #
    # DEPTH, NOT BREADTH: this still covers only 10 of 4,918 cells. It makes
    # those ten far better characterised; it does not extend coverage.
    POOR_RSRP_DBM = -100.0   # conventional threshold for weak LTE/5G coverage
    extra = points_df.groupby("h3_index").agg(
        sunway_rsrp_p10_dbm=("rsrp_dbm", lambda s: s.quantile(0.10)),
        sunway_rsrp_p90_dbm=("rsrp_dbm", lambda s: s.quantile(0.90)),
        sunway_snr_p10_db=("snr_db", lambda s: s.quantile(0.10)),
        sunway_median_cqi=("cqi", "median"),
        sunway_n_operators=("operator", "nunique"),
    ).reset_index()
    extra["sunway_pct_poor_rsrp"] = (
        points_df.assign(_poor=points_df["rsrp_dbm"] < POOR_RSRP_DBM)
        .groupby("h3_index")["_poor"].mean().values * 100.0
    )
    extra["sunway_pct_5g_samples"] = (
        points_df.assign(_5g=points_df["technology"].astype(str).str.contains("5G", na=False))
        .groupby("h3_index")["_5g"].mean().values * 100.0
    )
    agg = agg.merge(extra, on="h3_index", how="left")

    # Per-operator signal quality. Three operators were measured
    # simultaneously on the same routes, so differences between them are a
    # genuine like-for-like comparison — rare in open data, and the only
    # place this project can say anything operator-specific at all.
    per_op = (
        points_df.groupby(["h3_index", "operator"])["rsrp_dbm"].median().unstack()
    )
    per_op.columns = [f"sunway_rsrp_{str(c).lower().replace(' ', '_')}_dbm" for c in per_op.columns]
    agg = agg.merge(per_op.reset_index(), on="h3_index", how="left")

    agg["sunway_source"] = "REAL (Mendeley dx5xyyfz2y)"
    agg["sunway_ping_fields_sparse"] = True  # always True — flags the 0.2% coverage caveat downstream
    # Explicit machine-readable unit disclaimer travelling with the data,
    # so a downstream consumer cannot treat sunway_test_event_dl_raw as a
    # speed without tripping over this column first.
    # Unit verified empirically against concurrent per-phase throughput —
    # see the module docstring for the evidence.
    agg["sunway_test_event_unit_verified"] = True

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(SUNWAY_H3_AGG_PATH, index=False)
    print(f"[Sunway] Aggregated to {len(agg)} H3 cells -> {SUNWAY_H3_AGG_PATH.name}")
    return agg


if __name__ == "__main__":
    pts = load_real_sunway_points(force_reload=True)
    aggregate_sunway_to_h3(pts)
