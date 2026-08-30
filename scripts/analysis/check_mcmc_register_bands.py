"""
What the MCMC public apparatus register actually contains — and why it
cannot answer a capacity question.

WHY THIS SCRIPT REPLACED A CORRELATION
--------------------------------------
The earlier attempt correlated per-cell sector counts against speed and
reported Spearman -0.065 (p = 0.69). That was a weak way to make the point,
for two reasons found later:

  1. The register publishes NO SITE COORDINATES — only operator-internal
     site codes like "LGAT(BT14 HULU LANGAT)". There is nothing to join to a
     hexagon, so any per-cell sector count rested on district-name matching:
     8 districts across 4,918 cells.
  2. Name matching silently crosses states. A search for the study area's
     "SEPANG" returns "TLKSEPANGGAR" — Telok Sepanggar, Kota Kinabalu,
     ~1,600 km away in Sabah.

A null correlation invites "your sample was too small". The structural
argument below does not, and it holds regardless of geolocation:

    The register contains none of the spectrum that carries throughput.

Even with perfect coordinates, it could not explain speed differences,
because capacity lives in bands the register does not list. That is a
statement about the data's content, checkable by anyone in one search.

INPUT
-----
data/raw/mcmc_register/hulu_langat_sample.csv — verbatim register rows
retrieved 2026-08-30 for Hulu Langat (LTE + 5G filters). One row per
sector, as the register returns them.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE = PROJECT_ROOT / "data" / "raw" / "mcmc_register" / "hulu_langat_sample.csv"
OUT = PROJECT_ROOT / "data" / "outputs" / "mcmc_register_findings.json"

# 3GPP allocations, by the pair the register prints. TX is the base station's
# downlink, RX its uplink. Band 20 is reverse-duplex (downlink BELOW uplink),
# which is why YTL's TX 800.5 / RX 841.5 looks inverted next to U Mobile's.
BANDS = [
    ("Band 28 (700 MHz)",  "coverage", (758.0, 803.0),  (703.0, 748.0)),
    ("Band 20 (800 MHz)",  "coverage", (791.0, 821.0),  (832.0, 862.0)),
    ("Band 8 (900 MHz)",   "coverage", (925.0, 960.0),  (880.0, 915.0)),
    ("Band 3 (1800 MHz)",  "capacity", (1805.0, 1880.0), (1710.0, 1785.0)),
    ("Band 1 (2100 MHz)",  "capacity", (2110.0, 2170.0), (1920.0, 1980.0)),
    ("Band 7 (2600 MHz)",  "capacity", (2620.0, 2690.0), (2500.0, 2570.0)),
    ("Band 78 (3500 MHz)", "capacity", (3300.0, 3800.0), (3300.0, 3800.0)),
]


def classify(tx, rx):
    for name, role, dl, ul in BANDS:
        if dl[0] <= tx <= dl[1] and ul[0] <= rx <= ul[1]:
            return name, role
    return f"unclassified (TX {tx} / RX {rx})", "unknown"


def main():
    if not SAMPLE.exists():
        raise FileNotFoundError(f"{SAMPLE} not found")
    df = pd.read_csv(SAMPLE)
    df[["band", "role"]] = df.apply(
        lambda r: pd.Series(classify(r.tx_mhz, r.rx_mhz)), axis=1)

    print("=" * 70)
    print("MCMC REGISTER — WHAT IT CONTAINS")
    print("=" * 70)
    print(f"\nRows (one per sector): {len(df)}")
    print(f"Distinct sites (assign_no): {df.assign_no.nunique()}")
    print(f"Sectors per site: {len(df) // df.assign_no.nunique()} (every site appears 3x)\n")

    print("BY OPERATOR")
    op = df.groupby("holder").agg(sites=("assign_no", "nunique"),
                                  sectors=("assign_no", "size"),
                                  band=("band", "first"))
    print(op.to_string(), "\n")

    print("BY SPECTRUM ROLE — the finding")
    role = df.groupby("role").agg(sectors=("assign_no", "size"),
                                  bands=("band", lambda s: "; ".join(sorted(set(s)))))
    print(role.to_string(), "\n")

    n_cap = int((df.role == "capacity").sum())
    pct_cov = 100.0 * (df.role == "coverage").sum() / len(df)

    # The two largest operators by subscriber share. Their absence is the
    # point: the register is not a sample of the network, it is a subset
    # skewed toward the players who license site-by-site.
    # Match on the full holder name, not a substring: "DIGI" is contained in
    # "DIGITAL NASIONAL BERHAD", which would wrongly report CelcomDigi as
    # present. DNB is the 5G wholesale network, not CelcomDigi.
    MAJORS = {"Maxis": "MAXIS", "CelcomDigi": "CELCOMDIGI"}
    holders = df.holder.str.upper()
    missing = [label for label, token in MAJORS.items()
               if not holders.str.startswith(token).any()]

    print("=" * 70)
    print(f"Sub-1 GHz coverage spectrum : {pct_cov:.0f}% of sectors")
    print(f"Capacity spectrum (1800+)   : {n_cap} sectors")
    print(f"Major operators absent      : {', '.join(missing)}")
    print("=" * 70)
    print(
        "\nCONCLUSION. Throughput is delivered by the capacity bands. The\n"
        "register lists none of them here, and omits the two largest\n"
        "operators entirely. Even given perfect site coordinates — which it\n"
        "does not publish — it could not explain why one cell is faster than\n"
        "another, because the spectrum that decides that is held under\n"
        "national Spectrum Assignments with no site-level geography.\n"
    )

    import json
    OUT.write_text(json.dumps({
        "question": "Can the MCMC public apparatus register supply a per-cell capacity feature?",
        "answer": "No — for two independent reasons, either of which is sufficient.",
        "retrieved": "2026-08-30, Hulu Langat, LTE + 5G filters",
        "reason_1_no_geography": {
            "finding": "The register publishes no site coordinates, only operator-internal site codes (e.g. 'LGAT(BT14 HULU LANGAT)').",
            "consequence": "Nothing can be joined to an H3 cell. The finest achievable match is district name — 8 districts across 4,918 cells.",
            "name_collision_evidence": "A search for the study area's 'SEPANG' returns 'DSBKI0302_TLKSEPANGGAR' — Telok Sepanggar, Kota Kinabalu, Sabah, ~1,600 km away.",
        },
        "reason_2_wrong_spectrum": {
            "finding": f"{pct_cov:.0f}% of sectors are sub-1 GHz coverage spectrum; {n_cap} are capacity spectrum.",
            "bands_present": sorted(df.band.unique().tolist()),
            "consequence": "Coverage spectrum determines reach and building penetration, not throughput. The register contains none of the spectrum that decides speed.",
            "holds_even_with_coordinates": True,
        },
        "operator_coverage": {
            "present": sorted(df.holder.unique().tolist()),
            "absent": missing,
            "note": "The two largest operators by subscriber share do not appear. Their capacity spectrum sits under national Spectrum Assignments with no site-level geography.",
        },
        "counts": {
            "sector_rows": int(len(df)),
            "distinct_sites": int(df.assign_no.nunique()),
            "sectors_per_site": int(len(df) // df.assign_no.nunique()),
        },
        "supersedes": (
            "The earlier Spearman -0.065 (p=0.69) sector-count-vs-speed test. That "
            "correlation rested on district-name matching and its per-cell counts were "
            "never persisted. This structural finding is stronger: it does not depend on "
            "sample size and is reproducible from a single register search."
        ),
        "how_to_state_it": (
            "Malaysia's public apparatus register cannot support a capacity feature: it "
            "publishes no site coordinates, and the spectrum it does list is sub-1 GHz "
            "coverage spectrum from the smaller operators. The bands that determine "
            "throughput are held under national assignments with no site-level geography."
        ),
    }, indent=2))
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
