"""
GeoQoE Pulse — Dynamic Mobile QoE & Demand Pressure Intelligence Dashboard.

Run with:  streamlit run app.py

Reads real pipeline outputs from data/outputs/ if present (see
data_loader.PIPELINE_OUTPUT_DIR), otherwise falls back to a seeded synthetic
demo dataset — clearly banner-flagged — so the dashboard runs standalone.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from data_loader import COLUMN_MAP, STUDY_AREA_NAME, load_data

C = COLUMN_MAP  # shorthand

st.set_page_config(page_title="GeoQoE Pulse", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    h1, h2, h3 { color: #0f172a; font-weight: 600 !important; }
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 12px 8px 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #475569; font-weight: 500; }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading GeoQoE Pulse data…")
def _load():
    return load_data()


try:
    df_static, df_panel, metadata, is_real, filled_cols = _load()
except (ImportError, KeyError) as e:
    st.error(str(e))
    st.stop()

df_panel[C["timestamp"]] = pd.to_datetime(df_panel[C["timestamp"]])
df = df_panel.merge(df_static, on=C["cell_id"], how="left")

# Derived confidence % — prefers the real Gi* p-value if the pipeline
# exported one, otherwise maps the confidence-band label to a rough percent.
_BAND_TO_PCT = {"99%": 99.0, "95%": 95.0, "90%": 90.0, "not significant": 50.0}
if "gi_star_p" in df.columns:
    CONF_COL = "qoe_confidence_pct"
    df[CONF_COL] = ((1 - df["gi_star_p"].clip(0, 1)) * 100).round(1)
else:
    CONF_COL = "qoe_confidence_pct"
    df[CONF_COL] = df[C["gi_confidence_band"]].map(_BAND_TO_PCT).fillna(50.0)

# --------------------------------------------------------------------------
# Sidebar — global controls
# --------------------------------------------------------------------------

st.sidebar.title("📡 GeoQoE Pulse")
st.sidebar.caption(f"Study area: {STUDY_AREA_NAME}")

if not is_real:
    st.sidebar.warning(
        "Showing **seeded synthetic demo data** — no pipeline output found "
        "under `data/outputs/`. Run the geoqoe_pulse pipeline and refresh "
        "to see real results.",
        icon="🧪",
    )
elif filled_cols:
    with st.sidebar.expander(f"⚠️ {len(filled_cols)} column(s) auto-filled", expanded=False):
        st.caption(
            "These weren't found in your real parquet files, so a fallback "
            "was used. Check COLUMN_MAP in data_loader.py against "
            "feature_dictionary.xlsx."
        )
        for c in filled_cols:
            st.write(f"• {c}")

timestamps = sorted(df[C["timestamp"]].unique())
selected_ts = st.sidebar.select_slider(
    "As-of time",
    options=timestamps,
    value=timestamps[-1],
    format_func=lambda t: pd.Timestamp(t).strftime("%a %d %b, %H:%M"),
)

profile_options = sorted(df[C["profile"]].dropna().unique().tolist())
selected_profiles = st.sidebar.multiselect(
    "Area profile", profile_options, default=profile_options
)

page = st.sidebar.radio(
    "View",
    [
        "🗺️ Overview",
        "🔥 Hotspot Explorer",
        "📈 Cell Drill-down",
        "🛠️ Interventions",
        "🧪 Data Provenance",
    ],
)

snapshot = df[(df[C["timestamp"]] == selected_ts) & (df[C["profile"]].isin(selected_profiles))]
history = df[df[C["profile"]].isin(selected_profiles)]

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

QOE_COLOR_SCALE = [
    [0, 178, 60, 200], [255, 214, 51, 200], [255, 140, 0, 200], [214, 39, 40, 220],
]


def _qoe_to_rgba(qoe: float):
    if qoe >= 80:
        return QOE_COLOR_SCALE[0]
    if qoe >= 60:
        return QOE_COLOR_SCALE[1]
    if qoe >= 40:
        return QOE_COLOR_SCALE[2]
    return QOE_COLOR_SCALE[3]


def h3_map(data: pd.DataFrame, value_col: str, tooltip_extra: str = ""):
    plot_df = data.copy()
    plot_df["fill_color"] = plot_df[C["qoe_score"]].apply(_qoe_to_rgba)
    layer = pdk.Layer(
        "H3HexagonLayer",
        plot_df,
        get_hexagon=C["cell_id"],
        get_fill_color="fill_color",
        get_line_color=[60, 60, 60],
        line_width_min_pixels=1,
        pickable=True,
        extruded=False,
        opacity=0.75,
    )
    view_state = pdk.ViewState(
        latitude=float(plot_df["lat"].mean()),
        longitude=float(plot_df["lon"].mean()),
        zoom=10, pitch=0,
    )
    tooltip = {
        "html": (
            f"<b>📍 Cell:</b> {{{C['cell_id']}}}<br/>"
            f"<b>📈 QoE score:</b> {{{C['qoe_score']}}} / 100<br/>"
            f"<b>👥 Population:</b> {{{C['population']}}}<br/>"
            f"<b>🏷️ Area profile:</b> {{{C['profile']}}}<br/>"
            f"<b>🤔 Likely cause:</b> "
            f"<span style='color:#fbbf24'>{{{C['likely_cause']}}}</span><br/>"
            f"<b>📊 Confidence:</b> {{{CONF_COL}}}%"
            f"{tooltip_extra}"
        ),
        "style": {
            "backgroundColor": "#0f172a",
            "color": "#f8fafc",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "fontSize": "13px",
        },
    }
    st.pydeck_chart(pdk.Deck(
        layers=[layer], initial_view_state=view_state, tooltip=tooltip,
        map_style="road",
    ))


# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------

if page == "🗺️ Overview":
    st.title("🗺️ Live QoE Intelligence")
    st.caption("*Turning invisible network congestion into visible, actionable decisions.*")

    total_cells = snapshot[C["cell_id"]].nunique()
    if C["is_degraded"] in snapshot.columns:
        # Convert to boolean if needed
        if snapshot[C["is_degraded"]].dtype != 'bool':
            snapshot[C["is_degraded"]] = snapshot[C["is_degraded"]].astype(bool)
        degraded = snapshot[snapshot[C["is_degraded"]]]
    else:
        # Fallback: use qoe_score to determine degradation
        degraded = snapshot[snapshot[C["qoe_score"]] < 60]

    pop_affected = int(degraded[C["population"]].sum()) if not degraded.empty else 0
    pct_degraded = 100 * len(degraded) / max(total_cells, 1)
    avg_qoe = snapshot[C["qoe_score"]].mean()
    hi_conf_hotspots = int((snapshot[C["gi_confidence_band"]] == "99%").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📡 Total cells", f"{total_cells:,}")
    c2.metric("⚠️ Degraded cells", f"{len(degraded):,}",
              delta=f"{pct_degraded:.1f}% of cells", delta_color="inverse")
    c3.metric("👥 Population affected", f"{pop_affected:,}")
    c4.metric("🔥 High-confidence hotspots", hi_conf_hotspots)
    c5.metric("📊 Avg city-wide QoE", f"{avg_qoe:.0f} / 100")

    h3_map(snapshot, C["qoe_score"])
    st.caption("🟢 QoE ≥ 80  🟡 60–79  🟠 40–59  🔴 < 40")

    st.subheader("City-wide QoE trend")
    trend = history.groupby(C["timestamp"])[C["qoe_score"]].mean().reset_index()
    fig = px.line(trend, x=C["timestamp"], y=C["qoe_score"], labels={C["qoe_score"]: "Avg QoE"})
    fig.add_vline(x=selected_ts, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# 2. Hotspot Explorer
# --------------------------------------------------------------------------

elif page == "🔥 Hotspot Explorer":
    st.title("🔥 Hotspot Explorer — statistically validated, not just low scores")
    st.caption(
        "Getis-Ord Gi* / Local Moran's I confidence bands separate real recurring "
        "clusters from noisy one-off low readings."
    )

    band_options = ["99%", "95%", "90%", "not significant"]
    bands = st.multiselect("Confidence band", band_options, default=["99%", "95%"])
    quad = st.multiselect(
        "Cluster type", ["HH", "LL", "NS"], default=["HH"],
        help="HH = high-value cluster (persistent bad QoE surrounded by bad neighbours)",
    )

    filtered = snapshot[
        snapshot[C["gi_confidence_band"]].isin(bands) & snapshot[C["moran_quadrant"]].isin(quad)
    ]
    st.write(f"**{len(filtered)}** cells match the current filter at this timestamp.")
    h3_map(filtered if len(filtered) else snapshot.iloc[0:0], C["qoe_score"])

    st.subheader("Top recurring hotspots (by historical frequency)")
    freq_col = C["hotspot_freq_hist"]
    if freq_col not in history.columns:
        freq = (
            history[history[C["gi_confidence_band"]].isin(["95%", "99%"])]
            .groupby(C["cell_id"]).size().rename("hotspot_hours").reset_index()
        )
    else:
        freq = history.groupby(C["cell_id"])[freq_col].max().reset_index()
        freq.columns = [C["cell_id"], "hotspot_hours"]

    top = (
        freq.merge(df_static, on=C["cell_id"])
        .sort_values("hotspot_hours", ascending=False)
        .head(15)
    )
    st.dataframe(
        top[[C["cell_id"], C["profile"], C["population"], "hotspot_hours"]],
        use_container_width=True, hide_index=True,
    )
    pick = st.selectbox("Investigate a cell in the Drill-down tab", top[C["cell_id"]].tolist())
    if st.button("Open in Cell Drill-down →"):
        st.session_state["selected_cell"] = pick
        st.info("Switch to the **📈 Cell Drill-down** tab in the sidebar.")

# --------------------------------------------------------------------------
# 3. Cell Drill-down
# --------------------------------------------------------------------------

elif page == "📈 Cell Drill-down":
    st.title("📈 Cell Drill-down — why is this cell failing?")

    cell_list = sorted(df_static[C["cell_id"]].unique().tolist())
    default_cell = st.session_state.get("selected_cell", cell_list[0])
    cell_id = st.selectbox(
        "Select a cell (H3 index)", cell_list,
        index=cell_list.index(default_cell) if default_cell in cell_list else 0,
    )

    info = df_static[df_static[C["cell_id"]] == cell_id].iloc[0]
    cell_hist = df[df[C["cell_id"]] == cell_id].sort_values(C["timestamp"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Area profile", info[C["profile"]])
    c2.metric("Population", f"{info[C['population']]:,.0f}")
    c3.metric("Cell towers nearby", int(info[C["num_towers"]]))
    c4.metric("Built-up ratio", f"{info[C['built_up_ratio']]:.0%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cell_hist[C["timestamp"]], y=cell_hist[C["qoe_score"]],
        name="QoE score", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=cell_hist[C["timestamp"]], y=cell_hist[C["demand_pressure"]],
        name="Demand pressure", yaxis="y2", line=dict(dash="dot"),
    ))
    fig.update_layout(
        yaxis=dict(title="QoE score (0-100)", range=[0, 100]),
        yaxis2=dict(title="Demand pressure", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h"), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Root cause distribution (degraded hours in window)")
    # --- FIX: Safe boolean filtering ---
    if C["is_degraded"] in cell_hist.columns:
        if cell_hist[C["is_degraded"]].dtype != 'bool':
            cell_hist[C["is_degraded"]] = cell_hist[C["is_degraded"]].astype(bool)
        degraded_hist = cell_hist[cell_hist[C["is_degraded"]]]
    else:
        degraded_hist = cell_hist[cell_hist[C["qoe_score"]] < 60]
    if len(degraded_hist):
        cause_counts = degraded_hist[C["likely_cause"]].value_counts(normalize=True) * 100
        st.bar_chart(cause_counts)
    else:
        st.caption("No degraded hours in the selected window for this cell.")

    st.subheader("🧐 Current status")
    latest = cell_hist.iloc[-1] if not cell_hist.empty else None

    if latest is not None and bool(latest[C["is_degraded"]]):
        cause = str(latest.get(C["likely_cause"], "unknown"))
        confidence = float(latest.get(CONF_COL, 50.0))

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Degradation confidence", f"{confidence:.0f}%")
            st.progress(min(max(confidence / 100, 0.0), 1.0))
        with col_b:
            recs = {
                "congestion": "Consider traffic steering or temporary small-cell "
                              "capacity during the identified peak hours.",
                "indoor_attenuation": "Building survey via MCMC Nexus; consider "
                                       "DAS/small-cell or Wi-Fi offload.",
                "coverage_gap": "Schedule a physical site visit — tower density is low here.",
                "backhaul": "Check backhaul link utilisation independent of radio congestion.",
            }
            st.info(
                f"**Dominant cause:** {cause.replace('_', ' ').title()}\n\n"
                f"**Suggested next step:** {recs.get(cause, 'Investigate further — no dominant cause pattern.')}\n\n"
                f"*Note: pattern-matched from recent readings, not live network telemetry.*"
            )
    else:
        st.success("✅ This cell is performing within acceptable thresholds as of the selected time.")

# --------------------------------------------------------------------------
# 4. Interventions
# --------------------------------------------------------------------------

elif page == "🛠️ Interventions":
    st.title("🛠️ Prioritized Interventions")
    st.caption(
        "Ranked by affected population, severity, and hotspot confidence. "
        "Illustrative prioritisation, not a trained forecasting model."
    )

    # --- FIX: Safe boolean filtering ---
    if C["is_degraded"] in snapshot.columns:
        if snapshot[C["is_degraded"]].dtype != 'bool':
            snapshot[C["is_degraded"]] = snapshot[C["is_degraded"]].astype(bool)
        is_degraded_condition = snapshot[C["is_degraded"]]
    else:
        is_degraded_condition = snapshot[C["qoe_score"]] < 60

    candidates = snapshot[
        is_degraded_condition & snapshot[C["gi_confidence_band"]].isin(["95%", "99%"])
].copy()

    if not candidates.empty:
        action_map = {
            "congestion": "Traffic steering / temporary capacity",
            "indoor_attenuation": "Indoor survey → DAS / small cell / Wi-Fi offload",
            "coverage_gap": "Site visit — candidate for new tower or repeater",
            "backhaul": "Backhaul capacity upgrade",
        }
        candidates["recommended_action"] = (
            candidates[C["likely_cause"]].map(action_map).fillna("Investigate further")
        )

        pop_max = max(candidates[C["population"]].max(), 1)
        candidates["priority_score"] = (
            candidates[C["population"]] / pop_max * 0.5
            + (1 - candidates[C["qoe_score"]] / 100) * 0.3
            + (candidates[C["gi_confidence_band"]] == "99%").astype(int) * 0.2
        )
        candidates = candidates.sort_values("priority_score", ascending=False)
        candidates["Urgency"] = candidates["priority_score"].apply(
            lambda x: "🔴 High" if x > 0.7 else ("🟡 Medium" if x > 0.4 else "🟢 Low")
        )

        top = candidates.head(15)
        st.dataframe(
            top[[C["cell_id"], C["profile"], C["population"], C["qoe_score"],
                 "Urgency", C["likely_cause"], "recommended_action"]],
            column_config={
                C["cell_id"]: "Cell ID",
                C["profile"]: "Area type",
                C["population"]: st.column_config.NumberColumn("Population", format="%d"),
                C["qoe_score"]: st.column_config.NumberColumn("QoE", format="%.0f"),
                "Urgency": st.column_config.TextColumn("Priority"),
                C["likely_cause"]: "Root cause",
                "recommended_action": "Recommended action",
            },
            use_container_width=True, hide_index=True,
        )

        st.subheader("📍 Hotspot location map")
        h3_map(top, C["qoe_score"])

        csv = top.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export top 15 recommendations (CSV)", data=csv,
            file_name="geoqoe_interventions.csv", mime="text/csv",
        )
    else:
        st.success("✅ No high-confidence degraded hotspots detected at this timestamp.")

# --------------------------------------------------------------------------
# 5. Data Provenance
# --------------------------------------------------------------------------

else:
    st.title("🧪 Data Provenance & Quality")
    st.caption(
        "Every feature that falls back to synthetic data is flagged, never silently faked."
    )

    st.json(metadata, expanded=False)

    synth_cols = [c for c in df.columns if c.endswith("_is_synthetic")]
    if synth_cols:
        frac = df[synth_cols].mean().sort_values(ascending=False) * 100
        frac.index = [c.replace("_is_synthetic", "") for c in frac.index]
        st.subheader("Share of synthetic (vs. real) values by feature")
        st.bar_chart(frac)
    else:
        st.info("No `*_is_synthetic` flag columns found in the loaded dataset.")

    st.subheader("Dataset shape")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cells", df_static[C["cell_id"]].nunique())
    c2.metric("Timestamps", df[C["timestamp"]].nunique())
    c3.metric("Panel rows", len(df))

    with st.expander("📄 Full technical report (for engineers & auditors)"):
        st.json({
            "study_area": STUDY_AREA_NAME,
            "total_cells": int(df[C["cell_id"]].nunique()),
            "time_periods": int(df[C["timestamp"]].nunique()),
            "avg_hotspot_confidence_pct": float(df[CONF_COL].mean()),
            "synthetic_data_used": not is_real,
            "auto_filled_columns": filled_cols,
        })