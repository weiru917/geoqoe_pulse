/* ==========================================================================
   GeoQoE Pulse dashboard.js
   All data below is loaded from the REAL geoqoe_pulse pipeline output
   (converted once by build_dashboard_data.py). Nothing here is fabricated;
   see the Data Quality panel / summary_stats.json "disclaimer" field for
   what is real vs. simulated in the source pipeline itself.
   ========================================================================== */

const DATA_URLS = {
  grid: "./data/h3_grid.geojson",
  panel: "./data/panel.json",
  cityTimeseries: "./data/city_timeseries.json",
  summary: "./data/summary_stats.json",
  onset: "./data/onset_model.json",
  modelCard: "./data/model_card.json",
};

const state = {
  grid: null,
  panel: null,
  cityTS: null,
  summary: null,
  onset: null,      // trained AI model output (static estimate + onset risk) — single fixed
                     // snapshot at the model's latest scored hour, NOT part of the 168-hour
                     // panel. Indexed by the SAME cell index `c` as panel.fields (see
                     // build_onset_data.py — it reindexes to panel.h3_ids order).
  modelCard: null,   // frozen model metrics/limitations, for the AI Model Info modal
  onsetSnapshotT: null,      // panel.json hour-index matching the AI snapshot, or -1/null if none
  cityStats: null,           // city-wide mean/std of rtt/jitter/loss/demand, for per-cell AI explanations
  aiSummary: null,           // cached { degradedNowCount, topRiskCell } from renderAiPanel()
  lastCandidates: [],        // cached from renderInterventions(), reused by renderExecOverview()
  viewMode: "full",          // "full" | "overview" — see setViewMode()
  h3IndexOf: new Map(),     // h3_index -> cell index (0..C-1)
  gridPropsByH3: new Map(), // h3_index -> geojson feature properties
  currentT: 0,              // current timestamp index (map/KPIs)
  rangeStart: 0,            // trend-range selection (charts/range summary)
  rangeEnd: 0,
  granularity: "hourly",    // "hourly" | "daily"
  confidenceThreshold: 0,   // 0-1, filters ranked lists & interventions only
  colorBy: "qoe",
  basemap: "labels",
  compareMode: false,
  compareCells: [],         // array of h3_index, max COMPARE_MAX
  tracking: {},             // h3_index -> "planned" | "done", persisted to localStorage
  profile: "balanced",      // QoE weighting profile — see PROFILE_WEIGHTS
  whatIfEnabled: false,
  whatIfPct: 20,            // -50..+100, % capacity change
  weatherEnabled: false,
  weatherLayer: null,
  darkMode: false,
  map: null,
  geoLayer: null,
  towerLayer: null,
  tileLayers: {},
  playing: false,
  playTimer: null,
};
const COMPARE_MAX = 5;
const TRACKING_KEY = "geoqoe_pulse_tracking_v1";
const DARKMODE_KEY = "geoqoe_pulse_darkmode_v1";

// Application-specific QoE weighting profiles. "balanced" reproduces the
// pipeline's real target_qoe_index formula exactly (weights taken from
// summary.qoe_formula, sourced from config.QOE_INDEX_WEIGHTS). The other
// three are OUR OWN authored reweighting of the same composite-index
// methodology for different use cases — not a validated per-application
// standard (the pipeline's own docstring is explicit that even the
// balanced formula isn't a validated ITU-T G.107 E-model; these are
// further from that, not closer).
//
// dl+ul weights are kept summing to 1.0 in every profile, mirroring a
// constraint qoe_target.py enforces (with an assertion) on its own real
// weights: "download+upload MUST sum to 1.0 or the index's documented
// 0-100 range becomes unreachable" — a bug the pipeline authors already
// hit and fixed once; the first version of these profile weights here
// made the identical mistake (dl+ul summing to well under 1.0), silently
// capping the composite at ~30 regardless of how good the connection
// was, caught during development by cross-checking against real cells.
//
// jitter_weight has no pipeline equivalent (the real formula doesn't use
// jitter at all) — added here using the same free-allowance + exponential
// -penalty shape as the real rtt term (see JITTER_PENALTY_FREE_MS/SCALE_MS
// below), for profiles where jitter plausibly matters (gaming, video).
const PROFILE_WEIGHTS = {
  balanced:  { dl: 0.6, ul: 0.4, rtt: 0.4, loss: 0.4, jitter: 0.0, label: "Balanced — reproduces the pipeline's real target_qoe_index exactly" },
  video:     { dl: 0.3, ul: 0.7, rtt: 0.5, loss: 0.4, jitter: 0.2, label: "Video calling — weighted toward upload + latency (authored weighting, not a validated standard)" },
  gaming:    { dl: 0.5, ul: 0.5, rtt: 0.6, loss: 0.3, jitter: 0.6, label: "Gaming — weighted toward latency + jitter (authored weighting, not a validated standard)" },
  streaming: { dl: 0.9, ul: 0.1, rtt: 0.15, loss: 0.3, jitter: 0.02, label: "Streaming — weighted toward download throughput (authored weighting, not a validated standard)" },
};
const JITTER_PENALTY_FREE_MS = 5.0;   // our own term, chosen to mirror rtt's 20ms free allowance
const JITTER_PENALTY_SCALE_MS = 25.0;

// Illustrative-only planning assumptions — this dataset has no real per-cell
// operator attribution or verified deployment cost data, so these are
// editable placeholders, not measured figures. Edit to match your org.
const ILLUSTRATIVE_COST_RANGES = {
  congestion: "~RM 15k–40k (temporary capacity / traffic steering)",
  coverage: "~RM 150k–400k (new small cell/macro site)",
  indoor: "~RM 20k–80k (DAS / Wi-Fi offload survey + install)",
  backhaul: "~RM 30k–100k (backhaul capacity upgrade)",
  insufficient: "~RM 2k–5k (targeted field survey)",
};
// No real operator-per-cell mapping exists in this dataset. Fill this in if
// you have one (e.g. from a licence/coverage-obligation dataset); otherwise
// the UI falls back to a generic escalation line.
const OPERATOR_CONTACTS = {};

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

async function boot() {
  const [grid, panel, cityTS, summary, onset, modelCard] = await Promise.all([
    fetch(DATA_URLS.grid).then(r => r.json()),
    fetch(DATA_URLS.panel).then(r => r.json()),
    fetch(DATA_URLS.cityTimeseries).then(r => r.json()),
    fetch(DATA_URLS.summary).then(r => r.json()),
    fetch(DATA_URLS.onset).then(r => r.ok ? r.json() : null).catch(() => null),
    fetch(DATA_URLS.modelCard).then(r => r.ok ? r.json() : null).catch(() => null),
  ]);

  const REQUIRED_FIELDS = [
    "qoe", "degradation_class", "demand_pressure", "gi_star_z", "hotspot_code",
    "moran_code", "confidence", "dl_mbps", "ul_mbps", "rtt_ms", "jitter_ms", "loss_pct",
    "next_hour_degraded_historical", "priority_base_score", "ookla_anchored",
  ];
  const missing = REQUIRED_FIELDS.filter(f => !(f in (panel.fields || {})));
  if (missing.length) {
    throw new Error(
      `panel.json is missing required field(s): ${missing.join(", ")}. ` +
      `Re-run build_dashboard_data.py, or check that column names in the ` +
      `"panel_cols" list there still match your feature_dictionary.xlsx.`
    );
  }
  if (!grid.features?.length) throw new Error("h3_grid.geojson has no features — check data/processed/grid_geometry.geojson exists.");
  if (!summary.qoe_formula || !summary.demand_elasticity) {
    throw new Error(
      "summary_stats.json is missing qoe_formula/demand_elasticity constants " +
      "(needed for the What-if simulator and application profiles) — re-run build_dashboard_data.py."
    );
  }

  // The trained AI model package is optional at boot: if data/onset_model.json
  // wasn't generated (see build_onset_data.py), the dashboard degrades gracefully
  // — the AI layer/panel just hides itself rather than breaking the whole page.
  const ONSET_REQUIRED_FIELDS = [
    "static_data_status", "display_dl_mbps", "ai_ensemble_estimate_mbps",
    "onset_priority_tier", "next_hour_onset_risk_score", "onset_operational_status",
  ];
  if (onset) {
    const missingOnset = ONSET_REQUIRED_FIELDS.filter(f => !(f in (onset.fields || {})));
    const cellsMatch = onset.num_cells === panel.num_cells &&
      onset.h3_ids?.length === panel.h3_ids.length &&
      onset.h3_ids.every((h, i) => h === panel.h3_ids[i]);
    if (missingOnset.length || !cellsMatch) {
      console.error(
        "data/onset_model.json is present but doesn't line up with panel.json " +
        `(missing fields: ${missingOnset.join(", ") || "none"}; cell order match: ${cellsMatch}) — ` +
        "re-run build_onset_data.py against the current panel.json. Disabling the AI layer for this session."
      );
      state.onset = null;
    } else {
      state.onset = onset;
      // Which panel.json hour, if any, the frozen AI snapshot corresponds to —
      // used to (a) warn honestly when the slider is on a different hour instead
      // of pretending the AI forecast moves with it, and (b) pull the RIGHT
      // hour's raw metrics for the per-cell explanation block.
      const snapIso = onset.snapshot_timestamp.replace(" ", "T") + (onset.snapshot_timestamp.endsWith("Z") ? "" : "Z");
      state.onsetSnapshotT = panel.timestamps_iso.indexOf(snapIso);
      if (state.onsetSnapshotT === -1) {
        console.warn(`AI snapshot timestamp ${onset.snapshot_timestamp} isn't one of panel.json's hours — time-sync banner will stay in "no matching hour" mode.`);
      }
    }
  }
  state.modelCard = modelCard || null;

  state.grid = grid;
  state.panel = panel;
  state.cityTS = cityTS;
  state.summary = summary;
  state.currentT = panel.num_timestamps - 1; // default to latest hour
  state.rangeStart = 0;
  state.rangeEnd = panel.num_timestamps - 1;
  loadTracking();

  panel.h3_ids.forEach((h3, i) => state.h3IndexOf.set(h3, i));
  grid.features.forEach(f => state.gridPropsByH3.set(f.properties.h3_index, f.properties));
  state.cityStats = computeCityStats(state.onsetSnapshotT ?? panel.num_timestamps - 1);

  document.getElementById("regionBadge").textContent =
    summary.study_area.split(":")[0] || "Klang Valley";

  const slider = document.getElementById("timeSlider");
  slider.min = 0;
  slider.max = panel.num_timestamps - 1;
  slider.value = state.currentT;
  slider.addEventListener("input", e => {
    state.currentT = parseInt(e.target.value, 10);
    renderAll();
  });

  document.getElementById("playBtn").addEventListener("click", togglePlay);
  document.getElementById("colorBySelect").addEventListener("change", e => {
    state.colorBy = e.target.value;
    renderMapColors();
    renderLegend();
  });
  document.getElementById("toggleTowers").addEventListener("change", e => {
    state.towerLayer.setStyle({ opacity: e.target.checked ? 1 : 0, fillOpacity: e.target.checked ? 0.6 : 0 });
  });
  document.getElementById("basemapSelect").addEventListener("change", e => {
    setBasemap(e.target.value);
  });
  document.getElementById("compareToggleBtn").addEventListener("click", toggleCompareMode);
  document.getElementById("confidenceFilter").addEventListener("input", e => {
    state.confidenceThreshold = parseInt(e.target.value, 10) / 100;
    document.getElementById("confidenceFilterLabel").textContent = e.target.value + "%";
    renderHotspotLists();
    renderInterventions();
  });

  const rs = document.getElementById("rangeStart"), re = document.getElementById("rangeEnd");
  rs.max = re.max = panel.num_timestamps - 1;
  rs.value = state.rangeStart; re.value = state.rangeEnd;
  const onRangeChange = () => {
    let a = parseInt(rs.value, 10), b = parseInt(re.value, 10);
    if (a > b) [a, b] = [b, a];
    state.rangeStart = a; state.rangeEnd = b;
    renderRangeSummary();
    updateChartRangeShading();
  };
  rs.addEventListener("input", onRangeChange);
  re.addEventListener("input", onRangeChange);

  document.querySelectorAll(".granularity-toggle button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".granularity-toggle button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.granularity = btn.dataset.granularity;
      buildCharts();
      renderRangeSummary();
    });
  });

  document.getElementById("hideDoneToggle").addEventListener("change", renderInterventions);
  document.getElementById("resetTrackingBtn").addEventListener("click", () => {
    state.tracking = {};
    saveTracking();
    renderInterventions();
  });

  document.getElementById("profileSelect").addEventListener("change", e => {
    state.profile = e.target.value;
    document.getElementById("profileNote").textContent = PROFILE_WEIGHTS[state.profile].label;
    renderMapColors();
    renderKpis();
    renderExecOverview();
  });
  document.getElementById("profileNote").textContent = PROFILE_WEIGHTS[state.profile].label;

  const whatIfCheckbox = document.getElementById("whatIfEnabled");
  const whatIfSlider = document.getElementById("whatIfSlider");
  const updateWhatIfLabel = () => {
    const v = parseInt(whatIfSlider.value, 10);
    document.getElementById("whatIfLabel").textContent = `${v >= 0 ? "+" : ""}${v}% capacity`;
  };
  whatIfCheckbox.addEventListener("change", e => {
    state.whatIfEnabled = e.target.checked;
    whatIfSlider.disabled = !state.whatIfEnabled;
    document.body.classList.toggle("whatif-active", state.whatIfEnabled);
    renderMapColors();
    renderKpis();
  });
  whatIfSlider.addEventListener("input", e => {
    state.whatIfPct = parseInt(e.target.value, 10);
    updateWhatIfLabel();
    renderMapColors();
    renderKpis();
  });
  updateWhatIfLabel();

  document.getElementById("toggleWeather").addEventListener("change", e => {
    state.weatherEnabled = e.target.checked;
    toggleWeatherLayer();
  });

  document.getElementById("darkModeToggle").addEventListener("click", toggleDarkMode);
  initDarkMode();

  document.getElementById("exportReportBtn").addEventListener("click", exportExecutiveSummary);

  initAiPanel();
  initPlaceSearch();

  buildMap();
  buildCharts();
  renderAll();
  renderRangeSummary();
}

// ---------------------------------------------------------------------
// Trained AI model access — the onset dataset is a single fixed snapshot
// (see state.onset.snapshot_timestamp), aligned to panel.json's cell
// order, so `onsetValue(field, c)` mirrors `cellValue(field, t, c)` but
// with no time argument.
// ---------------------------------------------------------------------

function onsetValue(field, c) {
  return state.onset?.fields?.[field]?.[c] ?? null;
}

function cellValue(field, t, c) {
  return state.panel.fields[field][t * state.panel.num_cells + c];
}

// ---------------------------------------------------------------------
// Per-cell AI explainability — a real, computed abnormality/contribution
// analysis (city-wide mean/std of the same real hourly fields the onset
// model is trained on), NOT a fabricated SHAP value. Answers "why is this
// cell flagged?" using the pipeline's own metrics at the AI snapshot hour.
// ---------------------------------------------------------------------

function computeCityStats(t) {
  const C = state.panel.num_cells;
  const fields = ["rtt_ms", "jitter_ms", "loss_pct", "demand_pressure"];
  const stats = {};
  fields.forEach(f => {
    let sum = 0, n = 0;
    for (let c = 0; c < C; c++) {
      const v = cellValue(f, t, c);
      if (v != null) { sum += v; n++; }
    }
    const mean = n ? sum / n : 0;
    let sq = 0;
    for (let c = 0; c < C; c++) {
      const v = cellValue(f, t, c);
      if (v != null) sq += (v - mean) ** 2;
    }
    stats[f] = { mean, std: n ? Math.sqrt(sq / n) || 1 : 1 };
  });
  return stats;
}

const EXPLAIN_SIGNAL_META = {
  rtt_ms: { label: "RTT", unit: "ms", direction: "high" },
  jitter_ms: { label: "Jitter", unit: "ms", direction: "high" },
  loss_pct: { label: "Packet loss", unit: "%", direction: "high" },
  demand_pressure: { label: "Demand pressure", unit: "", direction: "high" },
};

function severityFromZ(z) {
  if (z >= 2) return { tag: "🔴", word: "very high" };
  if (z >= 1) return { tag: "🟠", word: "elevated" };
  if (z <= -1) return { tag: "🟢", word: "below normal" };
  return { tag: "🟢", word: "normal" };
}

// Explains a cell using the SAME hour the AI snapshot was scored at
// (state.onsetSnapshotT), regardless of where the time slider currently
// sits — the explanation must match what the model actually saw.
function explainCell(c) {
  const t = state.onsetSnapshotT;
  if (t == null || t < 0 || !state.cityStats) return null;
  const h3 = state.panel.h3_ids[c];

  const signals = Object.keys(EXPLAIN_SIGNAL_META).map(f => {
    const v = cellValue(f, t, c);
    const { mean, std } = state.cityStats[f];
    const z = v != null ? (v - mean) / std : 0;
    const sev = severityFromZ(z);
    return { field: f, ...EXPLAIN_SIGNAL_META[f], value: v, z, ...sev };
  }).filter(s => s.value != null)
    .sort((a, b) => b.z - a.z);

  const qoeNow = cellValue("qoe", t, c);
  const qoePrev = t > 0 ? cellValue("qoe", t - 1, c) : null;
  const qoeDeltaPct = (qoeNow != null && qoePrev != null && qoePrev > 0)
    ? 100 * (qoeNow - qoePrev) / qoePrev : null;

  const pct = causeScoresForCell(t, c, h3);
  const cause = dominantCause(pct);
  const top2 = signals.slice(0, 2).filter(s => s.z >= 1);
  const triggerParts = top2.map(s => s.word === "very high" ? `very high ${s.label.toLowerCase()}` : `elevated ${s.label.toLowerCase()}`);
  const likelyTrigger = triggerParts.length
    ? `${CAUSE_LABELS[cause]} — driven by ${triggerParts.join(" combined with ")}.`
    : `${CAUSE_LABELS[cause]} (no single signal is sharply abnormal — see full metrics).`;

  return { signals, qoeDeltaPct, likelyTrigger };
}

// ---------------------------------------------------------------------
// Real composite QoE formula — transcribed from qoe_target.py, verified
// to reproduce target_qoe_index to within JSON rounding error (<0.25pt
// over a 2000-sample check during development). See summary.qoe_formula.
// ---------------------------------------------------------------------

function computeQoeComposite(dl, ul, rtt, jitter, loss, weights) {
  const f = state.summary.qoe_formula;
  const sDl = Math.min(100, Math.log1p(dl) / Math.log1p(f.dl_log_saturation_mbps) * 100);
  const sUl = Math.min(100, Math.log1p(ul) / Math.log1p(f.ul_log_saturation_mbps) * 100);
  const penRtt = 100 * (1 - Math.exp(-Math.max(0, rtt - f.rtt_penalty_free_ms) / f.rtt_penalty_scale_ms));
  const penLoss = 100 * (1 - Math.exp(-Math.max(0, loss) / f.loss_penalty_scale_pct));
  const penJitter = 100 * (1 - Math.exp(-Math.max(0, jitter - JITTER_PENALTY_FREE_MS) / JITTER_PENALTY_SCALE_MS)); // our own term; real formula has none
  const raw = weights.dl * sDl + weights.ul * sUl - weights.rtt * penRtt
            - weights.loss * penLoss - weights.jitter * penJitter;
  return Math.max(0, Math.min(100, raw));
}

// ---------------------------------------------------------------------
// What-if capacity simulator — uses the REAL demand-elasticity
// coefficients from data_downloader.py's build_dynamic_panel(), applied
// as a documented approximation (see whatif-disclaimer in the UI): we
// don't have the per-cell load_factor or city-wide demand_mult that the
// original generator used, so demand_pressure_index (0-100) stands in as
// a monotonic proxy for "how far above baseline" a cell currently is.
// ---------------------------------------------------------------------

function applyWhatIf(t, c) {
  const dl0 = cellValue("dl_mbps", t, c), rtt0 = cellValue("rtt_ms", t, c);
  const ul0 = cellValue("ul_mbps", t, c), jitter0 = cellValue("jitter_ms", t, c);
  const loss0 = cellValue("loss_pct", t, c), demand0 = cellValue("demand_pressure", t, c);
  if (!state.whatIfEnabled || dl0 == null || rtt0 == null) {
    return { dl: dl0, ul: ul0, rtt: rtt0, jitter: jitter0, loss: loss0, demand: demand0 };
  }
  const anchored = cellValue("ookla_anchored", t, c);
  const el = state.summary.demand_elasticity;
  const dlSens = anchored ? el.anchored_dl_sensitivity : el.unanchored_dl_sensitivity;
  const rttSens = anchored ? el.anchored_rtt_sensitivity : el.unanchored_rtt_sensitivity;

  const relief = Math.max(-1, Math.min(1, state.whatIfPct / 100));
  const demandProxy = Math.max(0, (demand0 ?? 0) / 100);
  const newDemandProxy = Math.max(0, demandProxy * (1 - relief));

  const dl = dl0 * (1 + dlSens * demandProxy) / (1 + dlSens * newDemandProxy);
  const rtt = rtt0 * (1 + rttSens * newDemandProxy) / (1 + rttSens * demandProxy);
  const jitter = jitter0 != null ? jitter0 * (rtt / rtt0) : jitter0;
  const loss = loss0 != null ? Math.max(0, loss0 * (demandProxy > 0.001 ? newDemandProxy / demandProxy : 1)) : loss0;
  const demand = Math.max(0, Math.min(100, (demand0 ?? 0) * (1 - relief)));

  return { dl, ul: ul0, rtt, jitter, loss, demand };
}

function effectiveMetrics(t, c) {
  const m = applyWhatIf(t, c);
  if (m.dl == null || m.ul == null || m.rtt == null) {
    return { ...m, qoe: cellValue("qoe", t, c) };
  }
  const weights = PROFILE_WEIGHTS[state.profile];
  const qoe = computeQoeComposite(m.dl, m.ul, m.rtt, m.jitter ?? 0, m.loss ?? 0, weights);
  return { ...m, qoe: Math.round(qoe * 10) / 10 };
}

// ---------------------------------------------------------------------
// Intervention tracking persistence (per-browser, local machine only)
// ---------------------------------------------------------------------

function loadTracking() {
  try {
    state.tracking = JSON.parse(localStorage.getItem(TRACKING_KEY) || "{}");
  } catch { state.tracking = {}; }
}
function saveTracking() {
  try { localStorage.setItem(TRACKING_KEY, JSON.stringify(state.tracking)); } catch {}
}

// ---------------------------------------------------------------------
// Dark mode (per-browser preference, CSS variables only — no re-render
// of chart/map data needed, just class toggle + Plotly template swap)
// ---------------------------------------------------------------------

function initDarkMode() {
  let saved = null;
  try { saved = localStorage.getItem(DARKMODE_KEY); } catch {}
  state.darkMode = saved === "1";
  applyDarkMode();
}
function toggleDarkMode() {
  state.darkMode = !state.darkMode;
  try { localStorage.setItem(DARKMODE_KEY, state.darkMode ? "1" : "0"); } catch {}
  applyDarkMode();
}
function applyDarkMode() {
  document.body.classList.toggle("dark-mode", state.darkMode);
  document.getElementById("darkModeToggle").textContent = state.darkMode ? "☀️" : "🌙";
}

// ---------------------------------------------------------------------
// Color scales
// ---------------------------------------------------------------------

function lerpColor(a, b, t) {
  const ah = a.match(/\w\w/g).map(x => parseInt(x, 16));
  const bh = b.match(/\w\w/g).map(x => parseInt(x, 16));
  const rgb = ah.map((v, i) => Math.round(v + (bh[i] - v) * t));
  return `#${rgb.map(v => v.toString(16).padStart(2, "0")).join("")}`;
}

const QOE_STOPS = [
  [0, "#e53e3e"], [25, "#ed8936"], [50, "#ecc94b"], [75, "#48bb78"], [100, "#2f855a"],
];
function qoeColor(v) {
  if (v == null) return "#cbd5e1";
  for (let i = 0; i < QOE_STOPS.length - 1; i++) {
    const [s0, c0] = QOE_STOPS[i], [s1, c1] = QOE_STOPS[i + 1];
    if (v >= s0 && v <= s1) return lerpColor(c0, c1, (v - s0) / (s1 - s0));
  }
  return v < 0 ? QOE_STOPS[0][1] : QOE_STOPS[QOE_STOPS.length - 1][1];
}
function demandColor(v) {
  if (v == null) return "#cbd5e1";
  return lerpColor("#2b6cb0", "#e53e3e", Math.max(0, Math.min(1, v / 100)));
}
const HOTSPOT_COLORS = { 0: "#2f855a", 1: "#38a169", 2: "#68d391", 3: "#a0aec0", 4: "#f6ad55", 5: "#ed8936", 6: "#e53e3e" };
const MORAN_COLORS = { 0: "#a0aec0", 1: "#38a169", 2: "#d69e2e", 3: "#ed8936", 4: "#e53e3e" };
function riskColor(v) {
  if (v == null) return "#cbd5e1";
  return v === 1 ? "#e53e3e" : "#38a169";
}

// Trained-AI-model onset-risk layer (see ai_model/ package). Ordered worst
// to best, matching the priority tiers the model card / README recommend
// surfacing first: currently-degraded, then the two ranked-risk bands,
// then watch bands, then out-of-scope cells.
const ONSET_TIER_COLORS = {
  current_degradation: "#e53e3e",
  highest_priority: "#c05621",
  high_priority: "#ed8936",
  elevated_watch: "#ecc94b",
  routine_watch: "#9ae6b4",
  not_scored: "#cbd5e1",
};
const ONSET_TIER_LABELS = {
  current_degradation: "Current degradation",
  highest_priority: "Highest next-hour priority",
  high_priority: "High next-hour priority",
  elevated_watch: "Elevated watch",
  routine_watch: "Routine watch",
  not_scored: "Not scored",
};
function onsetTierColor(c) {
  if (!state.onset) return "#cbd5e1";
  return ONSET_TIER_COLORS[onsetValue("onset_priority_tier", c)] || "#cbd5e1";
}

// Static AI spatial-baseline layer: real Ookla speed where observed, the
// trained ensemble's estimate otherwise (display_dl_mbps already picks the
// right one — see ai_model README section 5). Log-scaled, same shape as
// the QoE dl term, over a wider Mbps range since this is raw throughput.
function staticDlColor(v) {
  if (v == null) return "#cbd5e1";
  const pct = Math.min(100, Math.log1p(Math.max(0, v)) / Math.log1p(300) * 100);
  return qoeColor(pct);
}

// ---------------------------------------------------------------------
// AI vs. traditional (Gi*) hotspot detection comparison. Both methods are
// evaluated at the SAME hour — the AI snapshot hour, since that's the only
// hour the onset model has ever scored — so this is an apples-to-apples
// comparison, not a mix of two different time points.
// ---------------------------------------------------------------------
const TRADITIONAL_HOTSPOT_CODES = [4, 5, 6]; // hot_90 / hot_95 / hot_99 (see panel.hotspot_code_labels)
const AI_FLAGGED_TIERS = ["current_degradation", "highest_priority", "high_priority", "elevated_watch"];
const COMPARISON_COLORS = { both: "#e53e3e", ai_only: "#ed8936", traditional_only: "#ecc94b", neither: "#cbd5e1" };
const COMPARISON_LABELS = { both: "AI + traditional hotspot", ai_only: "AI only", traditional_only: "Traditional hotspot only", neither: "Neither" };

function isTraditionalHotspot(c) {
  if (state.onsetSnapshotT == null || state.onsetSnapshotT < 0) return false;
  return TRADITIONAL_HOTSPOT_CODES.includes(cellValue("hotspot_code", state.onsetSnapshotT, c));
}
function isAiFlagged(c) {
  if (!state.onset) return false;
  return AI_FLAGGED_TIERS.includes(onsetValue("onset_priority_tier", c));
}
function detectionCategory(c) {
  const ai = isAiFlagged(c), trad = isTraditionalHotspot(c);
  if (ai && trad) return "both";
  if (ai) return "ai_only";
  if (trad) return "traditional_only";
  return "neither";
}
function comparisonColor(c) {
  return COMPARISON_COLORS[detectionCategory(c)];
}

function colorForMode(mode, t, c) {
  switch (mode) {
    case "qoe": return qoeColor(effectiveMetrics(t, c).qoe);
    case "demand": return demandColor(effectiveMetrics(t, c).demand);
    case "hotspot": return HOTSPOT_COLORS[cellValue("hotspot_code", t, c)] || "#cbd5e1";
    case "moran": return MORAN_COLORS[cellValue("moran_code", t, c)] || "#cbd5e1";
    case "risk": return riskColor(cellValue("next_hour_degraded_historical", t, c));
    case "onset": return onsetTierColor(c);
    case "static_ai": return staticDlColor(onsetValue("display_dl_mbps", c));
    case "comparison": return comparisonColor(c);
    default: return "#cbd5e1";
  }
}

// ---------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------

function buildMap() {
  const map = L.map("map", { zoomControl: true }).setView([3.05, 101.65], 11);

  state.tileLayers.labels = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors", maxZoom: 18,
  });
  state.tileLayers.light = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO", maxZoom: 19,
  });
  state.tileLayers[state.basemap].addTo(map);

  const geoLayer = L.geoJSON(state.grid, {
    style: () => ({ color: "#334155", weight: 0.5, fillOpacity: 0.75, fillColor: "#cbd5e1" }),
    onEachFeature: (feature, layer) => {
      const h3 = feature.properties.h3_index;
      layer.bindPopup(() => popupHtml(h3));
      layer.bindTooltip(() => hoverHtml(h3), { sticky: true, direction: "top" });
      layer.on("click", () => {
        if (state.compareMode) toggleCompareCell(h3);
      });
    },
  }).addTo(map);

  const towerLayer = L.layerGroup(
    state.grid.features
      .filter(f => f.properties.tower_count > 0)
      .map(f => L.circleMarker([f.properties.centroid_lat, f.properties.centroid_lon], {
        radius: Math.min(2 + Math.sqrt(f.properties.tower_count), 12),
        color: "#1a365d", weight: 1, fillColor: "#2b6cb0", fillOpacity: 0.6, opacity: 0,
      }).bindTooltip(`${f.properties.tower_count} tower site(s)`))
  ).addTo(map);

  state.map = map;
  state.geoLayer = geoLayer;
  state.towerLayer = towerLayer;
}

function setBasemap(name) {
  if (!state.tileLayers[name] || name === state.basemap) return;
  state.map.removeLayer(state.tileLayers[state.basemap]);
  state.tileLayers[name].addTo(state.map);
  state.basemap = name;
}

// ---------------------------------------------------------------------
// Live rainfall radar (RainViewer public API — real current conditions,
// no API key required). Deliberately NOT presented as an explanation for
// the historical/simulated QoE panel above: that panel covers a fixed
// past week, this shows whatever is raining right now, and those two
// time windows generally don't line up. Untested in this sandbox (no
// network) — degrades gracefully to a visible error note if the fetch
// fails, e.g. offline at demo time.
// ---------------------------------------------------------------------

async function toggleWeatherLayer() {
  if (!state.weatherEnabled) {
    if (state.weatherLayer) state.map.removeLayer(state.weatherLayer);
    document.getElementById("weatherDisclaimer")?.remove();
    renderLegend();
    return;
  }
  try {
    if (!state.weatherLayer) {
      const res = await fetch("https://api.rainviewer.com/public/weather-maps.json");
      const meta = await res.json();
      const frame = meta.radar?.past?.slice(-1)[0];
      if (!frame) throw new Error("no radar frames returned");
      state.weatherLayer = L.tileLayer(
        `https://tilecache.rainviewer.com${frame.path}/256/{z}/{x}/{y}/2/1_1.png`,
        { opacity: 0.5, attribution: "Rain radar: RainViewer.com" }
      );
    }
    state.weatherLayer.addTo(state.map);
    document.getElementById("weatherDisclaimer")?.remove();
    const el = document.createElement("div");
    el.id = "weatherDisclaimer";
    el.className = "whatif-disclaimer";
    el.style.margin = "4px 0";
    el.textContent = "Live current rainfall radar (real, via RainViewer) — for situational context only. The QoE panel above covers a separate simulated week and is not caused by today's real weather.";
    document.querySelector(".map-panel").insertBefore(el, document.getElementById("map"));
  } catch (err) {
    console.error("weather layer failed", err);
    document.getElementById("toggleWeather").checked = false;
    state.weatherEnabled = false;
    alert("Couldn't load live rain radar (no internet access, or RainViewer unreachable). This feature needs a network connection at demo time.");
  }
}

// Human-readable label for a cell — falls back to the raw H3 suffix if a
// place name wasn't assigned (e.g. grid built without the gazetteer step).
function cellName(h3) {
  return state.gridPropsByH3.get(h3)?.place_name || h3.slice(-6);
}

function hoverHtml(h3) {
  const c = state.h3IndexOf.get(h3);
  const t = state.currentT;
  const props = state.gridPropsByH3.get(h3);

  if (state.onset && state.colorBy === "onset") {
    const tier = onsetValue("onset_priority_tier", c);
    const score = onsetValue("next_hour_onset_risk_score", c);
    return `<b>AI onset priority:</b> ${ONSET_TIER_LABELS[tier] ?? tier ?? "—"}<br/>
            <b>Relative risk score:</b> ${score != null ? score.toFixed(2) : "—"} (not a probability)`;
  }
  if (state.onset && state.colorBy === "static_ai") {
    const status = onsetValue("static_data_status", c);
    const v = onsetValue("display_dl_mbps", c);
    return `<b>${status === "observed_ookla" ? "Observed" : "AI-estimated"} DL speed:</b> ${v != null ? v.toFixed(1) : "—"} Mbps`;
  }

  const primary = { qoe: "QoE", demand: "Demand", hotspot: "Hotspot", moran: "Moran", risk: "Risk" }[state.colorBy];
  const val = cellValue(
    { qoe: "qoe", demand: "demand_pressure", hotspot: "hotspot_code", moran: "moran_code", risk: "next_hour_degraded_historical" }[state.colorBy],
    t, c
  );
  const conf = cellValue("confidence", t, c);
  return `<b>${primary}:</b> ${val ?? "—"} · <b>Confidence:</b> ${conf != null ? (conf * 100).toFixed(0) + "%" : "—"}<br/>
          Towers: ${props.tower_count} · Pop (day): ${Math.round(props.population_day ?? 0).toLocaleString()}`;
}

function popupHtml(h3) {
  const c = state.h3IndexOf.get(h3);
  const t = state.currentT;
  const props = state.gridPropsByH3.get(h3);
  const realQoe = cellValue("qoe", t, c);
  const eff = effectiveMetrics(t, c);
  const showEff = state.profile !== "balanced" || state.whatIfEnabled;
  const sq = cellValue("degradation_class", t, c);
  const giz = cellValue("gi_star_z", t, c);
  const hotspotLabel = state.panel.hotspot_code_labels[cellValue("hotspot_code", t, c)];
  const classLabel = ["Normal", "Degraded", "Severe"][sq] ?? "—";
  return `
    <b>${cellName(h3)}</b><br/>
    QoE Index (real, balanced): <b>${realQoe ?? "—"}</b> / 100<br/>
    ${showEff ? `QoE (${state.profile}${state.whatIfEnabled ? ", what-if" : ""}): <b>${eff.qoe ?? "—"}</b> / 100<br/>` : ""}
    Degradation class (real): <b>${classLabel}</b><br/>
    Demand Pressure: <b>${eff.demand?.toFixed?.(0) ?? eff.demand ?? "—"}</b><br/>
    Population (day): <b>${Math.round(props.population_day ?? 0).toLocaleString()}</b><br/>
    Tower sites: <b>${props.tower_count}</b><br/>
    Hotspot (Gi*, latency): <b>${hotspotLabel}</b> (z=${giz ?? "—"})
    ${aiPopupSection(c)}
  `;
}

// Trained AI model block appended to every cell popup, regardless of the
// active color-by mode — a fixed snapshot from the frozen onset/static
// models (see ai_model/), not tied to the time slider.
function aiPopupSection(c) {
  if (!state.onset) return "";
  const status = onsetValue("static_data_status", c);
  const dl = onsetValue("display_dl_mbps", c);
  const reliability = onsetValue("relative_prediction_reliability", c);
  const tier = onsetValue("onset_priority_tier", c);
  const score = onsetValue("next_hour_onset_risk_score", c);
  const rank = onsetValue("onset_risk_rank", c);
  const opStatus = onsetValue("onset_operational_status", c);

  let onsetLine;
  if (opStatus === "currently_degraded") {
    onsetLine = `Currently degraded (AI-scored)`;
  } else if (opStatus === "not_scored_unanchored") {
    onsetLine = `Not scored — cell isn't Ookla-anchored`;
  } else {
    onsetLine = `${ONSET_TIER_LABELS[tier] ?? tier} · relative rank score <b>${score != null ? score.toFixed(2) : "—"}</b>` +
      `${rank != null ? ` (#${Math.round(rank)})` : ""} <span class="not-a-probability">— NOT a %</span>`;
  }

  const explainable = ["current_degradation", "highest_priority", "high_priority", "elevated_watch"].includes(tier) || opStatus === "currently_degraded";
  const explanation = explainable ? explainCell(c) : null;

  return `
    <hr style="margin:6px 0"/>
    <span class="cheat-badge ai" style="font-size:9px">🤖 TRAINED AI MODEL</span><br/>
    ${status === "observed_ookla" ? "Observed Ookla DL" : "AI-estimated DL (no direct label)"}: <b>${dl != null ? dl.toFixed(1) : "—"}</b> Mbps
    ${status === "ai_estimated" ? `· reliability: ${reliability ?? "—"}` : ""}<br/>
    Next-hour onset risk: ${onsetLine}<br/>
    ${explanation ? explainCellHtml(explanation) : ""}
    <span style="font-size:10px;color:#64748b">Score is a relative ranking within a simulated scenario, not a calibrated probability. Reflects the frozen snapshot at ${state.onset.snapshot_timestamp} UTC${state.onsetSnapshotT != null && state.onsetSnapshotT !== state.currentT ? " — the time slider is elsewhere right now, see the banner above the AI panel" : ""}.</span>
  `;
}

// Renders the "why is this cell flagged?" block: top abnormal real
// signals (city-wide z-score at the AI snapshot hour) plus the same
// rule-based cause categorization used in Root Cause / Interventions.
function explainCellHtml(exp) {
  const rows = exp.signals.slice(0, 4).map(s =>
    `<div class="explain-signal">${s.tag} ${s.label}: <b>${s.value.toFixed(1)}${s.unit}</b> — ${s.word}</div>`
  ).join("");
  const qoeRow = exp.qoeDeltaPct != null
    ? `<div class="explain-signal">${exp.qoeDeltaPct <= -10 ? "🔴" : exp.qoeDeltaPct <= -3 ? "🟠" : "🟢"} Recent QoE change: <b>${exp.qoeDeltaPct >= 0 ? "+" : ""}${exp.qoeDeltaPct.toFixed(0)}%</b> (vs. prior hour)</div>`
    : "";
  return `
    <div class="ai-explain-block">
      <div class="ai-explain-title">Why is this cell flagged?</div>
      ${rows}${qoeRow}
      <div class="ai-explain-trigger">Likely trigger: ${exp.likelyTrigger}</div>
    </div>
  `;
}

function renderMapColors() {
  const t = state.currentT;
  state.geoLayer.eachLayer(layer => {
    const h3 = layer.feature.properties.h3_index;
    const c = state.h3IndexOf.get(h3);
    const style = { fillColor: colorForMode(state.colorBy, t, c), color: "#334155", weight: 0.5, dashArray: null };
    if (state.onset && state.colorBy === "static_ai" && onsetValue("static_data_status", c) === "ai_estimated") {
      // Dashed outline flags AI-ESTIMATED cells vs. solid outline for observed_ookla —
      // never let the color alone imply a real measurement where there isn't one.
      style.dashArray = "2,2";
      style.color = "#805ad5";
      style.weight = 0.8;
    }
    if (state.onset && state.colorBy === "onset" && onsetValue("alert_at_frozen_threshold", c)) {
      style.color = "#742a2a";
      style.weight = 1.4;
    }
    layer.setStyle(style);
  });
}

function renderLegend() {
  const el = document.getElementById("mapLegend");
  let html = "";
  if (state.colorBy === "qoe") {
    html = QOE_STOPS.map(([s, c]) => `<span><span class="swatch" style="background:${c}"></span>${s}</span>`).join("");
  } else if (state.colorBy === "demand") {
    html = `<span><span class="swatch" style="background:#2b6cb0"></span>Low</span>
            <span><span class="swatch" style="background:#e53e3e"></span>High</span>`;
  } else if (state.colorBy === "hotspot") {
    html = `<span><span class="swatch" style="background:#e53e3e"></span>Hot 99%</span>
            <span><span class="swatch" style="background:#ed8936"></span>Hot 90–95%</span>
            <span><span class="swatch" style="background:#a0aec0"></span>Not significant</span>
            <span><span class="swatch" style="background:#38a169"></span>Cold spot</span>`;
  } else if (state.colorBy === "moran") {
    html = `<span><span class="swatch" style="background:#e53e3e"></span>HH</span>
            <span><span class="swatch" style="background:#ed8936"></span>HL</span>
            <span><span class="swatch" style="background:#d69e2e"></span>LH</span>
            <span><span class="swatch" style="background:#38a169"></span>LL</span>
            <span><span class="swatch" style="background:#a0aec0"></span>Not significant</span>`;
  } else if (state.colorBy === "risk") {
    html = `<span><span class="swatch" style="background:#e53e3e"></span>Degraded next hour (historical)</span>
            <span><span class="swatch" style="background:#38a169"></span>Not degraded next hour</span>`;
  } else if (state.colorBy === "onset") {
    html = Object.entries(ONSET_TIER_COLORS).map(([k, c]) =>
      `<span><span class="swatch" style="background:${c}"></span>${ONSET_TIER_LABELS[k]}</span>`).join("") +
      `<span class="legend-note">⚠️ <b>Score is a relative ranking/model-support indicator, NOT a calibrated probability.</b> Fixed AI model snapshot at ${state.onset?.snapshot_timestamp ?? "—"} UTC — independent of the time slider (see banner in the AI panel). Thick dark-red outline = above the frozen alert threshold.</span>`;
  } else if (state.colorBy === "static_ai") {
    html = `<span><span class="swatch" style="background:#e53e3e"></span>Low Mbps</span>
            <span><span class="swatch" style="background:#2f855a"></span>High Mbps</span>
            <span class="legend-note">Solid outline = real observed Ookla speed · Dashed purple outline = AI-estimated (no direct label)</span>`;
  } else if (state.colorBy === "comparison") {
    html = Object.entries(COMPARISON_COLORS).map(([k, c]) =>
      `<span><span class="swatch" style="background:${c}"></span>${COMPARISON_LABELS[k]}</span>`).join("") +
      `<span class="legend-note">Traditional = statistically significant Gi* latency hotspot (hot_90/95/99). Both methods compared at the AI snapshot hour (${state.onset?.snapshot_timestamp ?? "—"} UTC).</span>`;
  }
  el.innerHTML = html;
}

// ---------------------------------------------------------------------
// Place name search — matches against the real place_name (from the
// gazetteer join in the grid geometry) and raw h3_index, both already
// loaded client-side. No new data files needed.
// ---------------------------------------------------------------------

function initPlaceSearch() {
  const input = document.getElementById("placeSearchInput");
  const btn = document.getElementById("placeSearchBtn");
  const resultsEl = document.getElementById("placeSearchResults");
  const datalist = document.getElementById("placeSearchList");

  // Datalist gets deduplicated BASE names (gazetteer place names carry a
  // "(direction, distance)" suffix per cell, e.g. "Cyberjaya (W, 3.9km)" —
  // showing all ~4500 of those in a native dropdown is much less useful
  // than ~120 base names; the custom results list below still searches
  // the full per-cell strings, so specific nearby cells are still findable.
  const baseNames = new Set();
  state.grid.features.forEach(f => {
    const name = f.properties.place_name;
    if (!name) return;
    baseNames.add(name.replace(/\s*\([^)]*\)\s*$/, "").trim());
  });
  datalist.innerHTML = [...baseNames].sort().map(n => `<option value="${n}">`).join("");

  function runSearch(query) {
    query = query.trim().toLowerCase();
    if (query.length < 2) {
      resultsEl.classList.add("d-none");
      resultsEl.innerHTML = "";
      return;
    }
    const matches = [];
    for (const f of state.grid.features) {
      const p = f.properties;
      const name = p.place_name || "";
      if (name.toLowerCase().includes(query) || p.h3_index.toLowerCase().includes(query)) {
        matches.push(p);
        if (matches.length >= 8) break;
      }
    }
    if (!matches.length) {
      resultsEl.innerHTML = `<div class="result-empty">No place name or H3 index matches "${query}".</div>`;
    } else {
      resultsEl.innerHTML = matches.map(p => `
        <div class="result-item" data-h3="${p.h3_index}">
          <span>${p.place_name || "(unnamed cell)"}</span>
          <span class="h3-tag">${p.h3_index.slice(-6)}</span>
        </div>
      `).join("");
    }
    resultsEl.classList.remove("d-none");
  }

  let debounceTimer = null;
  input.addEventListener("input", e => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => runSearch(e.target.value), 150);
  });
  input.addEventListener("focus", () => { if (input.value.trim().length >= 2) runSearch(input.value); });
  input.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      const first = resultsEl.querySelector(".result-item");
      if (first) goToSearchResult(first.dataset.h3);
    } else if (e.key === "Escape") {
      resultsEl.classList.add("d-none");
    }
  });
  btn.addEventListener("click", () => {
    const first = resultsEl.querySelector(".result-item");
    if (first) goToSearchResult(first.dataset.h3);
    else runSearch(input.value);
  });
  resultsEl.addEventListener("click", e => {
    const item = e.target.closest(".result-item");
    if (item) goToSearchResult(item.dataset.h3);
  });
  document.addEventListener("click", e => {
    if (!e.target.closest(".map-search-row") && !e.target.closest(".place-search-results")) {
      resultsEl.classList.add("d-none");
    }
  });

  function goToSearchResult(h3) {
    const props = state.gridPropsByH3.get(h3);
    input.value = props?.place_name || h3;
    resultsEl.classList.add("d-none");
    flyToCell(h3);
  }
}

function flyToCell(h3) {
  const props = state.gridPropsByH3.get(h3);
  if (!props) return;
  state.map.flyTo([props.centroid_lat, props.centroid_lon], 14, { duration: 0.6 });
  state.geoLayer.eachLayer(layer => {
    if (layer.feature.properties.h3_index === h3) layer.openPopup();
  });
}

// ---------------------------------------------------------------------
// Compare mode — click cells on the map to line them up side by side
// ---------------------------------------------------------------------

function toggleCompareMode() {
  state.compareMode = !state.compareMode;
  const btn = document.getElementById("compareToggleBtn");
  btn.classList.toggle("btn-primary", state.compareMode);
  btn.classList.toggle("btn-outline-primary", !state.compareMode);
  btn.textContent = state.compareMode ? "Click hexes to compare…" : "Compare cells";
  document.getElementById("comparePanel").classList.toggle("d-none", !state.compareMode);
}

function toggleCompareCell(h3) {
  const i = state.compareCells.indexOf(h3);
  if (i >= 0) {
    state.compareCells.splice(i, 1);
  } else {
    if (state.compareCells.length >= COMPARE_MAX) state.compareCells.shift();
    state.compareCells.push(h3);
  }
  renderComparePanel();
}

function renderComparePanel() {
  const t = state.currentT;
  const tableEl = document.getElementById("compareTable");
  if (!state.compareCells.length) {
    tableEl.innerHTML = `<p class="text-muted" style="font-size:12px">No cells selected yet.</p>`;
    Plotly.purge("chartCompare");
    return;
  }
  const rows = state.compareCells.map(h3 => {
    const c = state.h3IndexOf.get(h3);
    const props = state.gridPropsByH3.get(h3);
    return {
      h3, qoe: cellValue("qoe", t, c), demand: cellValue("demand_pressure", t, c),
      conf: cellValue("confidence", t, c), pop: props.population_day, towers: props.tower_count,
    };
  });
  tableEl.innerHTML = `
    <table>
      <thead><tr><th>Cell</th><th>QoE</th><th>Demand</th><th>Conf.</th><th>Pop (day)</th><th>Towers</th><th></th></tr></thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td class="cell-id">${cellName(r.h3)}</td>
            <td>${r.qoe?.toFixed(0) ?? "—"}</td>
            <td>${r.demand?.toFixed(0) ?? "—"}</td>
            <td>${r.conf != null ? (r.conf * 100).toFixed(0) + "%" : "—"}</td>
            <td>${Math.round(r.pop ?? 0).toLocaleString()}</td>
            <td>${r.towers}</td>
            <td><span class="compare-remove" onclick="toggleCompareCell('${r.h3}')">✕</span></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;

  const traces = state.compareCells.map(h3 => {
    const c = state.h3IndexOf.get(h3);
    const y = [];
    for (let tt = 0; tt < state.panel.num_timestamps; tt++) y.push(cellValue("qoe", tt, c));
    return { x: state.cityTS.map(r => new Date(r.timestamp)), y, type: "scatter", mode: "lines", name: cellName(h3) };
  });
  Plotly.newPlot("chartCompare", traces, {
    margin: { l: 40, r: 10, t: 10, b: 24 }, height: 200,
    legend: { orientation: "h", font: { size: 9 } },
    yaxis: { title: "QoE", tickfont: { size: 9 } }, xaxis: { tickfont: { size: 9 } },
  }, plotlyConfig());
}

// ---------------------------------------------------------------------
// Time-series charts (Plotly)
// ---------------------------------------------------------------------

function aggregatedSeries() {
  if (state.granularity === "hourly") return state.cityTS;
  // Daily bucket aggregation (client-side, on the already-tiny cityTS array —
  // this is the "pre-aggregate for performance" path for the trend charts).
  const byDay = new Map();
  state.cityTS.forEach(r => {
    const day = new Date(r.timestamp).toISOString().slice(0, 10);
    if (!byDay.has(day)) byDay.set(day, { timestamp: r.timestamp, qoeSum: 0, demandSum: 0, n: 0, normal: 0, degraded: 0, severe: 0 });
    const b = byDay.get(day);
    b.qoeSum += r.avg_qoe ?? 0; b.demandSum += r.avg_demand_pressure ?? 0; b.n++;
    b.normal += r.normal_count; b.degraded += r.degraded_count; b.severe += r.severe_count;
  });
  return [...byDay.values()].map(b => ({
    timestamp: b.timestamp, avg_qoe: b.qoeSum / b.n, avg_demand_pressure: b.demandSum / b.n,
    normal_count: Math.round(b.normal / b.n), degraded_count: Math.round(b.degraded / b.n), severe_count: Math.round(b.severe / b.n),
  }));
}

function buildCharts() {
  const series = aggregatedSeries();
  const ts = series.map(r => new Date(r.timestamp));

  Plotly.newPlot("chartQoe", [{
    x: ts, y: series.map(r => r.avg_qoe),
    type: "scatter", mode: "lines", line: { color: "#2b6cb0" }, name: "Avg QoE",
  }], plotlyLayout("Average QoE Score"), plotlyConfig());

  Plotly.newPlot("chartDemand", [{
    x: ts, y: series.map(r => r.avg_demand_pressure),
    type: "scatter", mode: "lines", fill: "tozeroy",
    line: { color: "#d69e2e" }, name: "Avg Demand Pressure",
  }], plotlyLayout("Average Demand Pressure Index"), plotlyConfig());

  Plotly.newPlot("chartSeverity", [
    { x: ts, y: series.map(r => r.normal_count), type: "bar", name: "Normal", marker: { color: "#38a169" } },
    { x: ts, y: series.map(r => r.degraded_count), type: "bar", name: "Degraded", marker: { color: "#d69e2e" } },
    { x: ts, y: series.map(r => r.severe_count), type: "bar", name: "Severe", marker: { color: "#e53e3e" } },
  ], { ...plotlyLayout("Degradation Severity Mix"), barmode: "stack" }, plotlyConfig());

  ["chartQoe", "chartDemand", "chartSeverity"].forEach(id => {
    document.getElementById(id).on("plotly_click", ev => {
      if (state.granularity !== "hourly") return; // daily points don't map 1:1 to an hour index
      const idx = ev.points[0].pointIndex;
      state.currentT = idx;
      document.getElementById("timeSlider").value = idx;
      renderAll();
    });
  });
  updateChartRangeShading();
}

function plotlyLayout(title) {
  return {
    title: { text: title, font: { size: 12, color: "#2d3748" } },
    margin: { l: 40, r: 10, t: 28, b: 24 },
    height: 170,
    showlegend: title.includes("Severity"),
    legend: { orientation: "h", font: { size: 9 } },
    xaxis: { showgrid: false, tickfont: { size: 9 } },
    yaxis: { showgrid: true, gridcolor: "#f1f5f9", tickfont: { size: 9 } },
    shapes: [
      {
        type: "line", x0: state.cityTS[state.currentT]?.timestamp, x1: state.cityTS[state.currentT]?.timestamp,
        xref: "x", y0: 0, y1: 1, yref: "paper", line: { color: "#e53e3e", dash: "dot", width: 1 },
      },
      {
        type: "rect", x0: state.cityTS[state.rangeStart]?.timestamp, x1: state.cityTS[state.rangeEnd]?.timestamp,
        xref: "x", y0: 0, y1: 1, yref: "paper", fillcolor: "#2b6cb0", opacity: 0.08, line: { width: 0 },
      },
    ],
  };
}
function plotlyConfig() { return { displayModeBar: false, responsive: true }; }

function updateChartMarkers() {
  ["chartQoe", "chartDemand", "chartSeverity"].forEach(id => {
    Plotly.relayout(id, {
      "shapes[0].x0": state.cityTS[state.currentT].timestamp,
      "shapes[0].x1": state.cityTS[state.currentT].timestamp,
    });
  });
}

function updateChartRangeShading() {
  const x0 = state.cityTS[state.rangeStart].timestamp;
  const x1 = state.cityTS[state.rangeEnd].timestamp;
  ["chartQoe", "chartDemand", "chartSeverity"].forEach(id => {
    Plotly.relayout(id, {
      "shapes[1]": {
        type: "rect", x0, x1, xref: "x", y0: 0, y1: 1, yref: "paper",
        fillcolor: "#2b6cb0", opacity: 0.08, line: { width: 0 },
      },
    });
  });
  document.getElementById("rangeLabel").textContent =
    `${new Date(x0).toUTCString().slice(0, 22)} → ${new Date(x1).toUTCString().slice(0, 22)}`;
}

function renderRangeSummary() {
  const a = state.rangeStart, b = state.rangeEnd;
  const slice = state.cityTS.slice(a, b + 1);
  if (!slice.length) return;
  const avgQoe = slice.reduce((s, r) => s + (r.avg_qoe ?? 0), 0) / slice.length;
  const totalDegradedHours = slice.reduce((s, r) => s + r.degraded_count + r.severe_count, 0);
  const worstHour = slice.reduce((w, r) => (r.avg_qoe ?? 100) < (w.avg_qoe ?? 100) ? r : w, slice[0]);
  document.getElementById("rangeSummary").innerHTML =
    `Over the selected <b>${slice.length}h</b> window: avg QoE <b>${avgQoe.toFixed(0)}</b>, ` +
    `<b>${totalDegradedHours.toLocaleString()}</b> cell-hours degraded or severe, ` +
    `worst hour was <b>${new Date(worstHour.timestamp).toUTCString().slice(0, 22)}</b> (avg QoE ${worstHour.avg_qoe?.toFixed(0)}).`;
}

// ---------------------------------------------------------------------
// KPI cards
// ---------------------------------------------------------------------

function statusClass(value, goodAbove, warnAbove) {
  if (value >= goodAbove) return "status-green";
  if (value >= warnAbove) return "status-yellow";
  return "status-red";
}

function renderKpis() {
  const t = state.currentT;
  const cur = state.cityTS[t];
  const C = state.panel.num_cells;

  let peakDemand = -Infinity, popAffected = 0;
  let whatIfDegraded = 0, whatIfQoeSum = 0, whatIfN = 0;
  const thr = state.summary.qoe_thresholds;

  for (let c = 0; c < C; c++) {
    const d = cellValue("demand_pressure", t, c);
    if (d != null && d > peakDemand) peakDemand = d;
    const cls = cellValue("degradation_class", t, c);
    if (cls >= 1) {
      const h3 = state.panel.h3_ids[c];
      popAffected += state.gridPropsByH3.get(h3)?.population_day || 0;
    }
    if (state.whatIfEnabled) {
      const m = applyWhatIf(t, c);
      if (m.dl != null) {
        whatIfN++;
        whatIfQoeSum += effectiveMetrics(t, c).qoe;
        if (m.dl < thr.dl_degraded_mbps || m.rtt > thr.rtt_degraded_ms) whatIfDegraded++;
      }
    }
  }

  const anchoredPct = state.summary.ookla_anchored_pct;

  // Overall QoE and Degraded Cells already appear in the Network Health /
  // Degraded Cells cards up top (exec overview row) — no need to repeat
  // them here, so this row sticks to the metrics that are unique to it.
  const cards = [
    { label: "Peak Demand Pressure", value: peakDemand.toFixed(0), delta: "", status: statusClass(100 - peakDemand, 30, 15) },
    { label: "Population Affected", value: Math.round(popAffected).toLocaleString(), delta: "", status: statusClass(1, 1, 0) },
    { label: "Total Cells", value: C.toLocaleString(), delta: "", status: "status-green" },
    { label: "Ookla-Anchored Cells", value: anchoredPct + "%", delta: "", status: statusClass(anchoredPct, 50, 25) },
  ];

  document.getElementById("kpiRow").innerHTML = cards.map(k => `
    <div class="kpi-card">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-delta">${k.delta}</div>
      <div class="kpi-status-bar ${k.status}"></div>
    </div>
  `).join("");

  document.getElementById("timeLabel").textContent =
    new Date(cur.timestamp).toUTCString().replace(" GMT", " UTC");
  document.getElementById("liveIndicator").innerHTML =
    `<span class="dot"></span> Scenario data as of ${new Date(state.summary.time_range.end).toUTCString().replace(" GMT", " UTC")}`;

  if (state.whatIfEnabled && whatIfN) {
    const realDegradedCount = cur.degraded_count + cur.severe_count;
    const degradedDelta = realDegradedCount - whatIfDegraded;
    const qoeDelta = (whatIfQoeSum / whatIfN) - cur.avg_qoe;
    document.getElementById("whatIfSummary").innerHTML =
      `<b>Before → After (${state.whatIfPct >= 0 ? "+" : ""}${state.whatIfPct}% capacity):</b><br/>` +
      `Degraded cells: <b>${realDegradedCount}</b> → <b>${whatIfDegraded}</b> ` +
      `(${degradedDelta >= 0 ? "-" : "+"}${Math.abs(degradedDelta)} cells, ` +
      `${(100 * degradedDelta / Math.max(realDegradedCount, 1)).toFixed(0)}% reduction)<br/>` +
      `Avg QoE: <b>${cur.avg_qoe?.toFixed(0)}</b> → <b>${(whatIfQoeSum / whatIfN).toFixed(0)}</b> ` +
      `(${qoeDelta >= 0 ? "+" : ""}${qoeDelta.toFixed(1)} pts)`;
  } else {
    document.getElementById("whatIfSummary").innerHTML = "";
  }
}

function deltaStr(cur, prev, inverse) {
  if (cur == null || prev == null) return "";
  const diff = cur - prev;
  if (Math.abs(diff) < 0.05) return "→ no change";
  const up = diff > 0;
  const bad = inverse ? up : !up;
  const arrow = up ? "▲" : "▼";
  return `<span class="${bad ? 'up' : 'down'}">${arrow} ${Math.abs(diff).toFixed(1)} vs prior hour</span>`;
}

// ---------------------------------------------------------------------
// Hotspot lists
// ---------------------------------------------------------------------

function renderHotspotLists() {
  const t = state.currentT;
  const C = state.panel.num_cells;
  const minConf = state.confidenceThreshold;
  const rows = [];
  for (let c = 0; c < C; c++) {
    const conf = cellValue("confidence", t, c);
    if (conf != null && conf < minConf) continue; // confidence filter
    rows.push({
      c, h3: state.panel.h3_ids[c],
      qoe: cellValue("qoe", t, c),
      demand: cellValue("demand_pressure", t, c),
      hotspot: cellValue("hotspot_code", t, c),
    });
  }

  const worst = [...rows].sort((a, b) => a.qoe - b.qoe).slice(0, 5);
  const highDemand = [...rows].sort((a, b) => b.demand - a.demand).slice(0, 5);

  renderRankedList("worstQoeList", worst, r => `${r.qoe?.toFixed(0)} QoE`);
  renderRankedList("highDemandList", highDemand, r => `${r.demand?.toFixed(0)} pressure`);

  // Persistent hotspot watch: hot_95/hot_99 (codes 5,6) for >=3 of last 6 hours
  const WINDOW = 6, THRESHOLD = 3;
  const persistent = [];
  for (let c = 0; c < C; c++) {
    const conf = cellValue("confidence", t, c);
    if (conf != null && conf < minConf) continue;
    let count = 0;
    for (let dt = 0; dt < WINDOW && t - dt >= 0; dt++) {
      const code = cellValue("hotspot_code", t - dt, c);
      if (code === 5 || code === 6) count++;
    }
    if (count >= THRESHOLD) persistent.push({ c, h3: state.panel.h3_ids[c], count });
  }
  persistent.sort((a, b) => b.count - a.count);
  renderRankedList("persistentList", persistent.slice(0, 5), r => `${r.count}/${WINDOW}h hot`);

  // Emerging: QoE dropped >20% vs 2 hours ago, not currently a confirmed hotspot
  const emerging = [];
  if (t >= 2) {
    for (let c = 0; c < C; c++) {
      const conf = cellValue("confidence", t, c);
      if (conf != null && conf < minConf) continue;
      const now = cellValue("qoe", t, c);
      const prior = cellValue("qoe", t - 2, c);
      const code = cellValue("hotspot_code", t, c);
      if (now != null && prior && prior > 0 && (prior - now) / prior > 0.2 && code !== 5 && code !== 6) {
        emerging.push({ c, h3: state.panel.h3_ids[c], drop: (prior - now) / prior * 100 });
      }
    }
  }
  emerging.sort((a, b) => b.drop - a.drop);
  renderRankedList("emergingList", emerging.slice(0, 5), r => `-${r.drop.toFixed(0)}%`);
}

function renderRankedList(elId, rows, metricFn) {
  const el = document.getElementById(elId);
  if (!rows.length) { el.innerHTML = `<li class="text-muted">None at this hour</li>`; return; }
  el.innerHTML = rows.map(r => `
    <li onclick="flyToCell('${r.h3}')">
      <span class="cell-id">${cellName(r.h3)}</span>
      <span class="metric">${metricFn(r)}</span>
    </li>
  `).join("");
}

// ---------------------------------------------------------------------
// Root cause heuristic (computed client-side from real features; the
// pipeline itself does not export a "likely cause" column)
// ---------------------------------------------------------------------

const CAUSE_LABELS = {
  congestion: "Congestion", coverage: "Coverage Gap",
  indoor: "Indoor Attenuation", backhaul: "Backhaul Limitation",
  insufficient: "Insufficient Data",
};

function causeScoresForCell(t, c, h3) {
  const props = state.gridPropsByH3.get(h3);
  const demand = cellValue("demand_pressure", t, c) ?? 0;
  const rtt = cellValue("rtt_ms", t, c) ?? 0;
  const confidence = cellValue("confidence", t, c) ?? 1;

  const congestion = demand / 100;
  const coverage = Math.max(0, Math.min(1, (props.dist_to_nearest_tower_m ?? 0) / 1000));
  const indoor = props.built_up_ratio ?? 0;
  const rttNorm = Math.max(0, Math.min(1, rtt / 200));
  const backhaul = Math.max(0, rttNorm - congestion);
  const insufficient = 1 - Math.max(0, Math.min(1, confidence));

  const raw = { congestion, coverage, indoor, backhaul, insufficient };
  const sum = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  const pct = {};
  Object.entries(raw).forEach(([k, v]) => pct[k] = v / sum);
  return pct;
}

function dominantCause(pct) {
  return Object.entries(pct).sort((a, b) => b[1] - a[1])[0][0];
}

function renderRootCause() {
  const t = state.currentT;
  const C = state.panel.num_cells;
  const totals = { congestion: 0, coverage: 0, indoor: 0, backhaul: 0, insufficient: 0 };
  let nDegraded = 0;

  for (let c = 0; c < C; c++) {
    const cls = cellValue("degradation_class", t, c);
    if (cls < 1) continue;
    nDegraded++;
    const h3 = state.panel.h3_ids[c];
    const pct = causeScoresForCell(t, c, h3);
    Object.keys(totals).forEach(k => totals[k] += pct[k]);
  }

  const keys = Object.keys(totals);
  const labels = keys.map(k => CAUSE_LABELS[k]);
  const values = keys.map(k => nDegraded ? (100 * totals[k] / nDegraded) : 0);

  Plotly.newPlot("chartRootCause", [{
    x: values, y: labels, type: "bar", orientation: "h",
    marker: { color: ["#e53e3e", "#d69e2e", "#2b6cb0", "#805ad5", "#a0aec0"] },
  }], { margin: { l: 110, r: 20, t: 10, b: 30 }, height: 170, xaxis: { title: "% of degraded cells", tickfont: { size: 9 } }, yaxis: { tickfont: { size: 10 } } }, plotlyConfig());

  if (nDegraded === 0) {
    document.getElementById("rootCauseSummary").textContent = "No degraded cells at this hour.";
    return;
  }
  const sorted = keys.map((k, i) => [k, values[i]]).sort((a, b) => b[1] - a[1]);
  document.getElementById("rootCauseSummary").innerHTML =
    `Primary cause across <b>${nDegraded}</b> degraded cells is <b>${CAUSE_LABELS[sorted[0][0]]}</b> (${sorted[0][1].toFixed(0)}%), ` +
    `followed by <b>${CAUSE_LABELS[sorted[1][0]]}</b> (${sorted[1][1].toFixed(0)}%).`;
}

// ---------------------------------------------------------------------
// Interventions
// ---------------------------------------------------------------------

const ACTIONS = {
  congestion: "Traffic steering or temporary capacity augmentation (e.g. Cell on Wheels) during this peak window.",
  coverage: "Evaluate a new small cell or macro site — nearest tower is unusually far from this cell.",
  indoor: "Conduct an indoor walk test; consider DAS or Wi-Fi offload for this high built-up-ratio area.",
  backhaul: "Investigate backhaul link utilisation independent of radio congestion.",
  insufficient: "Low measurement confidence — schedule a targeted field survey before acting.",
};

function renderInterventions() {
  const t = state.currentT;
  const C = state.panel.num_cells;
  const minConf = state.confidenceThreshold;
  const hideDone = document.getElementById("hideDoneToggle").checked;
  const candidates = [];
  for (let c = 0; c < C; c++) {
    const cls = cellValue("degradation_class", t, c);
    if (cls < 1) continue;
    const conf = cellValue("confidence", t, c);
    if (conf != null && conf < minConf) continue;
    const h3 = state.panel.h3_ids[c];
    if (hideDone && state.tracking[h3] === "done") continue;
    candidates.push({
      c, h3,
      priority: cellValue("priority_base_score", t, c) ?? 0,
      qoe: cellValue("qoe", t, c),
      demand: cellValue("demand_pressure", t, c),
      rtt: cellValue("rtt_ms", t, c),
      historicalHotspotFreq: cellValue("historical_hotspot_freq", t, c),
      population: state.gridPropsByH3.get(h3)?.population_day ?? 0,
      cls,
    });
  }
  candidates.sort((a, b) => b.priority - a.priority);
  const top = candidates.slice(0, 5);
  state.lastCandidates = candidates; // reused by renderExecOverview() — same filters (confidence, hide-done)

  const el = document.getElementById("interventionsList");
  if (!top.length) { el.innerHTML = `<p class="text-muted">No degraded cells match the current filters at this hour.</p>`; return; }

  // P1/P2/P3 (item #8): same underlying urgency signal as before (severity +
  // affected population), just labelled the way an ops team actually talks
  // about ticket priority.
  const TIER_META = { high: { p: "P1", cls: "p1" }, medium: { p: "P2", cls: "p2" }, low: { p: "P3", cls: "p3" } };

  el.innerHTML = top.map((cand, i) => {
    const pct = causeScoresForCell(t, cand.c, cand.h3);
    const cause = dominantCause(pct);
    const urgency = cand.cls === 2 && cand.population > 1000 ? "high" : (cand.cls === 2 || cand.population > 1000 ? "medium" : "low");
    const tier = TIER_META[urgency];
    const status = state.tracking[cand.h3];
    const contact = OPERATOR_CONTACTS[cand.h3] || "Escalate to network operations (no per-cell operator data in this dataset)";

    // AI evidence row — only if the AI package is loaded, and clearly marked
    // as the frozen latest-run score rather than implying it's for hour t.
    let aiEvidence = "";
    if (state.onset) {
      const idx = state.h3IndexOf.get(cand.h3);
      const tierOnset = onsetValue("onset_priority_tier", idx);
      const score = onsetValue("next_hour_onset_risk_score", idx);
      const opStatus = onsetValue("onset_operational_status", idx);
      const label = opStatus === "currently_degraded" ? "currently degraded (AI-scored)"
        : opStatus === "not_scored_unanchored" ? "not scored"
        : `${ONSET_TIER_LABELS[tierOnset] ?? tierOnset}${score != null ? `, rank score ${score.toFixed(2)}` : ""}`;
      aiEvidence = ` · AI onset risk (latest run): ${label}`;
    }
    const persistentHotspot = cand.historicalHotspotFreq != null
      ? (cand.historicalHotspotFreq >= 0.3 ? "Yes" : "No") + ` (${(cand.historicalHotspotFreq * 100).toFixed(0)}% of hours)`
      : "—";

    return `
      <div class="intervention-card urgency-${urgency}${status === "done" ? " is-done" : ""}">
        <div onclick="flyToCell('${cand.h3}')" style="cursor:pointer">
          <div class="title">${cellName(cand.h3)}
            <span class="urgency-pill urgency-${urgency}">${tier.p} · ${urgency.toUpperCase()}</span>
          </div>
          <div class="row"><b>Problem:</b> ${CAUSE_LABELS[cause]} · class ${["Normal", "Degraded", "Severe"][cand.cls]}</div>
          <div class="row evidence-row"><b>Evidence:</b> QoE ${cand.qoe?.toFixed(0) ?? "—"} · Demand ${cand.demand?.toFixed(0) ?? "—"} · RTT ${cand.rtt?.toFixed(0) ?? "—"}ms · Persistent hotspot: ${persistentHotspot}${aiEvidence}</div>
          <div class="row">Affected population (day): ${Math.round(cand.population).toLocaleString()}</div>
          <div class="row">Recommended action: ${ACTIONS[cause]}</div>
          <div class="row">Contact: ${contact}</div>
          <div class="row cost-note">Illustrative cost range (edit ILLUSTRATIVE_COST_RANGES to match your org): ${ILLUSTRATIVE_COST_RANGES[cause]}</div>
        </div>
        <div class="track-actions">
          <button class="${status === 'planned' ? 'active' : ''}" onclick="setTracking('${cand.h3}','planned')">Mark planned</button>
          <button class="${status === 'investigating' ? 'active' : ''}" onclick="setTracking('${cand.h3}','investigating')">Investigating</button>
          <button class="${status === 'done' ? 'active' : ''}" onclick="setTracking('${cand.h3}','done')">Mark resolved</button>
          ${status ? `<button onclick="setTracking('${cand.h3}', null)">Clear</button>` : ""}
        </div>
      </div>
    `;
  }).join("");
}

function setTracking(h3, status) {
  if (status) state.tracking[h3] = status; else delete state.tracking[h3];
  saveTracking();
  renderInterventions();
}

// ---------------------------------------------------------------------
// Data trust panel
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Trained AI model panel — static spatial-QoE ensemble + next-hour onset
// model (see ai_model/). This is a single fixed snapshot, not part of the
// 168-hour animated panel, so it's rendered once at boot rather than on
// every renderAll() tick.
// ---------------------------------------------------------------------

function initAiPanel() {
  const panelEl = document.getElementById("aiModelPanel");
  const cmpEl = document.getElementById("detectionComparisonPanel");
  if (!state.onset) {
    if (panelEl) panelEl.classList.add("d-none");
    if (cmpEl) cmpEl.classList.add("d-none");
    document.querySelectorAll('#colorBySelect option[value="onset"], #colorBySelect option[value="static_ai"], #colorBySelect option[value="comparison"]')
      .forEach(opt => opt.disabled = true);
    return;
  }
  panelEl.classList.remove("d-none");
  if (cmpEl) cmpEl.classList.remove("d-none");
  renderAiPanel();
  renderModelBehaviorBadge();
  renderAiTimeBanner();
  renderDetectionComparison();
  populateAiModelModal();
  populateForecastValidation();
}

// Item #1 (time-awareness, honestly): the onset model only has ONE scored
// hour (state.onsetSnapshotT). Rather than fake per-hour forecasts, this
// banner is explicit about which hour the AI numbers below actually
// describe, and whether that matches what the time slider is showing.
function renderAiTimeBanner() {
  const el = document.getElementById("aiTimeBanner");
  if (!el || !state.onset) return;
  const atSnapshot = state.onsetSnapshotT != null && state.currentT === state.onsetSnapshotT;
  const snapLabel = state.onset.snapshot_timestamp + " UTC";
  if (state.onsetSnapshotT == null || state.onsetSnapshotT === -1) {
    el.className = "ai-time-banner warn";
    el.innerHTML = `⚠️ <b>AI Forecast — Latest Available Model Run.</b> Snapshot hour (${snapLabel}) isn't one of the panel's 168 hours — treat the AI layer as a standalone reference run, not part of this time series.`;
  } else if (atSnapshot) {
    el.className = "ai-time-banner ok";
    el.innerHTML = `✅ Time slider is on the AI model's scored hour (<b>${snapLabel}</b>) — the forecast below matches what you're viewing.`;
  } else {
    el.className = "ai-time-banner warn";
    el.innerHTML = `⚠️ <b>AI Forecast — Latest Available Model Run.</b> You're viewing <b>${state.panel.timestamps_iso[state.currentT]?.replace("T", " ").replace("Z", "")} UTC</b>, but the model has only ever been run once, for <b>${snapLabel}</b>. The AI panel below always shows that one run, not a forecast for your currently selected hour. <a href="#" onclick="jumpToAiSnapshotHour(); return false;">Jump to that hour</a>`;
  }
}

function jumpToAiSnapshotHour() {
  if (state.onsetSnapshotT == null || state.onsetSnapshotT === -1) return;
  state.currentT = state.onsetSnapshotT;
  document.getElementById("timeSlider").value = state.currentT;
  renderAll();
}

// Item #3 (score vs. probability): a small, unmissable badge stating the
// model's actual precision/recall trade-off, computed from the frozen
// model card — not a subjective claim.
function renderModelBehaviorBadge() {
  const el = document.getElementById("aiModelBehaviorBadge");
  const card = state.modelCard;
  if (!el || !card) return;
  const { precision, recall } = card.onset_model.final_test;
  let word = "Mixed alert profile";
  if (recall >= 0.7 && precision < 0.3) word = "High recall / high alert volume";
  else if (precision >= 0.6 && recall < 0.5) word = "High precision / conservative alerting";
  el.innerHTML = `⚠️ Model behavior: <b>${word}</b> — final-test precision ${(precision * 100).toFixed(0)}%, recall ${(recall * 100).toFixed(0)}%. A risk score is a rank, not a % chance of failure.`;
}

function renderAiPanel() {
  const C = state.panel.num_cells;
  const rows = [];
  for (let c = 0; c < C; c++) {
    rows.push({
      c, h3: state.panel.h3_ids[c],
      tier: onsetValue("onset_priority_tier", c),
      score: onsetValue("next_hour_onset_risk_score", c),
      rank: onsetValue("onset_risk_rank", c),
      opStatus: onsetValue("onset_operational_status", c),
      dl: onsetValue("display_dl_mbps", c),
    });
  }

  const degradedNow = rows.filter(r => r.opStatus === "currently_degraded");
  const topRisk = rows
    .filter(r => r.opStatus === "onset_scored" && r.rank != null)
    .sort((a, b) => a.rank - b.rank)
    .slice(0, 10);
  // Cached for the Executive Overview panel so it doesn't redo this scan.
  state.aiSummary = { degradedNowCount: degradedNow.length, topRiskCell: topRisk[0] || null };

  // Metric text is intentionally short here: this list sits in the narrower
  // right column, and the column header already states what it's ranked by
  // ("Currently degraded" / "Highest next-hour onset risk"), so repeating
  // that phrase on every row just crowds the row and forces awkward wraps.
  renderRankedList("aiDegradedNowList", degradedNow.slice(0, 10), r => r.dl != null ? `${r.dl.toFixed(0)} Mbps` : "degraded now");
  renderRankedList("aiTopRiskList", topRisk, r =>
    `<span class="tier-dot" style="background:${ONSET_TIER_COLORS[r.tier] || "#cbd5e1"}" title="${ONSET_TIER_LABELS[r.tier] ?? r.tier}"></span>${r.score?.toFixed(2) ?? "—"} <span class="not-a-probability">rank</span>`
  );

  const meta = state.onset;
  document.getElementById("aiSnapshotNote").innerHTML =
    `Frozen snapshot: <b>${meta.snapshot_timestamp} UTC</b> · ` +
    `${meta.static_observed_cells.toLocaleString()} cells with real Ookla labels, ` +
    `${meta.static_ai_estimated_cells.toLocaleString()} AI-estimated · ` +
    `${degradedNow.length} currently degraded · ${topRisk.length ? topRisk.length + "+ " : ""}scored for next-hour onset risk.` +
    `<br/>Model: <code>${meta.model_name}</code> (iteration ${meta.model_best_iteration}, frozen threshold ${meta.model_frozen_threshold.toFixed(3)}). ` +
    `This score ranks relative risk within a simulated hourly scenario — it is not a calibrated probability and has not been validated on real hourly telemetry.`;
}

// ---------------------------------------------------------------------
// Medium-priority #1: AI vs. traditional (Gi*) hotspot detection
// comparison — "does the AI model find things spatial-stats analysis
// misses, or vice versa?" Computed once (both methods pinned to the AI
// snapshot hour), not re-run every slider tick.
// ---------------------------------------------------------------------
function renderDetectionComparison() {
  const el = document.getElementById("detectionComparisonBody");
  if (!el || !state.onset) return;
  const C = state.panel.num_cells;
  const counts = { both: 0, ai_only: 0, traditional_only: 0, neither: 0 };
  const aiOnlyExamples = [];
  for (let c = 0; c < C; c++) {
    const cat = detectionCategory(c);
    counts[cat]++;
    if (cat === "ai_only" && aiOnlyExamples.length < 5) {
      aiOnlyExamples.push({ h3: state.panel.h3_ids[c], c });
    }
  }
  const aiTotal = counts.both + counts.ai_only;
  const tradTotal = counts.both + counts.traditional_only;

  el.innerHTML = `
    <div class="cmp-grid">
      <div class="cmp-stat"><div class="cmp-num">${tradTotal}</div><div class="cmp-label">Traditional hotspot flagged</div></div>
      <div class="cmp-stat"><div class="cmp-num">${aiTotal}</div><div class="cmp-label">AI model flagged</div></div>
      <div class="cmp-stat" style="color:#c05621"><div class="cmp-num">${counts.both}</div><div class="cmp-label">Both methods agree</div></div>
      <div class="cmp-stat" style="color:#ed8936"><div class="cmp-num">${counts.ai_only}</div><div class="cmp-label">AI-only</div></div>
      <div class="cmp-stat" style="color:#d69e2e"><div class="cmp-num">${counts.traditional_only}</div><div class="cmp-label">Traditional-only</div></div>
    </div>
    <p class="text-summary">${counts.ai_only > 0
      ? `The AI model flags <b>${counts.ai_only} cell${counts.ai_only === 1 ? "" : "s"}</b> that traditional Gi* hotspot analysis does not — e.g. ${[...new Set(aiOnlyExamples.map(x => cellName(x.h3)))].join(", ")}${aiOnlyExamples.length < counts.ai_only ? ", …" : ""}. These are candidates where the onset model's time-series/demand features caught something a purely spatial statistic wouldn't.`
      : `At this snapshot hour, every AI-flagged cell is also a traditional Gi* hotspot — no AI-only detections to highlight.`}
      ${counts.traditional_only > 0 ? ` Conversely, <b>${counts.traditional_only}</b> traditional hotspots aren't in the AI's top risk tiers this hour — spatial clustering alone isn't the same signal as the onset model's ranking.` : ""}</p>
    <div class="disclaimer-box mt-2">The AI model doesn't replace spatial analytics — it's a different lens (temporal/demand-driven vs. purely spatial) evaluated on the same simulated hour. Use "🤖 AI vs Hotspot" on the map's Color-by dropdown to see this spatially.</div>
  `;
}

function populateAiModelModal() {
  const card = state.modelCard;
  const el = document.getElementById("aiModelModalBody");
  if (!card || !el) return;

  const fmtPct = v => v != null ? (v * 100).toFixed(1) + "%" : "—";
  el.innerHTML = `
    <p>${card.recommended_pitch_wording ?? ""}</p>

    <h6>Static spatial QoE baseline — ${card.static_model.selected_model}</h6>
    <p class="text-summary">${card.static_model.purpose}</p>
    <table class="table table-sm">
      <tr><td>Observed (real Ookla) cells</td><td><b>${card.static_model.observed_cells.toLocaleString()}</b></td></tr>
      <tr><td>AI-estimated cells</td><td><b>${card.static_model.ai_estimated_cells.toLocaleString()}</b></td></tr>
      <tr><td>Spatial out-of-fold MAE</td><td><b>${card.static_model.spatial_oof_mae_mbps.toFixed(1)} Mbps</b></td></tr>
      <tr><td>Spatial out-of-fold R²</td><td><b>${card.static_model.spatial_oof_r2.toFixed(3)}</b> (rough estimator — see limitations)</td></tr>
    </table>

    <h6>Next-hour degradation-onset model — ${card.onset_model.selected_model}</h6>
    <p class="text-summary">${card.onset_model.purpose} Selected on ${card.onset_model.selection_basis.toLowerCase()}; frozen at iteration ${card.onset_model.frozen_best_iteration}, threshold ${card.onset_model.frozen_validation_threshold.toFixed(3)}.</p>
    <table class="table table-sm">
      <thead><tr><th></th><th>Validation</th><th>Final test (held-out)</th></tr></thead>
      <tbody>
        <tr><td>PR-AUC</td><td>${card.onset_model.validation.pr_auc.toFixed(3)}</td><td>${card.onset_model.final_test.pr_auc.toFixed(3)}</td></tr>
        <tr><td>ROC-AUC</td><td>${card.onset_model.validation.roc_auc.toFixed(3)}</td><td>${card.onset_model.final_test.roc_auc.toFixed(3)}</td></tr>
        <tr><td>Precision @ frozen threshold</td><td>${fmtPct(card.onset_model.validation.precision)}</td><td>${fmtPct(card.onset_model.final_test.precision)}</td></tr>
        <tr><td>Recall @ frozen threshold</td><td>${fmtPct(card.onset_model.validation.recall)}</td><td>${fmtPct(card.onset_model.final_test.recall)}</td></tr>
        <tr><td>F1 @ frozen threshold</td><td>${card.onset_model.validation.f1.toFixed(3)}</td><td>${card.onset_model.final_test.f1.toFixed(3)}</td></tr>
        <tr><td>True / false positives</td><td>${card.onset_model.validation.true_positives} / ${card.onset_model.validation.false_positives}</td><td>${card.onset_model.final_test.true_positives} / ${card.onset_model.final_test.false_positives}</td></tr>
      </tbody>
    </table>
    <p class="text-summary">On the held-out final test, the model caught ${card.onset_model.final_test.true_positives} of ${card.onset_model.final_test.true_positives + card.onset_model.final_test.false_negatives} simulated onsets at the frozen threshold, at the cost of ${card.onset_model.final_test.false_positives} false alarms — this is why the dashboard leads with ranking/Top-K, not a binary alert claim.</p>

    <h6>Stated limitations</h6>
    <ul>${card.limitations.map(l => `<li>${l}</li>`).join("")}</ul>
  `;
}

// ---------------------------------------------------------------------
// Medium-priority #2: Forecast Validation — real confusion matrices
// (validation + held-out final test) and the model card's own alert-
// volume/precision/recall sweep at different cutoffs. Everything here is
// from geoqoe_model_card.json — there is no ground truth for the dashboard's
// own snapshot hour (it's the panel's last hour, so "what actually happened
// next" was never observed), so this deliberately uses the frozen
// evaluation results rather than fabricating a snapshot-level matrix.
// ---------------------------------------------------------------------
function confusionMatrixHtml(m, title) {
  const total = m.true_positives + m.false_positives + m.true_negatives + m.false_negatives;
  return `
    <div class="cm-title">${title} <span class="text-summary">(n = ${total.toLocaleString()} cell-hours)</span></div>
    <table class="cm-table">
      <tr><td></td><td class="cm-head">Actually degraded</td><td class="cm-head">Actually OK</td></tr>
      <tr><td class="cm-head">Predicted high-risk</td><td class="cm-tp">TP ${m.true_positives}</td><td class="cm-fp">FP ${m.false_positives}</td></tr>
      <tr><td class="cm-head">Predicted OK</td><td class="cm-fn">FN ${m.false_negatives}</td><td class="cm-tn">TN ${m.true_negatives.toLocaleString()}</td></tr>
    </table>
  `;
}

function populateForecastValidation() {
  const card = state.modelCard;
  const el = document.getElementById("forecastValidationBody");
  if (!card || !el) return;
  const fmtPct = v => v != null ? (v * 100).toFixed(1) + "%" : "—";
  const om = card.onset_model;

  const topK = om.final_test_top_k || [];
  const cutoffLabel = k => ({
    top_20_per_hour: "Top 20/hr", top_50_per_hour: "Top 50/hr", top_100_per_hour: "Top 100/hr",
    top_1_percent_per_hour: "Top 1%/hr", top_5_percent_per_hour: "Top 5%/hr",
  }[k.cutoff] || k.cutoff);

  el.innerHTML = `
    <p class="text-summary">${om.score_interpretation ?? ""} These numbers come from the model's real held-out evaluation
      (see <code>geoqoe_model_card.json</code>) — there's no observed outcome for the dashboard's own snapshot hour
      to grade against, since it's the last hour in the 168-hour panel.</p>

    <h6>Confusion matrix @ frozen threshold (${om.frozen_validation_threshold.toFixed(3)})</h6>
    <div class="cm-row">
      ${confusionMatrixHtml(om.validation, "Validation set")}
      ${confusionMatrixHtml(om.final_test, "Final test (held-out)")}
    </div>
    <p class="text-summary"><b>Plain-English read:</b> on the final test, the model caught
      ${om.final_test.true_positives} of ${om.final_test.true_positives + om.final_test.false_negatives}
      (${fmtPct(om.final_test.recall)}) real onset events — but ${om.final_test.false_positives} of every
      ${om.final_test.true_positives + om.final_test.false_positives} alerts it raised were false alarms
      (precision ${fmtPct(om.final_test.precision)}). ${om.important_observation ?? ""}</p>

    <h6>Alert-volume trade-off (real, from the frozen model's top-K sweep)</h6>
    <table class="table table-sm">
      <thead><tr><th>Alert budget</th><th>Alerts issued</th><th>Onsets caught</th><th>Recall</th><th>Precision</th></tr></thead>
      <tbody>
        ${topK.map(k => `<tr><td>${cutoffLabel(k)}</td><td>${k.alerts_issued}</td><td>${k.captured_onsets} / ${k.total_test_onsets}</td><td>${fmtPct(k.recall)}</td><td>${fmtPct(k.precision)}</td></tr>`).join("")}
      </tbody>
    </table>
    <p class="text-summary">A stricter cutoff (fewer alerts) trades recall for precision — useful for deciding how
      aggressive alerting should be operationally, independent of the fixed threshold used elsewhere in this dashboard.</p>
  `;
}

// ---------------------------------------------------------------------
// Executive Overview — a 30-second, presentation-ready summary that
// reuses the same computations already done for Interventions/Root Cause
// (see state.lastCandidates, cached by renderInterventions() each tick),
// plus a mode toggle that hides the detailed panels for a clean readout.
// ---------------------------------------------------------------------

function renderExecOverview() {
  const el = document.getElementById("execOverviewGrid");
  if (!el) return;
  const t = state.currentT;
  const cur = state.cityTS[t];
  const prev = t > 0 ? state.cityTS[t - 1] : null;
  const C = state.panel.num_cells;
  const degradedCount = cur.degraded_count + cur.severe_count;
  const degradedPct = 100 * degradedCount / C;
  const prevDegradedPct = prev ? 100 * (prev.degraded_count + prev.severe_count) / C : null;
  const profileTag = state.profile !== "balanced" ? ` (${state.profile})` : "";
  const candidates = state.lastCandidates || [];

  const top = candidates[0];
  const topArea = top ? cellName(top.h3) : "None currently";

  let driverLabel = "—";
  if (candidates.length) {
    const totals = { congestion: 0, coverage: 0, indoor: 0, backhaul: 0, insufficient: 0 };
    candidates.forEach(cand => {
      const pct = causeScoresForCell(t, cand.c, cand.h3);
      Object.keys(totals).forEach(k => totals[k] += pct[k]);
    });
    const bestKey = Object.entries(totals).sort((a, b) => b[1] - a[1])[0][0];
    driverLabel = CAUSE_LABELS[bestKey];
  }

  const aiLine = state.onset
    ? `${state.aiSummary?.degradedNowCount ?? "—"} degraded now`
    : "unavailable";

  const qoeDelta = deltaStr(cur.avg_qoe, prev?.avg_qoe, false);
  const degradedDelta = deltaStr(degradedPct, prevDegradedPct, true);
  const cards = [
    { label: `Network Health${profileTag}`, value: `${cur.avg_qoe?.toFixed(0) ?? "—"} / 100`, sub: qoeDelta || "&nbsp;" },
    { label: "Degraded Cells", value: `${degradedCount.toLocaleString()}`, sub: `${degradedPct.toFixed(1)}% of ${C.toLocaleString()}${degradedDelta ? " · " + degradedDelta : ""}` },
    { label: "AI Forecast (latest run)", value: aiLine, sub: state.onset ? `snapshot ${state.onset.snapshot_timestamp} UTC` : "no AI package loaded" },
    { label: "Top Problem Area", value: topArea, sub: top ? `QoE ${top.qoe?.toFixed(0)}` : "" },
    { label: "Primary Driver", value: driverLabel, sub: "of currently degraded cells" },
    { label: "Recommended Action", value: `Investigate ${candidates.length}`, sub: candidates.length ? "cells (see Intervention Priorities)" : "no action needed right now" },
  ];

  el.innerHTML = cards.map(c => `
    <div class="exec-card">
      <div class="exec-label">${c.label}</div>
      <div class="exec-value">${c.value}</div>
      <div class="exec-sub">${c.sub}</div>
    </div>
  `).join("");
}

function setViewMode(mode) {
  state.viewMode = mode;
  document.getElementById("viewModeOverviewBtn").classList.toggle("active", mode === "overview");
  document.getElementById("viewModeFullBtn").classList.toggle("active", mode === "full");
  document.querySelectorAll(".full-dashboard-only").forEach(elm => elm.classList.toggle("d-none", mode === "overview"));
  renderExecOverview();
}

function renderTrustPanel() {
  const s = state.summary;
  document.getElementById("trustDisclaimer").textContent = s.disclaimer;

  const sourceNames = {
    ookla: "Ookla Open Data", buildings: "Google Open Buildings",
    towers: "OpenCellID", osm: "OpenStreetMap", worldpop: "WorldPop",
    dosm_population: "OpenDOSM", ridership: "Transit Ridership",
  };
  document.getElementById("sourceBadges").innerHTML = Object.entries(s.source_datasets).map(([k, present]) =>
    `<span class="source-badge${present ? "" : " partial"}">${present ? "✓" : "✗"} ${sourceNames[k] || k}</span>`
  ).join("") + Object.entries(s.synthetic_pct_by_source).map(([k, pct]) =>
    `<span class="source-badge${pct > 0 ? " partial" : ""}">${k}: ${(100 - pct).toFixed(0)}% real</span>`
  ).join("");

  document.getElementById("anchoringStat").innerHTML = `
    <div class="kpi-value" style="font-size:22px">${s.ookla_anchored_pct}%</div>
    <div class="anchor-bar"><div class="anchor-bar-fill" style="width:${s.ookla_anchored_pct}%"></div></div>
    <div class="text-summary">of cells have their simulated hourly QoE re-centred on a real quarterly
    Ookla average. The remaining ${(100 - s.ookla_anchored_pct).toFixed(1)}% are simulated-only —
    per the pipeline's own validation notes, almost all "degraded" hours occur on anchored cells.</div>
  `;
}

// ---------------------------------------------------------------------
// Executive summary export — opens a print-friendly report in a new tab
// and triggers the browser's native print dialog (Save as PDF works in
// every major browser, no extra libraries). Deliberately does NOT try to
// screenshot the Leaflet map: cross-origin tile canvases are frequently
// blocked from capture, which would silently produce a blank map image —
// a real top-5-hotspot table with coordinates is more reliable in print
// anyway. Chart images use Plotly.toImage, which works offline since it
// rasterises the already-rendered chart, not a network tile.
// ---------------------------------------------------------------------

async function exportExecutiveSummary() {
  const t = state.currentT;
  const cur = state.cityTS[t];
  const C = state.panel.num_cells;
  const degradedPct = 100 * (cur.degraded_count + cur.severe_count) / C;

  let popAffected = 0;
  const rows = [];
  for (let c = 0; c < C; c++) {
    const h3 = state.panel.h3_ids[c];
    const cls = cellValue("degradation_class", t, c);
    if (cls >= 1) popAffected += state.gridPropsByH3.get(h3)?.population_day || 0;
    rows.push({ h3, qoe: cellValue("qoe", t, c), hotspot: cellValue("hotspot_code", t, c) });
  }
  const worst5 = [...rows].sort((a, b) => a.qoe - b.qoe).slice(0, 5);

  const rootCauseImg = await Plotly.toImage("chartRootCause", { format: "png", width: 500, height: 220 });
  const qoeImg = await Plotly.toImage("chartQoe", { format: "png", width: 500, height: 180 });

  const win = window.open("", "_blank");
  win.document.write(`
    <html><head><title>GeoQoE Pulse — Executive Summary</title>
    <style>
      body { font-family: -apple-system, Arial, sans-serif; color: #2d3748; padding: 32px; max-width: 800px; margin: auto; }
      h1 { color: #1a365d; font-size: 20px; margin-bottom: 0; }
      .sub { color: #64748b; font-size: 12px; margin-top: 4px; }
      .kpis { display: flex; gap: 16px; margin: 20px 0; }
      .kpi { border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; flex: 1; }
      .kpi .label { font-size: 10px; color: #64748b; text-transform: uppercase; }
      .kpi .value { font-size: 20px; font-weight: 700; color: #1a365d; }
      table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0; }
      th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #e2e8f0; }
      img { max-width: 100%; margin: 10px 0; }
      .disclaimer { font-size: 10px; color: #94a3b8; margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 10px; }
      @media print { button { display: none; } }
    </style></head><body>
      <h1>GeoQoE Pulse — Executive Summary</h1>
      <div class="sub">${state.summary.study_area} &middot; Hour: ${new Date(cur.timestamp).toUTCString().replace(" GMT", " UTC")}</div>

      <div class="kpis">
        <div class="kpi"><div class="label">Overall QoE</div><div class="value">${cur.avg_qoe?.toFixed(0)}</div></div>
        <div class="kpi"><div class="label">Degraded Cells</div><div class="value">${degradedPct.toFixed(1)}%</div></div>
        <div class="kpi"><div class="label">Population Affected</div><div class="value">${Math.round(popAffected).toLocaleString()}</div></div>
        <div class="kpi"><div class="label">Total Cells</div><div class="value">${C.toLocaleString()}</div></div>
      </div>

      <h3>Top 5 worst-QoE cells this hour</h3>
      <table>
        <tr><th>Location</th><th>QoE</th><th>Hotspot class</th></tr>
        ${worst5.map(r => `<tr><td>${cellName(r.h3)}</td><td>${r.qoe?.toFixed(0)}</td><td>${state.panel.hotspot_code_labels[r.hotspot]}</td></tr>`).join("")}
      </table>

      <h3>QoE trend (selected range)</h3>
      <img src="${qoeImg}">

      <h3>Root cause breakdown (heuristic, degraded cells)</h3>
      <img src="${rootCauseImg}">

      ${state.onset ? `
      <h3>Trained AI model — latest snapshot</h3>
      <p>Onset model <code>${state.onset.model_name}</code> (frozen threshold ${state.onset.model_frozen_threshold.toFixed(3)}) scored at
      ${state.onset.snapshot_timestamp} UTC. Static spatial baseline: ${state.onset.static_observed_cells.toLocaleString()} cells with real
      Ookla labels, ${state.onset.static_ai_estimated_cells.toLocaleString()} AI-estimated. Onset risk scores are relative rankings within a
      simulated scenario, not calibrated probabilities — see the AI Model Info panel for validated precision/recall.</p>` : ""}

      <div class="disclaimer">${state.summary.disclaimer}<br/><br/>
        Root cause percentages are a client-side heuristic computed from real static features
        (tower distance, built-up ratio, latency vs. demand, sample confidence) — the pipeline
        itself exports no "likely cause" column. Generated ${new Date().toISOString()}.</div>
      <button onclick="window.print()" style="margin-top:16px;padding:8px 16px;">Print / Save as PDF</button>
    </body></html>
  `);
  win.document.close();
}

function renderAll() {
  renderMapColors();
  renderLegend();
  updateChartMarkers();
  renderKpis();
  renderHotspotLists();
  renderRootCause();
  renderInterventions();
  if (state.onset) renderAiTimeBanner();
  renderExecOverview();
  if (state.compareMode) renderComparePanel();
}

function togglePlay() {
  const btn = document.getElementById("playBtn");
  if (state.playing) {
    clearInterval(state.playTimer);
    state.playing = false;
    btn.textContent = "▶";
    return;
  }
  state.playing = true;
  btn.textContent = "⏸";
  state.playTimer = setInterval(() => {
    state.currentT = (state.currentT + 1) % state.panel.num_timestamps;
    document.getElementById("timeSlider").value = state.currentT;
    renderAll();
    if (state.currentT === state.panel.num_timestamps - 1) togglePlay();
  }, 700);
}

document.addEventListener("DOMContentLoaded", () => {
  boot().then(renderTrustPanel).catch(err => {
    console.error(err);
    document.body.insertAdjacentHTML("afterbegin",
      `<div style="background:#fee2e2;color:#7f1d1d;padding:12px;text-align:center">
        Failed to load dashboard data: ${err.message}. Make sure you're serving this
        folder over HTTP (e.g. <code>python3 -m http.server</code>) and that
        <code>data/</code> was generated by build_dashboard_data.py.
      </div>`);
  });
});
