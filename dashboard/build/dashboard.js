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
  quarterlyObserved: "./data/quarterly_observed_2026Q1.geojson",
  finals: "./data/finals_quarterly.geojson",
  finalsMetadata: "./data/finals_metadata.json",
};

const state = {
  grid: null,
  panel: null,
  cityTS: null,
  summary: null,
  onset: null,      // adapter for the authoritative quarterly finals forecast
  modelCard: null,
  quarterlyObserved: null,
  finals: null,      // authoritative 2026Q1 → 2026Q2 quarterly emerging-risk package
  finalsActualRevealed: false,
  timeMachineOutcomeFilter: "all",
  onsetSnapshotT: null,      // panel.json hour-index matching the AI snapshot, or -1/null if none
  cityStats: null,           // city-wide mean/std of rtt/jitter/loss/demand, for per-cell AI explanations
  aiSummary: null,           // cached { degradedNowCount, topRiskCell } from renderAiPanel()
  lastCandidates: [],        // cached from renderInterventions(), reused by renderExecOverview()
  selectedH3: null,
  insightTab: "overview",
  appPage: "command",
  analystTab: "map",
  workspaceTool: "overview",
  analysisContext: "quarterly",
  viewMode: "command",       // command is the decision-first default; analyst retains every control
  h3IndexOf: new Map(),     // h3_index -> cell index (0..C-1)
  gridPropsByH3: new Map(), // h3_index -> geojson feature properties
  currentT: 0,              // current timestamp index (map/KPIs)
  rangeStart: 0,            // trend-range selection (charts/range summary)
  rangeEnd: 0,
  granularity: "hourly",    // "hourly" | "daily"
  confidenceThreshold: 0,   // 0-1, filters ranked lists & interventions only
  interventionPriorityFilter: "all",
  interventionStatusFilter: "all",
  interventionEvidenceFilter: "all",
  quarterlyPriorityView: "future",
  quarterlyTriageFilter: "all",
  quarterlyMapView: "combined",
  aiForecastBandFilter: "high",
  aiForecastExpanded: false,
  expandedRankedLists: new Map(),
  colorBy: "onset",
  basemap: "labels",
  compareMode: false,
  compareCells: [],         // array of h3_index, max COMPARE_MAX
  tracking: {},             // h3_index -> "planned" | "done", persisted to localStorage
  investigationCases: {},  // h3_index -> locally stored quarterly investigation handoff
  profile: "balanced",      // QoE weighting profile — see PROFILE_WEIGHTS
  whatIfEnabled: false,
  whatIfPct: 20,            // -50..+100, % capacity change
  whatIfParent: null,
  weatherEnabled: false,
  weatherLayer: null,
  darkMode: false,
  map: null,
  geoLayer: null,
  activeHoverLayer: null,
  activePopup: null,
  towerLayer: null,
  tileLayers: {},
  playing: false,
  playTimer: null,
  emergingH3s: new Set(),
  emergingReady: false,
};
const COMPARE_MAX = 5;
const TRACKING_KEY = "geoqoe_pulse_tracking_v1";
const INVESTIGATION_CASES_KEY = "geoqoe_pulse_investigation_cases_v1";
const DARKMODE_KEY = "geoqoe_pulse_darkmode_v1";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

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
  none: "~RM 2k–5k (targeted field survey)",
};
// No real operator-per-cell mapping exists in this dataset. Fill this in if
// you have one (e.g. from a licence/coverage-obligation dataset); otherwise
// the UI falls back to a generic escalation line.
const OPERATOR_CONTACTS = {};

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

async function boot() {
  const [grid, panel, cityTS, summary, quarterlyObserved, finals, finalsMetadata] = await Promise.all([
    fetch(DATA_URLS.grid).then(r => r.json()),
    fetch(DATA_URLS.panel).then(r => r.json()),
    fetch(DATA_URLS.cityTimeseries).then(r => r.json()),
    fetch(DATA_URLS.summary).then(r => r.json()),
    fetch(DATA_URLS.quarterlyObserved).then(r => r.json()),
    fetch(DATA_URLS.finals).then(r => r.ok ? r.json() : null).catch(() => null),
    fetch(DATA_URLS.finalsMetadata).then(r => r.ok ? r.json() : null).catch(() => null),
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

  if (finals?.features?.length) {
    state.quarterlyObserved = quarterlyObserved;
    state.finals = { geojson: finals, metadata: finalsMetadata || {} };
    state.onset = buildFinalsAiAdapter(finals, quarterlyObserved, finalsMetadata || {}, panel);
    state.onsetSnapshotT = null;
  } else {
    throw new Error("Finals quarterly forecast package is missing or empty. Expected data/finals_quarterly.geojson.");
  }
  state.modelCard = null;

  state.grid = grid;
  state.panel = panel;
  state.cityTS = cityTS;
  state.summary = summary;
  state.currentT = panel.num_timestamps - 1; // default to latest hour
  state.rangeStart = 0;
  state.rangeEnd = panel.num_timestamps - 1;
  loadTracking();
  loadInvestigationCases();

  panel.h3_ids.forEach((h3, i) => state.h3IndexOf.set(h3, i));
  grid.features.forEach(f => state.gridPropsByH3.set(f.properties.h3_index, f.properties));
  state.cityStats = computeCityStats(state.onsetSnapshotT ?? panel.num_timestamps - 1);

  document.getElementById("regionBadge").textContent =
    "Klang Valley, Malaysia";

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
    syncPrimaryMapControls();
    renderMapColors();
    renderLegend();
  });
  document.querySelectorAll(".map-mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mapMode;
      if (mode === "onset" && !state.onset) return;
      state.colorBy = mode;
      document.getElementById("colorBySelect").value = mode;
      syncPrimaryMapControls();
      renderMapColors();
      renderLegend();
    });
  });
  document.querySelectorAll(".quarterly-map-view-btn").forEach(btn => {
    btn.addEventListener("click", () => setQuarterlyMapView(btn.dataset.quarterlyView));
  });
  document.querySelectorAll(".insight-tab[data-insight-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      setInsightTab(btn.dataset.insightTab);
    });
  });
  document.getElementById("toggleTowers").addEventListener("change", e => {
    state.towerLayer.eachLayer(marker => marker.setStyle({
      opacity: e.target.checked ? 1 : 0,
      fillOpacity: e.target.checked ? 0.6 : 0,
    }));
  });
  document.getElementById("basemapSelect").addEventListener("change", e => {
    setBasemap(e.target.value);
  });
  document.getElementById("compareToggleBtn")?.addEventListener("click", toggleCompareMode);
  document.getElementById("confidenceFilter").addEventListener("input", e => {
    state.confidenceThreshold = parseInt(e.target.value, 10) / 100;
    document.getElementById("confidenceFilterLabel").textContent = e.target.value + "%";
    renderHotspotLists();
    renderInterventions();
    renderPriorityActionCentre();
    renderExecOverview();
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
  document.getElementById("guidedDemoBtn").addEventListener("click", advanceGuidedDemo);
  document.getElementById("mapFullscreenBtn")?.addEventListener("click", toggleMapFullscreen);
  document.addEventListener("fullscreenchange", syncMapFullscreenState);
  document.getElementById("sidebarToggle")?.addEventListener("click", () => document.body.classList.toggle("sidebar-collapsed"));
  document.querySelectorAll("[data-page]").forEach(btn => btn.addEventListener("click", () => setAppPage(btn.dataset.page)));
  document.querySelectorAll(".ai-workspace-tab").forEach(btn => btn.addEventListener("click", () => setAiWorkspaceTab(btn.dataset.aiTab)));
  document.querySelectorAll(".analyst-tab").forEach(btn => btn.addEventListener("click", () => setAnalystTab(btn.dataset.analystTab)));

  const actionsHost = document.querySelector(".full-width-row");
  if (actionsHost) {
    actionsHost.classList.add("workspace-actions-host");
    document.getElementById("mainWorkspace")?.appendChild(actionsHost);
  }
  document.querySelectorAll(".workspace-overview-panel, .workspace-more-panel, .priority-action-panel, .compare-panel, .hotspot-panel, .rootcause-panel, .trust-panel, .timeseries-panel, .ai-panel, .selected-area-panel, .workspace-actions-host").forEach(el => el.classList.add("workspace-analysis-panel"));
  document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      bootstrap.Modal.getOrCreateInstance(document.getElementById("commandPaletteModal")).show();
      window.setTimeout(() => document.getElementById("commandPaletteSearch")?.focus(), 120);
    }
  });

  initAiPanel();
  initPlaceSearch();
  initLayerSearchPlacement();

  buildMap();
  if (window.ResizeObserver) {
    new ResizeObserver(() => state.map?.invalidateSize()).observe(document.querySelector(".map-panel"));
  }
  initWorkspaceResize();
  buildCharts();
  setAppPage("command");
  setWorkspaceTool("overview");
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

// Adapts the finals quarterly GeoJSON to the map's existing cell-index lookup.
// This is a field mapping only: no risk score, rank, or SHAP value is recomputed
// in the browser. Missing cells remain explicitly unscored.
function buildFinalsAiAdapter(finalsGeojson, observedGeojson, metadata, panel) {
  const byH3 = new Map(finalsGeojson.features.map(feature => [feature.properties?.h3_index, feature.properties || {}]));
  const observedByH3 = new Map((observedGeojson?.features || []).map(feature => [feature.properties?.h3_index, feature.properties || {}]));
  const names = [
    "onset_risk_rank", "onset_priority_tier", "next_hour_onset_risk_score", "onset_operational_status",
    "alert_at_frozen_threshold", "risk_band_label", "risk_percentile", "measurement_support", "ookla_tests",
    "display_dl_mbps", "ookla_dl_mbps", "ookla_ul_mbps", "ookla_rtt_ms", "dl_percentile_in_quarter",
    "current_status", "actual_next_quarter_status", "forecast_for_quarter", "static_data_status",
    "relative_prediction_reliability", "time_machine_result", "score_is_probability", "interpretation_note",
    "population_density_per_km2", "population_day", "district_population", "viirs_rad_quarter",
    "viirs_rad_yoy_log_ratio", "tower_density_per_km2", "dist_to_nearest_tower_m",
    "building_density_per_km2", "built_up_ratio", "landcover_built_up_pct",
    "road_density_km_per_km2", "intersection_density_per_km2", "dist_to_nearest_transit_m",
    "risk_driver_1_label", "risk_driver_1_value", "risk_driver_1_family", "risk_driver_2_label", "risk_driver_2_value", "risk_driver_2_family", "risk_driver_3_label", "risk_driver_3_value", "risk_driver_3_family",
    "protective_1_label", "protective_1_value", "protective_1_family", "protective_2_label", "protective_2_value", "protective_2_family",
  ];
  const fields = Object.fromEntries(names.map(name => [name, Array(panel.num_cells).fill(null)]));
  panel.h3_ids.forEach((h3, index) => {
    const row = byH3.get(h3);
    const observed = observedByH3.get(h3);
    fields.current_status[index] = row?.current_status ?? observed?.current_status ?? "not_measured_2026Q1";
    ["ookla_dl_mbps", "ookla_ul_mbps", "ookla_rtt_ms", "ookla_tests", "dl_percentile_in_quarter",
      "population_density_per_km2", "population_day", "district_population", "viirs_rad_quarter",
      "viirs_rad_yoy_log_ratio", "tower_density_per_km2", "dist_to_nearest_tower_m",
      "building_density_per_km2", "built_up_ratio", "landcover_built_up_pct",
      "road_density_km_per_km2", "intersection_density_per_km2", "dist_to_nearest_transit_m"
    ].forEach(name => { fields[name][index] = row?.[name] ?? observed?.[name] ?? null; });
    fields.display_dl_mbps[index] = fields.ookla_dl_mbps[index];
    fields.static_data_status[index] = fields.ookla_dl_mbps[index] != null ? "observed_ookla" : "not_available";
    if (!row) {
      fields.onset_priority_tier[index] = "not_scored";
      fields.onset_operational_status[index] = fields.current_status[index] === "currently_underperforming" ? "observed_current_issue" : "not_forecast_eligible";
      return;
    }
    fields.onset_risk_rank[index] = row.risk_rank ?? null;
    fields.onset_priority_tier[index] = row.risk_band ?? "not_scored";
    fields.next_hour_onset_risk_score[index] = row.xgb_full ?? null;
    fields.onset_operational_status[index] = "finals_scored";
    fields.alert_at_frozen_threshold[index] = (row.risk_rank ?? Infinity) <= 200;
    fields.risk_band_label[index] = row.risk_band_label ?? null;
    fields.risk_percentile[index] = row.risk_percentile ?? null;
    fields.measurement_support[index] = row.measurement_support ?? null;
    fields.relative_prediction_reliability[index] = row.measurement_support ?? null;
    fields.time_machine_result[index] = row.time_machine_result ?? null;
    ["actual_next_quarter_status", "forecast_for_quarter", "score_is_probability", "interpretation_note"].forEach(name => { fields[name][index] = row[name] ?? null; });
    [1, 2, 3].forEach(n => ["label", "value", "family"].forEach(part => { fields[`risk_driver_${n}_${part}`][index] = row[`risk_driver_${n}_${part}`] ?? null; }));
    [1, 2].forEach(n => ["label", "value", "family"].forEach(part => { fields[`protective_${n}_${part}`][index] = row[`protective_${n}_${part}`] ?? null; }));
  });
  return {
    snapshot_timestamp: `${metadata.predictor_quarter || "2026Q1"} → ${metadata.forecast_quarter || "2026Q2"}`,
    forecast_window: `${metadata.predictor_quarter || "2026Q1"} → ${metadata.forecast_quarter || "2026Q2"}`,
    num_cells: panel.num_cells,
    h3_ids: panel.h3_ids,
    model_name: metadata.model_version || "Unknown finals model",
    fields,
  };
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
  demand_pressure: { label: "Scenario Pressure", unit: "", direction: "high" },
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
  const likelyTrigger = cause !== "none"
    ? `${CAUSE_LABELS[cause]} is the highest-percentile triggered investigation signal${triggerParts.length ? `, alongside ${triggerParts.join(" and ")}` : ""}.`
    : "No investigation signal reaches the citywide top quartile for this scenario hour.";

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
function loadInvestigationCases() {
  try {
    const saved = JSON.parse(localStorage.getItem(INVESTIGATION_CASES_KEY) || "{}");
    state.investigationCases = saved && typeof saved === "object" ? saved : {};
  } catch { state.investigationCases = {}; }
}
function saveInvestigationCases() {
  try { localStorage.setItem(INVESTIGATION_CASES_KEY, JSON.stringify(state.investigationCases)); } catch {}
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
  const toggle = document.getElementById("darkModeToggle");
  toggle.innerHTML = `<i class="bi ${state.darkMode ? "bi-sun" : "bi-moon"}" aria-hidden="true"></i>`;
  toggle.setAttribute("aria-pressed", String(state.darkMode));
  toggle.setAttribute("aria-label", state.darkMode ? "Switch to light theme" : "Switch to dark theme");
  syncOperatorChartTheme();
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
  high: "#c43d54",
  elevated: "#ea8b2c",
  lower: "#8aa3b5",
  not_scored: "#cbd5e1",
};
const ONSET_TIER_LABELS = {
  high: "High emerging risk",
  elevated: "Elevated emerging risk",
  lower: "Lower emerging risk",
  not_scored: "Not scored",
};
const TIME_MACHINE_COLORS = {
  correctly_anticipated: "#2f855a",
  false_alert: "#7447a8",
  missed: "#2563a6",
  correctly_lower_risk: "#64748b",
};
function onsetTierColor(c) {
  if (!state.onset) return "#cbd5e1";
  const currentStatus = onsetValue("current_status", c);
  if (state.quarterlyMapView === "observed") {
    if (currentStatus === "currently_underperforming") return "#7f1d1d";
    if (currentStatus === "currently_not_underperforming") return "#dfe9e5";
    return "#e2e8f0";
  }
  if (state.quarterlyMapView === "forecast") {
    if (currentStatus === "currently_underperforming") return "#e2e8f0";
    return ONSET_TIER_COLORS[onsetValue("onset_priority_tier", c)] || "#cbd5e1";
  }
  if (state.finalsActualRevealed) {
    const result = onsetValue("time_machine_result", c);
    if (state.timeMachineOutcomeFilter !== "all" && result !== state.timeMachineOutcomeFilter) return "#e2e8f0";
    return TIME_MACHINE_COLORS[result] ||
      (onsetValue("current_status", c) === "currently_underperforming" ? "#7f1d1d" : "#e2e8f0");
  }
  if (onsetValue("current_status", c) === "currently_underperforming") return "#7f1d1d";
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
const AI_FLAGGED_TIERS = ["high", "elevated"];
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
  const map = L.map("map", { zoomControl: true, closePopupOnClick: true }).setView([3.05, 101.65], 11);

  state.tileLayers.labels = L.tileLayer("https://tile.openstreetmap.de/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors", maxZoom: 18,
  });
  state.tileLayers.light = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
    attribution: "Tiles &copy; Esri", maxZoom: 16,
  });
  state.tileLayers[state.basemap].addTo(map);

  const geoLayer = L.geoJSON(state.grid, {
    style: () => ({ color: "#334155", weight: 0.5, fillOpacity: 0.75, fillColor: "#cbd5e1" }),
    onEachFeature: (feature, layer) => {
      const h3 = feature.properties.h3_index;
      layer.bindTooltip(() => hoverHtml(h3), { sticky: true, direction: "top", permanent: false });
      layer.on("mouseover", () => {
        if (state.activeHoverLayer && state.activeHoverLayer !== layer) state.activeHoverLayer.closeTooltip();
        state.activeHoverLayer = layer;
        const selected = h3 === state.selectedH3 || state.compareCells.includes(h3);
        layer.setStyle({ weight: selected ? 3.4 : 2, opacity: 1, fillOpacity: selected ? 0.95 : 0.88 });
        layer.bringToFront();
      });
      layer.on("mouseout", () => {
        layer.closeTooltip();
        if (state.activeHoverLayer === layer) state.activeHoverLayer = null;
        renderMapColors();
      });
      layer.on("click", () => {
        if (state.activeHoverLayer) state.activeHoverLayer.closeTooltip();
        state.activeHoverLayer = null;
        map.closePopup();
        if (state.compareMode) toggleCompareCell(h3);
        else selectArea(h3);
      });
    },
  }).addTo(map);

  const towerLayer = L.layerGroup(
    state.grid.features
      .filter(f => f.properties.tower_count > 0)
      .map(f => L.circleMarker([f.properties.centroid_lat, f.properties.centroid_lon], {
        radius: Math.min(2 + Math.sqrt(f.properties.tower_count), 12),
        color: "#1a365d", weight: 1, fillColor: "#2b6cb0", fillOpacity: 0, opacity: 0,
      }).bindTooltip(`${f.properties.tower_count} tower site(s)`))
  ).addTo(map);

  state.map = map;
  state.geoLayer = geoLayer;
  map.on("zoomend", () => { if (state.panel) renderMapColors(); });
  map.on("popupopen", event => {
    if (state.activePopup && state.activePopup !== event.popup) map.closePopup(state.activePopup);
    state.activePopup = event.popup;
  });
  map.on("popupclose", event => {
    if (state.activePopup === event.popup) state.activePopup = null;
  });
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
    const rank = onsetValue("onset_risk_rank", c);
    if (state.finals) {
      const current = onsetValue("current_status", c);
      if (state.quarterlyMapView === "observed") {
        return `<b>2026Q1 observed status:</b> ${current === "currently_underperforming" ? "Underperforming" : current === "currently_not_underperforming" ? "Not underperforming" : "Not measured"}`;
      }
      if (state.quarterlyMapView === "forecast") {
        if (current === "currently_underperforming") return `<b>2026Q2 AI forecast:</b> Not evaluated because underperformance was already observed in 2026Q1`;
        return `<b>2026Q2 AI priority:</b> ${ONSET_TIER_LABELS[tier] ?? "Not evaluated"}${rank != null ? ` · #${rank} of ${state.finals.metadata.eligible_cells?.toLocaleString?.() ?? "—"}` : ""}`;
      }
      if (state.finalsActualRevealed && onsetValue("time_machine_result", c)) {
        const result = onsetValue("time_machine_result", c);
        const actual = onsetValue("actual_next_quarter_status", c) === "underperforming" ? "Underperforming" : "Not underperforming";
        const support = onsetValue("measurement_support", c);
        const tests = onsetValue("ookla_tests", c);
        return `<b>Held-out validation:</b> ${timeMachineLabel(result)}<br/>
                <b>2026Q1 input:</b> Not underperforming<br/>
                <b>AI shortlist:</b> ${rank != null && rank <= 200 ? `Top 200 · rank #${rank}` : rank != null ? `Outside Top 200 · rank #${rank}` : "Not ranked"}<br/>
                <b>2026Q2 observed result:</b> ${actual}<br/>
                <b>Measurement support:</b> ${support ? `${support.charAt(0).toUpperCase()}${support.slice(1)}` : "Not available"}${tests != null ? ` · ${tests} tests` : ""}`;
      }
      if (current === "currently_underperforming") return `<b>2026Q1 observed:</b> Underperforming<br/><b>AI forecast:</b> Not evaluated because the issue already existed`;
      return `<b>2026Q1 observed:</b> ${current === "currently_not_underperforming" ? "Not underperforming" : "Not measured"}<br/><b>2026Q2 priority:</b> ${ONSET_TIER_LABELS[tier] ?? "Not eligible"}${rank != null ? ` · #${rank} of ${state.finals.metadata.eligible_cells?.toLocaleString?.() ?? "—"}` : ""}`;
    }
    const score = onsetValue("next_hour_onset_risk_score", c);
    return `<b>AI onset priority:</b> ${ONSET_TIER_LABELS[tier] ?? tier ?? "—"}<br/>
            <b>Relative risk score:</b> ${score != null ? score.toFixed(2) : "—"} (not a probability)`;
  }
  if (state.onset && state.colorBy === "static_ai") {
    const status = onsetValue("static_data_status", c);
    const v = onsetValue("display_dl_mbps", c);
    return `<b>${status === "observed_ookla" ? "Observed" : "AI-estimated"} DL speed:</b> ${v != null ? v.toFixed(1) : "—"} Mbps`;
  }

  const primary = { qoe: "QoE", demand: "Scenario Pressure", hotspot: "Hotspot", moran: "Moran", risk: "Risk" }[state.colorBy];
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
  const realQoe = cellValue("qoe", t, c);
  const eff = effectiveMetrics(t, c);
  const showEff = state.profile !== "balanced" || state.whatIfEnabled;
  const sq = cellValue("degradation_class", t, c);
  const classLabel = ["Normal", "Degraded", "Severe"][sq] ?? "—";
  const qoeText = realQoe != null ? realQoe.toFixed(0) : "—";
  const demandText = eff.demand != null ? eff.demand.toFixed(0) : "—";
  const rtt = cellValue("rtt_ms", t, c);
  const qoeTone = realQoe == null ? "metric-muted" : realQoe < 50 ? "metric-critical" : realQoe < 70 ? "metric-watch" : "metric-good";
  const demandTone = eff.demand == null ? "metric-muted" : eff.demand >= 15 ? "metric-critical" : eff.demand >= 11 ? "metric-watch" : "metric-good";
  const rttTone = rtt == null ? "metric-muted" : rtt >= 70 ? "metric-critical" : rtt >= 45 ? "metric-watch" : "metric-good";
  return `
    <div class="cell-popup">
      <div class="cell-popup-header"><strong>${cellName(h3)}</strong><span class="cell-status status-${classLabel.toLowerCase()}">${classLabel}</span></div>
      <div class="cell-popup-metrics"><span class="${qoeTone}"><b>${qoeText}</b>QoE</span><span class="${demandTone}"><b>${demandText}</b>Scenario Pressure</span><span class="${rttTone}"><b>${rtt?.toFixed?.(0) ?? "—"}ms</b>RTT</span></div>
      <div class="cell-popup-note">Quick map context · full quarterly status, model contributors and investigation guidance are shown in Cell Intelligence.</div>
      ${showEff ? `<div class="cell-popup-note">Active profile: ${state.profile}${state.whatIfEnabled ? " · what-if capacity active" : ""} · effective QoE ${eff.qoe?.toFixed?.(0) ?? "—"}</div>` : ""}
    </div>
  `;
}

// Trained AI model block appended to every cell popup, regardless of the
// active color-by mode — a fixed snapshot from the frozen onset/static
// models (see ai_model/), not tied to the time slider.
function aiPopupSection(c) {
  if (!state.onset) return "";
  if (state.finals) {
    const tier = onsetValue("onset_priority_tier", c);
    const rank = onsetValue("onset_risk_rank", c);
    const support = onsetValue("measurement_support", c);
    const tests = onsetValue("ookla_tests", c);
    const drivers = [1, 2, 3].map(n => ({ label: onsetValue(`risk_driver_${n}_label`, c), value: onsetValue(`risk_driver_${n}_value`, c), family: onsetValue(`risk_driver_${n}_family`, c) })).filter(row => row.label != null);
    return `<hr style="margin:6px 0"/><span class="cheat-badge ai" style="font-size:9px">AI MODEL · QUARTERLY</span><br/>
      Emerging risk: <b>${ONSET_TIER_LABELS[tier] ?? "Not eligible"}</b>${rank != null ? ` · Rank <b>#${rank}</b> of ${state.finals.metadata.eligible_cells?.toLocaleString?.() ?? "—"}` : ""}<br/>
      Quarterly measurement support: <b>${support ?? "Unavailable"}</b>${tests != null ? ` · ${tests} Ookla tests` : ""}
      ${drivers.length ? `<div class="ai-explain-block"><div class="ai-explain-title">Top model contributors</div>${drivers.map(row => `<div class="explain-signal"><b>${row.label}</b><span>Observed value: ${formatModelObservedValue(row.label, row.value)}</span><em>↑ Higher risk</em></div>`).join("")}</div>` : ""}
      <span style="font-size:10px;color:#64748b">These factors are contributors to the model's risk assessment. They influenced its ranking and do not establish engineering cause. The final model uses 2026Q1 context to forecast emerging underperformance in 2026Q2.</span>`;
  }
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
      <div class="ai-explain-trigger">Highest relative signal: ${exp.likelyTrigger}</div>
    </div>
  `;
}

function renderMapColors() {
  const t = state.currentT;
  let peakDemand = 0;
  for (let c = 0; c < state.panel.num_cells; c++) peakDemand = Math.max(peakDemand, cellValue("demand_pressure", t, c) ?? 0);
  state.geoLayer.eachLayer(layer => {
    const h3 = layer.feature.properties.h3_index;
    const c = state.h3IndexOf.get(h3);
    const isSelected = h3 === state.selectedH3 || state.compareCells.includes(h3);
    const hasSelection = Boolean(state.selectedH3) || state.compareCells.length > 0;
    const timeMachineResult = state.onset ? onsetValue("time_machine_result", c) : null;
    const filteredOut = state.colorBy === "onset" && state.finalsActualRevealed && state.timeMachineOutcomeFilter !== "all" && timeMachineResult !== state.timeMachineOutcomeFilter;
    const style = { fillColor: colorForMode(state.colorBy, t, c), color: isSelected ? "#c4b5fd" : "#334155", weight: isSelected ? 3.4 : (state.map.getZoom() <= 11 ? 0.25 : 0.5), opacity: isSelected ? 1 : (filteredOut ? 0.12 : (state.map.getZoom() <= 11 ? 0.3 : 0.65)), dashArray: null, fillOpacity: filteredOut ? 0.08 : (hasSelection && !isSelected ? 0.34 : (isSelected ? 0.95 : (state.map.getZoom() <= 11 ? 0.55 : 0.7))) };
    if (state.onset && state.colorBy === "static_ai" && onsetValue("static_data_status", c) === "ai_estimated") {
      // Dashed outline flags AI-ESTIMATED cells vs. solid outline for observed_ookla —
      // never let the color alone imply a real measurement where there isn't one.
      style.dashArray = "2,2";
      style.color = "#805ad5";
      style.weight = 0.8;
    }
    if (state.onset && state.colorBy === "onset" && state.quarterlyMapView !== "observed" && onsetValue("alert_at_frozen_threshold", c)) {
      style.color = "#742a2a";
      style.weight = 1.4;
    }
    const prominentValidationResult = ["correctly_anticipated", "false_alert", "missed"].includes(timeMachineResult) ||
      (state.timeMachineOutcomeFilter === "correctly_lower_risk" && timeMachineResult === "correctly_lower_risk");
    if (state.colorBy === "onset" && state.finalsActualRevealed && prominentValidationResult && !filteredOut) {
      style.color = TIME_MACHINE_COLORS[timeMachineResult];
      style.weight = state.map.getZoom() <= 11 ? 1.35 : 1.9;
      style.opacity = 1;
      style.fillOpacity = hasSelection && !isSelected ? 0.52 : 0.84;
    } else if (state.colorBy === "onset" && state.finalsActualRevealed && timeMachineResult === "correctly_lower_risk" && state.timeMachineOutcomeFilter === "all") {
      style.color = TIME_MACHINE_COLORS.correctly_lower_risk;
      style.weight = 0.2;
      style.opacity = 0.18;
      style.fillOpacity = hasSelection && !isSelected ? 0.08 : 0.2;
    }
    if (isSelected) Object.assign(style, { color: "#c4b5fd", weight: 3.4, opacity: 1, fillOpacity: 0.95 });
    layer.setStyle(style);
    const path = layer.getElement();
    if (!path) return;
    path.classList.remove("map-glow-red", "map-glow-warm", "map-glow-violet", "map-glow-orange", "map-selected-cell", "map-active-cell");
    if (isSelected) path.classList.add("map-selected-cell");
    if (h3 === state.selectedH3) path.classList.add("map-active-cell");
    const qoe = cellValue("qoe", t, c) ?? 100;
    const demand = cellValue("demand_pressure", t, c) ?? 0;
    const hotspot = cellValue("hotspot_code", t, c) ?? 0;
    const emerging = t >= 2 && qoe < (cellValue("qoe", t - 2, c) ?? qoe) * .8 && hotspot < 5;
    let glowClass = "";
    if (state.colorBy === "qoe" && qoe <= 45) glowClass = "map-glow-red";
    if (state.colorBy === "demand" && demand >= peakDemand - .25) glowClass = "map-glow-warm";
    if (state.colorBy === "hotspot" && (hotspot >= 5 || emerging)) glowClass = emerging ? "map-glow-orange" : "map-glow-red";
    if (state.colorBy === "onset" && state.quarterlyMapView !== "observed" && state.onset && (onsetValue("onset_risk_rank", c) ?? Infinity) <= 5) glowClass = "map-glow-violet";
    if (glowClass) path.classList.add(glowClass);
  });
}

function renderLegend() {
  const el = document.getElementById("mapLegend");
  el.classList.add("legend-changing");
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
    if (state.quarterlyMapView === "observed") {
      html = `<span><span class="swatch" style="background:#7f1d1d"></span>Underperforming · 2026Q1</span><span><span class="swatch" style="background:#dfe9e5"></span>Not underperforming · 2026Q1</span><span><span class="swatch" style="background:#e2e8f0"></span>Not measured</span><span class="legend-note"><b>Observed quarterly status only.</b></span>`;
    } else if (state.quarterlyMapView === "forecast") {
      html = Object.entries(ONSET_TIER_COLORS).map(([k, c]) => `<span><span class="swatch" style="background:${c}"></span>${k === "high" ? "High risk" : k === "elevated" ? "Elevated" : k === "lower" ? "Lower risk" : "Not evaluated"}</span>`).join("") + `<span class="legend-note"><b>2026Q2 AI priority among cells not underperforming in 2026Q1.</b> Ranks are not probabilities.</span>`;
    } else html = (state.finalsActualRevealed
      ? Object.entries(TIME_MACHINE_COLORS).map(([k, c]) => `<span><span class="swatch" style="background:${c}"></span>${timeMachineLabel(k)}</span>`).join("") + `<span><span class="swatch" style="background:#7f1d1d"></span>Already underperforming in 2026Q1</span>`
      : `<span><span class="swatch" style="background:#7f1d1d"></span>Observed issue · 2026Q1</span>` + Object.entries(ONSET_TIER_COLORS).map(([k, c]) =>
        `<span><span class="swatch" style="background:${c}"></span>${k === "high" ? "High risk" : k === "elevated" ? "Elevated" : k === "lower" ? "Lower risk" : "Not scored"}</span>`).join("")) +
      (state.finals
        ? `<span class="legend-note"><b>${state.finalsActualRevealed ? "2026Q2 actual outcome check" : "2026Q1 observed status + 2026Q2 AI priority"}.</b> Forecast ranks are not probabilities.</span>`
        : `<span class="legend-note">⚠️ <b>Score is a relative ranking/model-support indicator, NOT a calibrated probability.</b> Fixed AI model snapshot at ${state.onset?.snapshot_timestamp ?? "—"} UTC — independent of the time slider (see banner in the AI panel). Thick dark-red outline = above the frozen alert threshold.</span>`);
  } else if (state.colorBy === "static_ai") {
    html = `<span><span class="swatch" style="background:#e53e3e"></span>Low Mbps</span>
            <span><span class="swatch" style="background:#2f855a"></span>High Mbps</span>
            <span class="legend-note">Solid outline = real observed Ookla speed · Dashed purple outline = AI-estimated (no direct label)</span>`;
  } else if (state.colorBy === "comparison") {
    html = Object.entries(COMPARISON_COLORS).map(([k, c]) =>
      `<span><span class="swatch" style="background:${c}"></span>${COMPARISON_LABELS[k]}</span>`).join("") +
      `<span class="legend-note">Traditional = statistically significant Gi* latency hotspot (hot_90/95/99). Both methods compared at the AI snapshot hour (${state.onset?.snapshot_timestamp ?? "—"} UTC).</span>`;
  }
  window.requestAnimationFrame(() => {
    el.innerHTML = html;
    window.requestAnimationFrame(() => el.classList.remove("legend-changing"));
  });
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
    input.parentElement.classList.toggle("has-value", Boolean(e.target.value));
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
    input.value = "";
    input.parentElement.classList.remove("has-value");
    resultsEl.classList.add("d-none");
    resultsEl.innerHTML = "";
    input.focus();
  });
  resultsEl.addEventListener("click", e => {
    const item = e.target.closest(".result-item");
    if (item) goToSearchResult(item.dataset.h3);
  });
  document.addEventListener("click", e => {
    if (!e.target.closest(".map-search-control") && !e.target.closest(".place-search-results")) {
      resultsEl.classList.add("d-none");
    }
  });

  function goToSearchResult(h3) {
    const props = state.gridPropsByH3.get(h3);
    input.value = props?.place_name || h3;
    input.parentElement.classList.add("has-value");
    resultsEl.classList.add("d-none");
    flyToCell(h3);
  }
}

function initLayerSearchPlacement() {
  const menus = [...document.querySelectorAll(".map-control-menu")];
  menus.forEach(menu => menu.addEventListener("toggle", () => {
    if (menu.open) menus.forEach(other => { if (other !== menu) other.open = false; });
  }));
  document.addEventListener("click", event => {
    if (!event.target.closest(".map-control-menu")) menus.forEach(menu => { menu.open = false; });
  });
}

function flyToCell(h3) {
  const props = state.gridPropsByH3.get(h3);
  if (!props) return;
  state.map.flyTo([props.centroid_lat, props.centroid_lon], 14, { duration: 0.6 });
  selectArea(h3);
  playMapFlyRing(props);
}

function playMapFlyRing(props) {
  const ring = L.circleMarker([props.centroid_lat, props.centroid_lon], {
    radius: 13, color: "#c4b5fd", weight: 2, opacity: .95, fillOpacity: 0,
    interactive: false, className: "map-fly-ring",
  }).addTo(state.map);
  window.setTimeout(() => state.map?.removeLayer(ring), 850);
}

function selectArea(h3) {
  const preserveExplain = state.workspaceTool === "explain" ||
    (state.workspaceTool === "cell" && state.insightTab === "ai");
  if (state.activeHoverLayer) state.activeHoverLayer.closeTooltip();
  state.activeHoverLayer = null;
  state.map?.closePopup();
  state.selectedH3 = h3;
  state.insightTab = preserveExplain ? "ai" : "overview";
  renderMapColors();
  renderSelectedAreaInsights();
  syncWorkspaceSelection();
  syncMapSelectedBar();
  syncMapWorkspaceInsight();
  syncActiveToolSelection();
  if (!preserveExplain && state.workspaceTool !== "cell") setWorkspaceTool("cell");
  const panel = document.getElementById("selectedAreaPanel");
  document.getElementById("analysisStage")?.scrollTo({ top: 0, behavior: "instant" });
  if (window.gsap && panel && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    gsap.fromTo(panel, { autoAlpha: 0.72, x: 10 }, { autoAlpha: 1, x: 0, duration: .22, ease: "power2.out", overwrite: "auto" });
  }
}

function syncActiveToolSelection() {
  const h3 = state.selectedH3;
  const c = state.h3IndexOf.get(h3);
  if (c == null) return;
  const active = document.querySelector(".workspace-analysis-panel[style*='display: block'], .workspace-analysis-panel[style*='display:block']");
  if (!active || active.id === "workspaceOverviewPanel") return;
  // Explain already renders the selected cell's identity inside Cell Intelligence,
  // while Prioritise identifies every cell in its queue. A second page-level strip
  // repeats the same location/status and adds no decision value in either view.
  if (["selectedAreaPanel", "quarterlyPriorityPanel"].includes(active.id)) {
    active.querySelector(".active-selection-context")?.remove();
    return;
  }
  let context = active.querySelector(".active-selection-context");
  if (!context) {
    context = document.createElement("div");
    context.className = "active-selection-context";
    active.prepend(context);
  }
  const quarterly = state.analysisContext === "quarterly";
  const contextMetrics = quarterly
    ? `2026Q1 observed: ${onsetValue("current_status", c) === "currently_underperforming" ? "Underperforming" : "Not underperforming"} · 2026Q2 AI rank ${onsetValue("onset_risk_rank", c) != null ? `#${Math.round(onsetValue("onset_risk_rank", c))}` : "not evaluated"}`
    : `Scenario QoE ${cellValue("qoe", state.currentT, c)?.toFixed(0) ?? "—"} · Scenario pressure ${cellValue("demand_pressure", state.currentT, c)?.toFixed(0) ?? "—"} · ${new Date(state.cityTS[state.currentT].timestamp).toUTCString().slice(5, 22)}`;
  context.innerHTML = `<i class="bi bi-geo-alt-fill"></i><strong>${cellName(h3)}</strong><span>${contextMetrics}</span>`;
  context.classList.remove("selection-flash");
  void context.offsetWidth;
  context.classList.add("selection-flash");
}

function syncWorkspaceSelection() {
  const workspaceCell = document.getElementById("workspaceSelectedCell");
  const selectedBody = document.getElementById("selectedAreaBody");
  if (workspaceCell && selectedBody) {
    workspaceCell.innerHTML = `<div class="workspace-divider"></div><div class="workspace-panel-title">Selected area</div>${selectedBody.innerHTML}`;
    workspaceCell.classList.remove("selection-flash");
    void workspaceCell.offsetWidth;
    workspaceCell.classList.add("selection-flash");
  }
}

function syncMapSelectedBar() {
  const bar = document.getElementById("mapSelectedBar");
  if (!bar) return;
  const h3 = state.selectedH3;
  const c = state.h3IndexOf.get(h3);
  if (!h3 || c == null) { bar.classList.add("d-none"); return; }
  bar.classList.remove("d-none");
  bar.innerHTML = `<span class="selected-kicker"><i></i> Active cell</span><strong>${cellName(h3)}</strong><button onclick="clearSelectedArea()" aria-label="Clear selected location">×</button>`;
}

function syncMapWorkspaceInsight() {
  const el = document.getElementById("mapWorkspaceInsight");
  if (!el) return;
  const h3 = state.selectedH3;
  const c = state.h3IndexOf.get(h3);
  if (h3 && c != null) {
    const qoe = cellValue("qoe", state.currentT, c);
    const demand = cellValue("demand_pressure", state.currentT, c);
    const quarterly = state.analysisContext === "quarterly";
    const detail = quarterly
      ? `2026Q1 observed ${onsetValue("current_status", c) === "currently_underperforming" ? "underperformance" : "not underperforming"} · 2026Q2 AI rank ${onsetValue("onset_risk_rank", c) != null ? `#${Math.round(onsetValue("onset_risk_rank", c))}` : "not evaluated"}`
      : `Scenario QoE ${qoe?.toFixed(1) ?? "—"} · pressure ${demand?.toFixed(1) ?? "—"}`;
    el.innerHTML = `<i class="bi bi-hexagon-fill"></i><span><b>${cellName(h3)}</b> active</span><small>${detail}</small>`;
  } else {
    el.innerHTML = `<i class="bi bi-cursor"></i><span><b>Map ready</b></span><small>Click a hexagon to inspect its condition, forecast, and next action.</small>`;
  }
}

function clearSelectedArea() {
  state.selectedH3 = null;
  state.insightTab = "overview";
  renderMapColors();
  syncMapSelectedBar();
  syncMapWorkspaceInsight();
  renderSelectedAreaInsights();
  const workspaceCell = document.getElementById("workspaceSelectedCell");
  if (workspaceCell) workspaceCell.innerHTML = "";
  if (state.workspaceTool === "cell") setWorkspaceTool("overview");
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
  if (state.compareMode && state.workspaceTool !== "compare") setWorkspaceTool("compare");
  if (!state.compareMode && state.workspaceTool === "compare") setWorkspaceTool("overview");
}

function toggleCompareCell(h3) {
  const i = state.compareCells.indexOf(h3);
  if (i >= 0) {
    state.compareCells.splice(i, 1);
  } else {
    if (state.compareCells.length >= COMPARE_MAX) state.compareCells.shift();
    state.compareCells.push(h3);
  }
  state.selectedH3 = h3;
  renderMapColors();
  syncActiveToolSelection();
  renderComparePanel();
}

function toggleMoreTools(event) {
  event?.stopPropagation();
  const popover = document.getElementById("moreToolsPopover");
  popover?.classList.toggle("d-none");
}

function renderComparePanel() {
  const t = state.currentT;
  const tableEl = document.getElementById("compareTable");
  if (!state.compareCells.length) {
    tableEl.innerHTML = `<div class="compare-guidance"><strong>Select cells directly on the map</strong><span>0 / ${COMPARE_MAX} selected</span><p>With Compare Cells active, click hexagons on the live map. Choose two or more areas to reveal their QoE trends and current network metrics.</p><ol><li>Click a hexagon to add it.</li><li>Click it again to remove it.</li><li>Use the ✕ control to remove a row.</li></ol></div>`;
    Plotly.purge("chartCompare");
    return;
  }
  const rows = state.compareCells.map(h3 => {
    const c = state.h3IndexOf.get(h3);
    const props = state.gridPropsByH3.get(h3);
    return {
      h3, qoe: cellValue("qoe", t, c), demand: cellValue("demand_pressure", t, c),
      rtt: cellValue("rtt_ms", t, c), loss: cellValue("loss_pct", t, c),
      aiRank: state.onset ? onsetValue("onset_risk_rank", c) : null,
      conf: cellValue("confidence", t, c), pop: props.population_day, towers: props.tower_count,
    };
  });
  const bestQoe = [...rows].sort((a, b) => (b.qoe ?? -Infinity) - (a.qoe ?? -Infinity))[0];
  const lowestDemand = [...rows].sort((a, b) => (a.demand ?? Infinity) - (b.demand ?? Infinity))[0];
  const worstQoe = [...rows].sort((a, b) => (a.qoe ?? Infinity) - (b.qoe ?? Infinity))[0];
  const qoeGap = bestQoe?.qoe != null && worstQoe?.qoe != null ? bestQoe.qoe - worstQoe.qoe : null;
  const demandBand = value => value >= 15 ? "High" : value >= 11 ? "Medium" : "Low";
  const metricRows = [
    { label: "Scenario QoE", value: r => r.qoe?.toFixed(0) ?? "—", pct: r => Math.max(0, Math.min(100, r.qoe ?? 0)), className: "qoe" },
    { label: "Scenario Pressure", value: r => demandBand(r.demand ?? 0), pct: r => Math.max(0, Math.min(100, (r.demand ?? 0) * 5)), className: "demand" },
    { label: "Scenario RTT", value: r => r.rtt != null ? `${r.rtt.toFixed(0)} ms` : "—", pct: r => Math.max(0, Math.min(100, (r.rtt ?? 0) / 1.2)), className: "rtt" },
    { label: "Scenario loss", value: r => r.loss != null ? `${r.loss.toFixed(1)}%` : "—", pct: r => Math.max(0, Math.min(100, (r.loss ?? 0) * 25)), className: "loss" },
    { label: "AI Rank", value: r => r.aiRank != null ? `#${Math.round(r.aiRank)}` : "Not scored", pct: r => r.aiRank != null ? Math.max(10, 100 - Math.min(100, r.aiRank / 3)) : 0, className: "ai" },
  ];
  tableEl.innerHTML = `<div class="compare-guidance compact"><strong>${state.compareCells.length} / ${COMPARE_MAX} selected</strong><p>Hourly scenario metrics use the selected timeline hour. Quarterly AI rank uses the finals forecast context.</p></div>
    <div class="compare-verdict"><strong>Scenario comparison</strong><p><b>${cellName(bestQoe.h3)}</b> has the strongest current-hour QoE (${bestQoe.qoe?.toFixed(0) ?? "—"}). <b>${cellName(lowestDemand.h3)}</b> has the lowest Scenario Pressure (${lowestDemand.demand?.toFixed(0) ?? "—"}).${qoeGap != null ? ` The QoE gap between best and weakest selected cells is <b>${qoeGap.toFixed(0)} points</b>.` : ""}</p><small>Hourly scenario comparison; quarterly AI rank is labelled separately.</small></div>
    <div class="compare-visual" style="--compare-cols:${rows.length}">
      <div class="compare-visual-head"><span>Metric</span>${rows.map((r, i) => `<span class="${r.h3 === state.selectedH3 ? "active-compare-cell" : ""}"><b>Cell ${String.fromCharCode(65 + i)}${r.h3 === state.selectedH3 ? " · Active" : ""}</b><small>${cellName(r.h3)}</small><button onclick="toggleCompareCell('${r.h3}')" aria-label="Remove ${cellName(r.h3)}">×</button></span>`).join("")}</div>
      ${metricRows.map((metric, metricIndex) => `<div class="compare-metric-row ${metricIndex === 4 ? "quarterly-metric-start" : ""}"><strong>${metric.label}</strong>${rows.map(r => `<span class="${r.h3 === state.selectedH3 ? "active-compare-cell" : ""}"><b>${metric.value(r)}</b><i class="compare-bar ${metric.className}"><em style="width:${metric.pct(r)}%"></em></i></span>`).join("")}</div>`).join("")}
    </div>
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
    line: { color: "#d69e2e" }, name: "Average Scenario Pressure",
  }], plotlyLayout("Average Scenario Pressure Index"), plotlyConfig());

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

function syncOperatorChartTheme() {
  if (!window.Plotly) return;
  const colors = getComputedStyle(document.body);
  const foreground = colors.getPropertyValue("--op-secondary").trim();
  const surface = colors.getPropertyValue("--op-surface").trim();
  const border = colors.getPropertyValue("--op-border").trim();
  document.querySelectorAll(".js-plotly-plot").forEach(plot => {
    if (!plot.layout || !plot.offsetWidth || !plot.offsetHeight) return;
    Plotly.relayout(plot, {
      paper_bgcolor: surface, plot_bgcolor: surface,
      "font.family": "Segoe UI, Arial, sans-serif", "font.color": foreground,
      "title.font.color": foreground,
      "xaxis.gridcolor": border, "yaxis.gridcolor": border,
      "xaxis.tickfont.color": foreground, "yaxis.tickfont.color": foreground,
    });
  });
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
  for (const [id, timestamp] of [["rangeStartTimestamp", x0], ["rangeEndTimestamp", x1]]) {
    const output = document.getElementById(id);
    if (output) output.textContent = new Date(timestamp).toUTCString();
  }
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
  const totalCellHours = slice.length * state.panel.num_cells;
  const degradedPct = totalCellHours ? 100 * totalDegradedHours / totalCellHours : 0;
  const worstHour = slice.reduce((w, r) => (r.avg_qoe ?? 100) < (w.avg_qoe ?? 100) ? r : w, slice[0]);
  document.getElementById("rangeSummary").innerHTML =
    `Over the selected <b>${slice.length}h</b> window: avg QoE <b>${avgQoe.toFixed(0)}</b>, ` +
    `<b>${totalDegradedHours.toLocaleString()}</b> cell-hours degraded or severe (${degradedPct.toFixed(1)}% of cell-hours), ` +
    `worst hour was <b>${new Date(worstHour.timestamp).toUTCString().slice(0, 22)}</b> (avg QoE ${worstHour.avg_qoe?.toFixed(0)}).`;
  const first = slice[0], last = slice[slice.length - 1];
  const qoeChange = (last.avg_qoe ?? 0) - (first.avg_qoe ?? 0);
  const demandChange = (last.avg_demand_pressure ?? 0) - (first.avg_demand_pressure ?? 0);
  const qoeDirection = qoeChange > 1 ? "improved" : qoeChange < -1 ? "declined" : "remained broadly stable";
  const demandDirection = demandChange > 1 ? "increased" : demandChange < -1 ? "reduced" : "was broadly stable";
  const interpretation = document.getElementById("timeseriesInterpretation");
  if (interpretation) interpretation.innerHTML = `<strong>Interpretation</strong><ul><li><b>QoE:</b> ${qoeDirection} by ${Math.abs(qoeChange).toFixed(1)} points across this window.</li><li><b>Scenario pressure:</b> ${demandDirection} by ${Math.abs(demandChange).toFixed(1)} points.</li><li><b>Investigate:</b> the lowest QoE period, ${new Date(worstHour.timestamp).toUTCString().slice(5, 22)}, alongside its scenario pressure and severity bars.</li></ul><small>These charts describe the documented scenario pattern; they indicate where to investigate, not a proven causal relationship.</small>`;
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
  let whatIfDegraded = 0, whatIfQoeSum = 0, whatIfDemandSum = 0, whatIfN = 0;
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
        whatIfDemandSum += m.demand ?? 0;
        if (m.dl < thr.dl_degraded_mbps || m.rtt > thr.rtt_degraded_ms) whatIfDegraded++;
      }
    }
  }

  const anchoredPct = state.summary.ookla_anchored_pct;

  // Overall QoE and Degraded Cells already appear in the Network Health /
  // Degraded Cells cards up top (exec overview row) — no need to repeat
  // them here, so this row sticks to the metrics that are unique to it.
  const cards = [
    { label: "Peak Scenario Pressure", value: peakDemand.toFixed(0), delta: "", status: statusClass(100 - peakDemand, 30, 15) },
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
  const mapTimeHud = document.getElementById("mapTimeHud");
  if (mapTimeHud) mapTimeHud.textContent = new Date(cur.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  const mapFooterTime = document.getElementById("mapFooterTime");
  if (mapFooterTime) mapFooterTime.textContent = new Date(cur.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  document.getElementById("liveIndicator").innerHTML =
    `<span class="dot"></span> Scenario data as of ${new Date(state.summary.time_range.end).toUTCString().replace(" GMT", " UTC")}`;

  if (state.whatIfEnabled && whatIfN) {
    const realDegradedCount = cur.degraded_count + cur.severe_count;
    const degradedDelta = realDegradedCount - whatIfDegraded;
    const qoeDelta = (whatIfQoeSum / whatIfN) - cur.avg_qoe;
    document.getElementById("whatIfSummary").innerHTML =
      `<div class="whatif-result-title">Scenario effect · ${state.whatIfPct >= 0 ? "+" : ""}${state.whatIfPct}% capacity</div>` +
      `<div class="whatif-result-grid"><div><span>Degraded cells</span><b>${realDegradedCount} <i>→</i> ${whatIfDegraded}</b><small>${degradedDelta >= 0 ? "−" : "+"}${Math.abs(degradedDelta)} cells</small></div>` +
      `<div><span>Average QoE</span><b>${cur.avg_qoe?.toFixed(0)} <i>→</i> ${(whatIfQoeSum / whatIfN).toFixed(0)}</b><small>${qoeDelta >= 0 ? "+" : ""}${qoeDelta.toFixed(1)} points</small></div></div>`;
    const avgProjectedDemand = whatIfDemandSum / whatIfN;
    document.getElementById("whatIfExplanation").innerHTML = state.whatIfPct >= 0
      ? `<strong>Model response</strong><p>Extra capacity reduces the Scenario Pressure proxy from ${cur.avg_demand_pressure?.toFixed(1) ?? "—"} to ${avgProjectedDemand.toFixed(1)}. Under these illustrative assumptions, the calculation increases download throughput and reduces RTT, jitter and loss. These are scenario responses, not measured operational outcomes.</p>`
      : `<strong>Model response</strong><p>Less capacity raises the Scenario Pressure proxy from ${cur.avg_demand_pressure?.toFixed(1) ?? "—"} to ${avgProjectedDemand.toFixed(1)}. The model therefore reduces download throughput and increases RTT, jitter, and loss first in high-pressure cells.</p>`;
  } else {
    document.getElementById("whatIfSummary").innerHTML = "";
    document.getElementById("whatIfExplanation").innerHTML = "";
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

  const worst = [...rows].sort((a, b) => a.qoe - b.qoe);
  const highDemand = [...rows].sort((a, b) => b.demand - a.demand);

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
  state.persistentCount = persistent.length;
  renderRankedList("persistentList", persistent, r => `${r.count}/${WINDOW}h hot`);

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
  state.emergingCount = emerging.length;
  const currentEmerging = new Set(emerging.map(row => row.h3));
  if (state.playing && state.emergingReady) {
    const newlyEmerging = emerging.filter(row => !state.emergingH3s.has(row.h3)).slice(0, 3);
    if (newlyEmerging.length) playEmergenceRings(newlyEmerging);
  }
  state.emergingH3s = currentEmerging;
  state.emergingReady = true;
  renderRankedList("emergingList", emerging, r => `-${r.drop.toFixed(0)}%`);

  const moranCounts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0 };
  for (let c = 0; c < C; c++) moranCounts[cellValue("moran_code", t, c)] = (moranCounts[cellValue("moran_code", t, c)] || 0) + 1;
  const moranSummary = document.getElementById("moranSummary");
  if (moranSummary) moranSummary.innerHTML = [["High-high clusters", moranCounts[4]],["High-low outliers", moranCounts[3]],["Low-high outliers", moranCounts[2]],["Cold spots", moranCounts[1]],["Not significant", moranCounts[0]]].map(([label,count]) => `<div><span>${label}</span><b>${count.toLocaleString()}</b></div>`).join("");
}

function playEmergenceRings(rows) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  rows.forEach(row => {
    const props = state.gridPropsByH3.get(row.h3);
    if (!props || !state.map) return;
    const point = [props.centroid_lat, props.centroid_lon];
    const ring = L.circleMarker(point, { radius: 8, color: "#fb923c", weight: 2.2, opacity: .9, fillOpacity: 0, interactive: false, className: "map-emergence-ring" }).addTo(state.map);
    const label = L.marker(point, { interactive: false, icon: L.divIcon({ className: "map-emergence-label", html: "<span>New hotspot</span>", iconSize: [82, 22], iconAnchor: [41, 35] }) }).addTo(state.map);
    window.setTimeout(() => { state.map?.removeLayer(ring); state.map?.removeLayer(label); }, 950);
  });
}

function renderRankedList(elId, rows, metricFn) {
  const el = document.getElementById(elId);
  if (!rows.length) { el.innerHTML = `<li class="text-muted">None at this hour</li>`; return; }
  const incremental = ["worstQoeList", "highDemandList", "persistentList", "emergingList"].includes(elId);
  const visibleCount = state.expandedRankedLists.get(elId) || 5;
  const visibleRows = rows.slice(0, visibleCount);
  const remaining = Math.max(0, rows.length - visibleRows.length);
  el.innerHTML = visibleRows.map((r, index) => {
    const rankClass = index === 0 ? "rank-critical" : index === 1 ? "rank-high" : index === 2 ? "rank-watch" : "rank-standard";
    return `
      <li onclick="flyToCell('${r.h3}')">
        <span class="rank-badge ${rankClass}">${String(index + 1).padStart(2, "0")}</span>
        <span class="cell-id">${cellName(r.h3)}</span>
        <span class="metric">${metricFn(r)}</span>
      </li>
    `;
  }).join("") + (rows.length > 5 ? `<li class="ranked-list-toggle"><button onclick="toggleRankedList('${elId}', ${rows.length}, ${incremental})">${remaining ? (incremental ? `Show ${Math.min(20, remaining)} more` : `Show more (${rows.length.toLocaleString()})`) : "Show less"}</button></li>` : "");
}

function toggleRankedList(elId, totalRows, incremental = true) {
  const visibleCount = state.expandedRankedLists.get(elId) || 5;
  if (visibleCount >= totalRows) state.expandedRankedLists.delete(elId);
  else state.expandedRankedLists.set(elId, incremental ? Math.min(totalRows, visibleCount + 20) : totalRows);
  renderHotspotLists();
}

// ---------------------------------------------------------------------
// Investigation-evidence heuristic for the hourly Demo / Scenario.
// It suggests possible drivers; it does not infer engineering causes.
// ---------------------------------------------------------------------

const CAUSE_LABELS = {
  coverage: "Sparse mapped-tower context",
  backhaul: "Elevated relative latency",
  congestion: "Elevated activity-pressure proxy",
  indoor: "Built-environment intensity",
  none: "No strong relative signal",
};

const INVESTIGATION_SIGNAL_KEYS = ["coverage", "backhaul", "congestion", "indoor"];
const investigationContextCache = new Map();

function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position), upper = Math.ceil(position);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function percentileRank(sorted, value) {
  let low = 0, high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (sorted[middle] <= value) low = middle + 1; else high = middle;
  }
  return sorted.length ? low / sorted.length : 0;
}

function investigationValuesForCell(t, c, h3) {
  const props = state.gridPropsByH3.get(h3) || {};
  return {
    coverage: props.dist_to_nearest_tower_m,
    backhaul: cellValue("rtt_ms", t, c),
    congestion: cellValue("demand_pressure", t, c),
    indoor: props.built_up_ratio,
  };
}

function investigationContext(t) {
  if (investigationContextCache.has(t)) return investigationContextCache.get(t);
  const distributions = Object.fromEntries(INVESTIGATION_SIGNAL_KEYS.map(key => [key, []]));
  for (let c = 0; c < state.panel.num_cells; c++) {
    const values = investigationValuesForCell(t, c, state.panel.h3_ids[c]);
    INVESTIGATION_SIGNAL_KEYS.forEach(key => {
      if (Number.isFinite(values[key])) distributions[key].push(values[key]);
    });
  }
  Object.values(distributions).forEach(values => values.sort((a, b) => a - b));
  const thresholds = Object.fromEntries(INVESTIGATION_SIGNAL_KEYS.map(key => [key, quantile(distributions[key], .75)]));
  const context = { distributions, thresholds };
  investigationContextCache.set(t, context);
  return context;
}

function causeScoresForCell(t, c, h3) {
  const context = investigationContext(t);
  const values = investigationValuesForCell(t, c, h3);
  const percentiles = {};
  INVESTIGATION_SIGNAL_KEYS.forEach(key => {
    percentiles[key] = Number.isFinite(values[key]) ? percentileRank(context.distributions[key], values[key]) : null;
  });
  return percentiles;
}

function dominantCause(pct) {
  const triggered = INVESTIGATION_SIGNAL_KEYS.filter(key => pct[key] != null && pct[key] >= .75);
  return triggered.sort((a, b) => pct[b] - pct[a])[0] || "none";
}

function triggeredCauses(pct) {
  return INVESTIGATION_SIGNAL_KEYS.filter(key => pct[key] != null && pct[key] >= .75)
    .sort((a, b) => pct[b] - pct[a]);
}

function scenarioSupport(confidence) {
  if (confidence == null) return { key: "missing", label: "Unavailable" };
  if (confidence === 0) return { key: "none", label: "No support" };
  if (confidence < .3) return { key: "limited", label: "Limited" };
  if (confidence < .6) return { key: "moderate", label: "Moderate" };
  return { key: "stronger", label: "Stronger" };
}

function renderRootCause() {
  const t = state.currentT;
  const C = state.panel.num_cells;
  const counts = { coverage: 0, backhaul: 0, congestion: 0, indoor: 0 };
  const supportCounts = { none: 0, limited: 0, moderate: 0, stronger: 0, missing: 0 };
  let nDegraded = 0;
  let noSignal = 0;

  for (let c = 0; c < C; c++) {
    const cls = cellValue("degradation_class", t, c);
    if (cls < 1) continue;
    nDegraded++;
    const h3 = state.panel.h3_ids[c];
    const pct = causeScoresForCell(t, c, h3);
    const triggered = triggeredCauses(pct);
    if (!triggered.length) noSignal++;
    triggered.forEach(key => counts[key]++);
    supportCounts[scenarioSupport(cellValue("confidence", t, c)).key]++;
  }

  document.getElementById("rootCauseCellCount").textContent = nDegraded.toLocaleString();
  document.getElementById("rootCauseNoSignal").textContent = noSignal.toLocaleString();
  document.getElementById("rootCauseSupport").innerHTML = [
    ["No support", supportCounts.none], ["Limited", supportCounts.limited],
    ["Moderate", supportCounts.moderate], ["Stronger", supportCounts.stronger],
  ].map(([label, count]) => `<div><span>${label}</span><b>${count}</b><small>${nDegraded ? (100 * count / nDegraded).toFixed(0) : 0}%</small></div>`).join("");

  const sorted = Object.keys(counts).sort((a, b) => counts[a] - counts[b]);
  const labels = sorted.map(k => CAUSE_LABELS[k]);
  const values = sorted.map(k => counts[k]);
  const percentages = sorted.map(k => nDegraded ? 100 * counts[k] / nDegraded : 0);
  const colors = sorted.map(k => ({ congestion: "#7b4fa6", coverage: "#8e68b5", indoor: "#a183bf", backhaul: "#b19ac8" })[k]);

  const chart = document.getElementById("chartRootCause");
  Plotly.newPlot(chart, [{
    x: values, y: labels, type: "bar", orientation: "h", customdata: sorted.map((k, i) => [k, percentages[i]]),
    marker: { color: colors, line: { color: "#fff", width: 1 } },
    text: values.map((count, i) => `${count} · ${percentages[i].toFixed(0)}%`), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{y}<br>%{x} cells · %{customdata[1]:.1f}% of degraded cells<br>Cells may have multiple signals<br>Click to filter the queue<extra></extra>"
  }, { x: values.map(v => v === 1 ? 1 : null), y: labels, type: "scatter", mode: "markers", marker: { color: colors, size: 7, line: { color: "#fff", width: 1 } }, hoverinfo: "skip", showlegend: false }], { margin: { l: 220, r: 72, t: 12, b: 35 }, height: 285, showlegend: false, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", xaxis: { title: "Scenario cells", rangemode: "tozero", gridcolor: "#eee9f4", zeroline: false, tickfont: { size: 9 } }, yaxis: { tickfont: { size: 10, color: "#4f435c" }, automargin: true } }, plotlyConfig());
  chart.removeAllListeners?.("plotly_click");
  chart.on?.("plotly_click", event => {
    const key = event.points?.[0]?.customdata?.[0];
    if (key) setInterventionEvidenceFilter(state.interventionEvidenceFilter === key ? "all" : key);
  });

  if (nDegraded === 0) {
    document.getElementById("rootCauseSummary").innerHTML = `<b>Key takeaway:</b> No degraded scenario cells are present at this hour.`;
    return;
  }
  const leading = [...sorted].sort((a, b) => counts[b] - counts[a]);
  document.getElementById("rootCauseSummary").innerHTML =
    `<b>Key takeaway:</b> ${CAUSE_LABELS[leading[0]]} is the most common relative investigation signal: <b>${counts[leading[0]]}</b> of <b>${nDegraded}</b> degraded cells (${(100 * counts[leading[0]] / nDegraded).toFixed(0)}%). <b>${noSignal}</b> cells have no strong relative signal. Scenario evidence coverage is shown separately.`;
}

// ---------------------------------------------------------------------
// Interventions
// ---------------------------------------------------------------------

const ACTIONS = {
  congestion: "Compare the activity-pressure proxy with operator utilisation evidence.",
  coverage: "Review mapped-site context with field or operator coverage evidence.",
  indoor: "Compare the built-environment signal with indoor and field measurements.",
  backhaul: "Review the relative latency pattern with operator transport and field evidence.",
  none: "Collect targeted measurements and review the cell with network operations.",
};

function renderInterventionsLegacy() {
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
  const urgencyRank = cand => cand.cls === 2 && cand.population > 1000 ? 3 : (cand.cls === 2 || cand.population > 1000 ? 2 : 1);
  candidates.sort((a, b) => urgencyRank(b) - urgencyRank(a) || b.priority - a.priority);
  const top = candidates;
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
    const urgency = urgencyRank(cand) === 3 ? "high" : urgencyRank(cand) === 2 ? "medium" : "low";
    const tier = TIER_META[urgency];
    const status = state.tracking[cand.h3];
    const contact = OPERATOR_CONTACTS[cand.h3] || "Escalate to network operations (no per-cell operator data in this dataset)";

    // AI evidence row — only if the AI package is loaded, and clearly marked
    // as the frozen latest-run score rather than implying it's for hour t.
    let aiEvidence = "";
    if (state.onset) {
      const idx = state.h3IndexOf.get(cand.h3);
      const tierOnset = onsetValue("onset_priority_tier", idx);
      if (state.finals) {
        const rank = onsetValue("onset_risk_rank", idx);
        const current = onsetValue("current_status", idx);
        const label = rank != null ? `${ONSET_TIER_LABELS[tierOnset] ?? tierOnset}, rank #${rank}` : current === "currently_underperforming" ? "observed current issue; not forecast-eligible" : "not forecast-eligible";
        aiEvidence = ` · Quarterly AI priority: ${label}`;
      } else {
        const score = onsetValue("next_hour_onset_risk_score", idx);
        const opStatus = onsetValue("onset_operational_status", idx);
        const label = opStatus === "currently_degraded" ? "currently degraded (AI-scored)"
          : opStatus === "not_scored_unanchored" ? "not scored"
          : `${ONSET_TIER_LABELS[tierOnset] ?? tierOnset}${score != null ? `, rank score ${score.toFixed(2)}` : ""}`;
        aiEvidence = ` · AI onset risk (latest run): ${label}`;
      }
    }
    const persistentHotspot = cand.historicalHotspotFreq != null
      ? (cand.historicalHotspotFreq >= 0.3 ? "Yes" : "No") + ` (${(cand.historicalHotspotFreq * 100).toFixed(0)}% of hours)`
      : "—";

    return `
      <div class="intervention-card urgency-${urgency}${status === "done" ? " is-done" : ""}">
        <div onclick="flyToCell('${cand.h3}')" style="cursor:pointer">
          <div class="title"><span class="intervention-rank">${String(i + 1).padStart(2, "0")}</span>${cellName(cand.h3)}
            <span class="urgency-pill urgency-${urgency}">${tier.p} · ${urgency.toUpperCase()}</span>
          </div>
          <div class="intervention-section intervention-problem"><span>Problem</span><strong>${CAUSE_LABELS[cause]}</strong><p>Service class: ${["Normal", "Degraded", "Severe"][cand.cls]}</p></div>
          <div class="intervention-section intervention-evidence"><span>Evidence</span><div class="evidence-chips"><b>QoE ${cand.qoe?.toFixed(0) ?? "—"}</b><b>Scenario Pressure ${cand.demand?.toFixed(0) ?? "—"}</b><b>RTT ${cand.rtt?.toFixed(0) ?? "—"}ms</b><b>Population ${Math.round(cand.population).toLocaleString()}</b></div><p>Persistent hotspot: ${persistentHotspot}${aiEvidence}</p></div>
          <div class="intervention-section intervention-action"><span>Recommended action</span><p>${ACTIONS[cause]}</p></div>
          <details class="intervention-details"><summary>Operational detail</summary><p><b>Contact:</b> ${contact}</p><p class="cost-note"><b>Illustrative cost range:</b> ${ILLUSTRATIVE_COST_RANGES[cause]}</p></details>
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

function renderInterventions() {
  const t = state.currentT;
  const minConf = state.confidenceThreshold;
  const urgencyRank = cand => cand.cls === 2 && cand.population > 1000 ? 3 : (cand.cls === 2 || cand.population > 1000 ? 2 : 1);
  const urgencyName = cand => urgencyRank(cand) === 3 ? "high" : urgencyRank(cand) === 2 ? "medium" : "low";
  const workflowName = h3 => state.tracking[h3] === "done" ? "resolved" : state.tracking[h3] === "investigating" ? "investigating" : "active";
  const candidates = [];

  for (let c = 0; c < state.panel.num_cells; c++) {
    const cls = cellValue("degradation_class", t, c);
    if (cls < 1) continue;
    const confidence = cellValue("confidence", t, c);
    if (confidence != null && confidence < minConf) continue;
    const h3 = state.panel.h3_ids[c];
    const population = state.gridPropsByH3.get(h3)?.population_day ?? 0;
    const signalScores = causeScoresForCell(t, c, h3);
    const causes = triggeredCauses(signalScores);
    const cause = causes[0] || "none";
    const candidate = {
      c, h3, cls, cause, causes, signalScores, confidence, support: scenarioSupport(confidence), population,
      priority: cellValue("priority_base_score", t, c) ?? 0,
      qoe: cellValue("qoe", t, c),
      demand: cellValue("demand_pressure", t, c),
      rtt: cellValue("rtt_ms", t, c),
      loss: cellValue("loss_pct", t, c),
      historicalHotspotFreq: cellValue("historical_hotspot_freq", t, c),
    };
    candidate.urgency = urgencyName(candidate);
    candidate.workflow = workflowName(h3);
    candidates.push(candidate);
  }

  const countBy = (values, key) => values.reduce((acc, item) => { acc[item[key]] = (acc[item[key]] || 0) + 1; return acc; }, {});
  const priorityCounts = countBy(candidates, "urgency");
  const workflowCounts = countBy(candidates, "workflow");
  const evidenceCount = state.interventionEvidenceFilter === "all" ? candidates.length : candidates.filter(c => c.causes.includes(state.interventionEvidenceFilter)).length;
  let filtered = candidates.filter(c => state.interventionPriorityFilter === "all" || c.urgency === state.interventionPriorityFilter);
  filtered = filtered.filter(c => state.interventionStatusFilter === "all" || c.workflow === state.interventionStatusFilter);
  filtered = filtered.filter(c => state.interventionEvidenceFilter === "all" || c.causes.includes(state.interventionEvidenceFilter));
  filtered.sort((a, b) => urgencyRank(b) - urgencyRank(a) || b.priority - a.priority);
  state.lastCandidates = filtered;

  const filterButton = (type, value, label, count, active) => `<button class="queue-filter${active ? " active" : ""}" onclick="setInterventionFilter('${type}','${value}')"><span>${label}</span><b>${count}</b></button>`;
  const toolbar = document.getElementById("interventionsToolbar");
  toolbar.innerHTML = `<div class="queue-filter-row"><span>Scenario priority</span><div>${filterButton("priority", "all", "All", candidates.length, state.interventionPriorityFilter === "all")}${filterButton("priority", "high", "High", priorityCounts.high || 0, state.interventionPriorityFilter === "high")}${filterButton("priority", "medium", "Medium", priorityCounts.medium || 0, state.interventionPriorityFilter === "medium")}${filterButton("priority", "low", "Low", priorityCounts.low || 0, state.interventionPriorityFilter === "low")}</div></div>
    <div class="queue-filter-row"><span>Workflow</span><div>${filterButton("status", "all", "All", candidates.length, state.interventionStatusFilter === "all")}${filterButton("status", "active", "Active", workflowCounts.active || 0, state.interventionStatusFilter === "active")}${filterButton("status", "investigating", "Investigating", workflowCounts.investigating || 0, state.interventionStatusFilter === "investigating")}${filterButton("status", "resolved", "Resolved", workflowCounts.resolved || 0, state.interventionStatusFilter === "resolved")}</div></div>
    <div class="queue-toolbar-foot">${state.interventionEvidenceFilter !== "all" ? `<button class="queue-evidence-filter" onclick="setInterventionEvidenceFilter('all')"><span>Evidence: ${CAUSE_LABELS[state.interventionEvidenceFilter]}</span><b>${evidenceCount}</b><i>×</i></button>` : `<small>Select an Investigation Evidence bar to filter by category.</small>`}<button id="resetTrackingBtn" class="queue-reset" onclick="resetInterventionTracking()">Reset all statuses</button></div>`;

  const el = document.getElementById("interventionsList");
  if (!filtered.length) { el.innerHTML = `<div class="queue-empty"><b>No matching scenario cells</b><span>Adjust the priority, workflow, evidence, or confidence filters.</span></div>`; return; }
  const tierMeta = { high: "P1 · HIGH", medium: "P2 · MEDIUM", low: "P3 · LOW" };
  el.innerHTML = filtered.map((cand, i) => {
    const status = state.tracking[cand.h3] || "";
    const severity = cand.cls === 2 ? "Severe service degradation" : "Degraded service quality";
    const populationContext = cand.population > 1000 ? "higher affected population" : "localized affected population";
    const persistentHotspot = cand.historicalHotspotFreq != null ? `${(cand.historicalHotspotFreq * 100).toFixed(0)}% of scenario hours` : "Not available";
    let quarterlyContext = "No separate quarterly AI priority for this cell.";
    if (state.onset) {
      const idx = state.h3IndexOf.get(cand.h3), rank = onsetValue("onset_risk_rank", idx), tier = onsetValue("onset_priority_tier", idx), current = onsetValue("current_status", idx);
      quarterlyContext = rank != null ? `${ONSET_TIER_LABELS[tier] ?? tier}, rank #${Math.round(rank)}` : current === "currently_underperforming" ? "Observed quarterly issue; not forecast-eligible" : "Not forecast-eligible";
    }
    const supportingSignals = cand.causes.slice(1).map(key => CAUSE_LABELS[key]);
    const signalSummary = cand.causes.length ? cand.causes.map(key => `${CAUSE_LABELS[key]} (${(cand.signalScores[key] * 100).toFixed(0)}th percentile)`).join(" · ") : "No strong relative signal";
    return `<article class="intervention-card urgency-${cand.urgency}${status === "done" ? " is-done" : ""}">
      <button class="intervention-main" onclick="flyToCell('${cand.h3}')"><span class="intervention-rank">${String(i + 1).padStart(2, "0")}</span><span class="intervention-location">${cellName(cand.h3)}</span><span class="urgency-pill urgency-${cand.urgency}">${tierMeta[cand.urgency]}</span></button>
      <div class="intervention-section intervention-problem"><span>Primary investigation hypothesis</span><strong>${CAUSE_LABELS[cand.cause]}</strong></div>
      <div class="intervention-support">Scenario evidence coverage · <b>${cand.support.label}</b></div>
      <div class="intervention-section intervention-evidence"><span>Scenario metrics</span><div class="evidence-chips"><b>QoE ${cand.qoe?.toFixed(0) ?? "—"}</b><b>Pressure ${cand.demand?.toFixed(0) ?? "—"}</b><b>RTT ${cand.rtt?.toFixed(0) ?? "—"} ms</b><b>Population ${Math.round(cand.population).toLocaleString()}</b></div><p><b>Why prioritised:</b> ${severity} with ${populationContext}.</p></div>
      <div class="intervention-section intervention-action"><span>Recommended next step</span><p>${ACTIONS[cand.cause]}</p></div>
      <details class="intervention-details"><summary>Operational detail</summary><div><p><b>Scenario evidence coverage:</b> ${cand.support.label} (${cand.confidence != null ? `${(cand.confidence * 100).toFixed(0)}%` : "unavailable"})</p><p><b>Supporting signals:</b> ${supportingSignals.length ? supportingSignals.join(" · ") : "None"}</p><p><b>Signal percentiles:</b> ${signalSummary}</p><p><b>Quarterly AI context:</b> ${quarterlyContext}</p><p><b>Persistent scenario hotspot:</b> ${persistentHotspot}</p><p><b>Signal threshold:</b> citywide 75th percentile for the selected scenario hour</p><p><b>Packet loss:</b> ${cand.loss?.toFixed(2) ?? "—"}%</p><p><b>Contact:</b> ${OPERATOR_CONTACTS[cand.h3] || "Escalate to network operations; no per-cell operator field is available."}</p><p class="cost-note"><b>Illustrative cost range:</b> ${ILLUSTRATIVE_COST_RANGES[cand.cause] ?? "Determine after validation"}</p></div></details>
      <div class="track-actions"><button class="${!status || status === "planned" ? "active" : ""}" onclick="setTracking('${cand.h3}','planned')">Active</button><button class="${status === "investigating" ? "active" : ""}" onclick="setTracking('${cand.h3}','investigating')">Investigating</button><button class="${status === "done" ? "active" : ""}" onclick="setTracking('${cand.h3}','done')">Resolved</button></div>
    </article>`;
  }).join("");
}

function setInterventionFilter(type, value) {
  if (type === "priority") state.interventionPriorityFilter = value;
  if (type === "status") state.interventionStatusFilter = value;
  renderInterventions();
}

function setInterventionEvidenceFilter(value) {
  state.interventionEvidenceFilter = value;
  renderInterventions();
  if (value !== "all") setWorkspaceTool("actions");
}

function resetInterventionTracking() {
  if (!window.confirm("Reset all investigation statuses to Active? This clears saved Investigating and Resolved states for every scenario cell.")) return;
  state.tracking = {};
  saveTracking();
  renderInterventions();
  renderPriorityActionCentre();
  renderExecOverview();
}

function setTracking(h3, status) {
  if (status) state.tracking[h3] = status; else delete state.tracking[h3];
  saveTracking();
  renderInterventions();
  renderPriorityActionCentre();
  renderExecOverview();
  if (state.workspaceTool === "prioritise") renderQuarterlyQueue();
  if (h3 === state.selectedH3) {
    renderSelectedAreaInsights();
    syncWorkspaceSelection();
  }
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
  if (cmpEl) cmpEl.classList.toggle("d-none", Boolean(state.finals));
  if (state.finals) {
    document.querySelectorAll('#colorBySelect option[value="static_ai"], #colorBySelect option[value="comparison"]')
      .forEach(opt => opt.disabled = true);
  }
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
  if (state.finals) {
    el.className = "ai-time-banner ok";
    el.innerHTML = `✓ <b>Quarterly forecast:</b> ${state.onset.forecast_window}. This ranking is fixed to the finals model and does not change with scenario playback.`;
    return;
  }
  const atSnapshot = state.onsetSnapshotT != null && state.currentT === state.onsetSnapshotT;
  const snapLabel = state.onset.snapshot_timestamp + " UTC";
  if (state.onsetSnapshotT == null || state.onsetSnapshotT === -1) {
    el.className = "ai-time-banner warn";
    el.innerHTML = `⚠️ <b>Model hour:</b> ${snapLabel}. This is a standalone reference run.`;
  } else if (atSnapshot) {
    el.className = "ai-time-banner ok";
    el.innerHTML = `✅ <b>Model hour:</b> ${snapLabel}. The forecast matches the map.`;
  } else {
    el.className = "ai-time-banner warn";
    el.innerHTML = `⚠️ <b>Model hour:</b> ${snapLabel}; map is on a different hour. <a href="#" onclick="jumpToAiSnapshotHour(); return false;">Jump to model hour</a>`;
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
  el.innerHTML = `⚠️ <b>${word}</b>: precision ${(precision * 100).toFixed(0)}%, recall ${(recall * 100).toFixed(0)}%. Rank ≠ failure probability.`;
}

function renderAiPanel() {
  const C = state.panel.num_cells;
  if (state.finals) {
    document.getElementById("aiForecastBandFilters")?.classList.remove("d-none");
    document.querySelector(".ai-legacy-risk-list")?.classList.add("d-none");
    const rows = Array.from({ length: C }, (_, c) => ({
      c, h3: state.panel.h3_ids[c], tier: onsetValue("onset_priority_tier", c),
      rank: onsetValue("onset_risk_rank", c), support: onsetValue("measurement_support", c),
    })).filter(row => row.rank != null).sort((a, b) => a.rank - b.rank);
    const high = rows.filter(row => row.tier === "high");
    const elevated = rows.filter(row => row.tier === "elevated");
    const lower = rows.filter(row => row.tier === "lower");
    const currentIssues = state.onset.fields.current_status.filter(value => value === "currently_underperforming").length;
    const thresholds = state.finals.metadata.risk_thresholds || {};
    const highHeading = document.getElementById("aiHighRiskHeading");
    if (highHeading) highHeading.textContent = `High emerging risk (Top ${thresholds.high_max_rank ?? "—"})`;
    state.aiSummary = { degradedNowCount: currentIssues, topRiskCell: rows[0] || null };
    const bandRows = { high, elevated, lower };
    if (!bandRows[state.aiForecastBandFilter]) state.aiForecastBandFilter = "high";
    const counts = { high: high.length, elevated: elevated.length, lower: lower.length };
    const filters = document.getElementById("aiForecastBandFilters");
    if (filters) filters.innerHTML = ["high", "elevated", "lower"].map(band => `<button class="ai-band-chip band-${band}${state.aiForecastBandFilter === band ? " active" : ""}" aria-pressed="${state.aiForecastBandFilter === band}" onclick="setAiForecastBand('${band}')"><span>${band === "lower" ? "Lower" : band[0].toUpperCase() + band.slice(1)}</span><b>${counts[band].toLocaleString()}</b></button>`).join("");
    const selectedRows = bandRows[state.aiForecastBandFilter];
    const queue = document.getElementById("aiDegradedNowList");
    const visibleRows = state.aiForecastExpanded ? selectedRows : selectedRows.slice(0, 5);
    if (queue) queue.innerHTML = selectedRows.length ? visibleRows.map(row => `<li role="button" tabindex="0" class="ai-queue-item band-${row.tier}" onclick="flyToCell('${row.h3}')"><span class="rank-badge rank-${row.tier === "high" ? "critical" : row.tier === "elevated" ? "watch" : "standard"}">${String(Math.round(row.rank)).padStart(2,"0")}</span><span class="ai-queue-place"><b>${cellName(row.h3)}</b></span><span class="ai-queue-band">${ONSET_TIER_LABELS[row.tier] ?? row.tier}</span><i class="bi bi-chevron-right"></i></li>`).join("") + (selectedRows.length > 5 ? `<li class="ranked-list-toggle"><button onclick="toggleAiForecastList()">${state.aiForecastExpanded ? "Show less" : `Show more (${selectedRows.length.toLocaleString()})`}</button></li>` : "") : `<li class="text-muted">No cells in this risk band.</li>`;
    const queueHeading = document.getElementById("aiQueueHeading");
    if (queueHeading) queueHeading.textContent = `${state.aiForecastBandFilter === "lower" ? "Lower" : state.aiForecastBandFilter[0].toUpperCase() + state.aiForecastBandFilter.slice(1)}-risk investigation queue`;
    document.getElementById("aiModelBehaviorBadge").innerHTML = `✦ <b>Finals quarterly XGBoost:</b> ranks currently non-underperforming cells by emerging underperformance risk; raw scores are not displayed as probabilities.`;
    document.getElementById("aiSnapshotNote").innerHTML = `<b>${rows.length.toLocaleString()} evaluable cells</b> ranked for emerging underperformance in ${state.finals.metadata.forecast_quarter}. <span>Ranks are not probabilities.</span> <button class="ai-reveal-link" onclick="setWorkspaceTool('overview')">View observed issues in Reveal</button>`;
    const activeNotice = document.getElementById("aiActiveCellNotice");
    const activeIndex = state.h3IndexOf.get(state.selectedH3);
    const activeUnderperforming = activeIndex != null && onsetValue("current_status", activeIndex) === "currently_underperforming";
    if (activeNotice) { activeNotice.classList.toggle("d-none", !activeUnderperforming); activeNotice.textContent = activeUnderperforming ? "This active cell is already underperforming and is not forecast-eligible. Select a ranked cell below to inspect its emerging-risk forecast." : ""; }
    renderFinalsTimeMachine();
    return;
  }
  const rows = [];
  document.getElementById("aiForecastBandFilters")?.classList.add("d-none");
  document.querySelector(".ai-legacy-risk-list")?.classList.remove("d-none");
  const queueHeading = document.getElementById("aiQueueHeading");
  if (queueHeading) queueHeading.textContent = "Currently degraded";
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
    `<b>Snapshot:</b> ${meta.snapshot_timestamp} UTC · ${degradedNow.length} degraded · ${topRisk.length ? topRisk.length + "+" : "0"} risk-ranked.` +
    `<details><summary>Model details</summary><p>${meta.static_observed_cells.toLocaleString()} real Ookla-labelled cells and ${meta.static_ai_estimated_cells.toLocaleString()} AI-estimated cells. <code>${meta.model_name}</code>, threshold ${meta.model_frozen_threshold.toFixed(3)}. Scores are relative ranks in a simulated scenario, not calibrated probabilities.</p></details>`;
}

function setAiForecastBand(band) {
  if (!["high", "elevated", "lower"].includes(band)) return;
  state.aiForecastBandFilter = band;
  state.aiForecastExpanded = false;
  renderAiPanel();
}

function toggleAiForecastList() { state.aiForecastExpanded = !state.aiForecastExpanded; renderAiPanel(); }

function timeMachineLabel(value) {
  return ({
    correctly_anticipated: "Correctly anticipated",
    false_alert: "False alert",
    missed: "Missed",
    correctly_lower_risk: "Correctly lower-risk",
  })[value] || "Not available";
}

async function toggleMapFullscreen() {
  const panel = document.getElementById("mapPanel");
  if (!panel || !document.fullscreenEnabled) return;
  try {
    if (document.fullscreenElement === panel) await document.exitFullscreen();
    else {
      if (document.fullscreenElement) await document.exitFullscreen();
      await panel.requestFullscreen();
    }
  } catch (error) {
    console.error("Map full-screen mode failed", error);
  }
}

function syncMapFullscreenState() {
  const panel = document.getElementById("mapPanel");
  const button = document.getElementById("mapFullscreenBtn");
  const active = Boolean(panel && document.fullscreenElement === panel);
  document.body.classList.toggle("map-is-fullscreen", active);
  if (button) {
    button.setAttribute("aria-pressed", String(active));
    button.setAttribute("aria-label", active ? "Exit full-screen map" : "Present map full screen");
    button.title = active ? "Exit full-screen map" : "Present map full screen";
    button.innerHTML = `<i class="bi ${active ? "bi-fullscreen-exit" : "bi-arrows-fullscreen"}" aria-hidden="true"></i><span>${active ? "Exit full screen" : "Full screen"}</span>`;
    button.disabled = !document.fullscreenEnabled;
  }
  window.setTimeout(() => state.map?.invalidateSize(), 100);
}

function formatModelObservedValue(label, value) {
  if (value == null || value === "") return "—";
  const text = String(value).trim();
  const number = Number.parseFloat(text.replaceAll(",", ""));
  if (!Number.isFinite(number)) return text;
  const name = String(label || "").toLowerCase();
  if (name.includes("built-up ratio")) return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
  if (name.includes("viirs") || name.includes("human-activity")) return number.toFixed(1);
  if (name.includes("population")) return Math.round(number).toLocaleString();
  if (name.includes("underperformance rate") || name.includes("share") || name.includes("rate")) return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
  if (name.includes("download") || name.includes("upload") || /mbps/i.test(text)) return `${number.toFixed(1)} Mbps`;
  if (name.includes("latency") || name.includes("rtt") || /\bms\b/i.test(text)) return `${number.toFixed(1)} ms`;
  if (name.includes("distance")) return `${number.toLocaleString(undefined, { maximumFractionDigits: 1 })}${/\bm\b/i.test(text) ? " m" : ""}`;
  return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function modelContributorHtml(row, effect = "higher") {
  const higher = effect === "higher";
  return `<li class="model-contributor ${higher ? "risk-raising" : "risk-lowering"}"><div class="model-contributor-main"><b>${row.label}</b><small>Observed value</small><em>${formatModelObservedValue(row.label, row.value)}</em></div><strong><span>Model effect</span><b>${higher ? "↑ higher risk" : "↓ lower risk"}</b></strong></li>`;
}

function renderFinalsTimeMachine() {
  const el = document.getElementById("aiTimeMachineStatus");
  if (!el || !state.finals) return;
  const eligible = state.finals.metadata.eligible_cells || 0;
  const reviewTarget = state.finals.metadata.risk_thresholds?.elevated_max_rank || 200;
  const reducedPct = eligible ? (100 * (eligible - reviewTarget) / eligible).toFixed(1) : "—";
  if (!state.finalsActualRevealed) {
    el.innerHTML = `<div class="ai-value-strip" aria-label="Investigation workload reduction"><div><span>Evaluable forecast cells</span><b>${eligible.toLocaleString()}</b></div><i class="bi bi-arrow-right"></i><div><span>Review first</span><b>Top ${reviewTarget}</b></div><strong>${reducedPct}% fewer cells to triage first</strong></div><div class="ai-time-machine-card"><div><span>Held-out validation · Time Machine</span><b>Forecast ${state.finals.metadata.predictor_quarter} → known ${state.finals.metadata.forecast_quarter} outcome</b><small>Reveal the held-out result used to evaluate the ranking; this is not live future data.</small></div><button class="btn btn-sm btn-outline-primary" onclick="revealFinalsActual()">Reveal held-out result</button></div>`;
    return;
  }
  const results = state.onset.fields.time_machine_result.filter(Boolean);
  const counts = results.reduce((all, value) => ({ ...all, [value]: (all[value] || 0) + 1 }), {});
  const captured = counts.correctly_anticipated || 0;
  const totalIssues = captured + (counts.missed || 0);
  const reviewed = captured + (counts.false_alert || 0);
  const top50 = state.onset.fields.onset_risk_rank.map((rank, c) => ({rank, result: state.onset.fields.time_machine_result[c]})).filter(r => r.rank != null && r.rank <= 50 && r.result);
  const top50Hits = top50.filter(r => r.result === "correctly_anticipated").length;
  const recallPct = totalIssues ? (100 * captured / totalIssues).toFixed(1) : "—";
  const precisionPct = top50.length ? (100 * top50Hits / top50.length).toFixed(0) : "—";
  const active = state.timeMachineOutcomeFilter;
  const outcomeCard = (key, className, icon) => `<button type="button" class="${className}${active === key ? " is-active" : ""}" onclick="setTimeMachineOutcomeFilter('${key}')" aria-pressed="${active === key}"><i class="bi ${icon}"></i><span>${timeMachineLabel(key)}</span><b>${counts[key] || 0}</b><small>Show on map</small></button>`;
  el.innerHTML = `<div class="time-machine-filter-head"><div><b>Map validation outcomes</b><span>Select a category to locate its cells.</span></div><button type="button" class="btn btn-sm btn-outline-primary${active === "all" ? " active" : ""}" onclick="setTimeMachineOutcomeFilter('all')" aria-pressed="${active === "all"}">All outcomes</button></div><div class="time-machine-outcomes" aria-label="Held-out forecast outcome map filters">${outcomeCard("correctly_anticipated", "outcome-correct", "bi-check-circle-fill")}${outcomeCard("false_alert", "outcome-alert", "bi-exclamation-circle-fill")}${outcomeCard("missed", "outcome-missed", "bi-x-circle-fill")}${outcomeCard("correctly_lower_risk", "outcome-lower", "bi-shield-check")}</div><div class="time-machine-proof"><div><span>Top 200 reviewed</span><b>${reviewed} <small>/ ${results.length.toLocaleString()}</small></b></div><div><span>Emerging issues captured</span><b>${captured} <small>/ ${totalIssues}</small></b><em>${recallPct}%</em></div><div><span>Top-50 precision</span><b>${top50Hits} <small>/ ${top50.length}</small></b><em>${precisionPct}%</em></div></div><div class="ai-time-machine-card revealed"><div><span>Held-out ${state.finals.metadata.forecast_quarter} outcome</span><b>Forecast versus actual</b><small>The map now uses the four outcome categories. Hover or select a cell to compare its rank with the known result. Ranking scores are not probabilities.</small></div><button class="btn btn-sm btn-outline-primary" onclick="hideFinalsActual()">Back to forecast</button></div>`;
}

function setTimeMachineOutcomeFilter(filter) {
  if (!["all", ...Object.keys(TIME_MACHINE_COLORS)].includes(filter)) return;
  state.timeMachineOutcomeFilter = filter;
  renderFinalsTimeMachine();
  renderMapColors();
  renderLegend();
}

function hideFinalsActual() {
  state.finalsActualRevealed = false;
  state.timeMachineOutcomeFilter = "all";
  renderFinalsTimeMachine(); renderMapColors(); renderLegend(); syncPrimaryMapControls();
  if (state.selectedH3) renderSelectedAreaInsights();
}

function revealFinalsActual() {
  if (!state.finals) return;
  state.finalsActualRevealed = true;
  state.timeMachineOutcomeFilter = "all";
  renderFinalsTimeMachine();
  renderMapColors();
  renderLegend();
  syncPrimaryMapControls();
  if (state.selectedH3) renderSelectedAreaInsights();
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
  if (state.finals) {
    el.innerHTML = `<div class="scope-comparison"><div class="scope-heading"><span></span><span>QUARTERLY AI</span><span>DEMO HOTSPOTS</span></div>${[["Purpose","Future emerging risk","Current spatial clustering / outliers"],["Time basis","2026Q1 → 2026Q2","Selected scenario hour"],["Data","Real quarterly network + GeoAI context","Demo / Scenario hourly panel"],["Use for","Where to investigate future risk","Where degradation clusters in the current scenario"]].map(r=>`<div class="scope-row"><b>${r[0]}</b><span>${r[1]}</span><span>${r[2]}</span></div>`).join("")}</div><div class="disclaimer-box"><b>Different questions, different time bases.</b> These outputs should not be compared numerically.</div>`;
    return;
  }
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
      ? `The AI model flags <b>${counts.ai_only} cell${counts.ai_only === 1 ? "" : "s"}</b> that traditional Gi* hotspot analysis does not — e.g. ${[...new Set(aiOnlyExamples.map(x => cellName(x.h3)))].join(", ")}${aiOnlyExamples.length < counts.ai_only ? ", …" : ""}. These are candidates where the onset model's time-series and scenario-pressure features caught something a purely spatial statistic wouldn't.`
      : `At this snapshot hour, every AI-flagged cell is also a traditional Gi* hotspot — no AI-only detections to highlight.`}
      ${counts.traditional_only > 0 ? ` Conversely, <b>${counts.traditional_only}</b> traditional hotspots aren't in the AI's top risk tiers this hour — spatial clustering alone isn't the same signal as the onset model's ranking.` : ""}</p>
    <div class="disclaimer-box mt-2">The AI model doesn't replace spatial analytics — it's a different lens (temporal/scenario-pressure-driven versus purely spatial) evaluated on the same simulated hour. Use "🤖 AI vs Hotspot" on the map's Color-by dropdown to see this spatially.</div>
  `;
}

function populateAiModelModal() {
  const card = state.modelCard;
  const el = document.getElementById("aiModelModalBody");
  if (!el) return;
  if (state.finals) {
    const meta = state.finals.metadata;
    const test = meta.final_test || {};
    const topK = test.top_k || [];
    const thresholds = meta.risk_thresholds || {};
    const families = meta.shap_family_importance || [];
    const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
    el.innerHTML = `
      <p><b>What this means:</b> Use the ranking to decide where to investigate next. It does not diagnose a fault or provide an individual failure probability.</p>
      <table class="table table-sm"><tbody>
        <tr><td>Model version</td><td><b>${meta.model_version ?? "—"}</b></td></tr>
        <tr><td>Forecast window</td><td><b>${meta.predictor_quarter ?? "—"} → ${meta.forecast_quarter ?? "—"}</b></td></tr>
        <tr><td>Evaluable forecast cells</td><td><b>${meta.eligible_cells?.toLocaleString?.() ?? "—"}</b></td></tr>
        <tr><td>Final PR-AUC</td><td><b>${test.pr_auc?.toFixed?.(3) ?? "—"}</b></td></tr>
        <tr><td>Final ROC-AUC</td><td><b>${test.roc_auc?.toFixed?.(3) ?? "—"}</b></td></tr>
        ${topK.map(row => `<tr><td>Top-${row.k} outcome</td><td><b>${row.captured_events} / ${row.k}</b> deteriorated (${pct(row.precision)})</td></tr>`).join("")}
      </tbody></table>
      <details><summary>Technical methodology</summary><p class="text-summary">${meta.model_family ?? "Model"} with ${meta.feature_count ?? "—"} predictors. ${families.map(row => `${row.family}: ${row.importance_share_pct.toFixed(1)}%`).join(" · ")}. These shares describe model behaviour, not causality.</p><p class="text-summary">High: ranks 1–${thresholds.high_max_rank ?? "—"}; elevated: ${Number(thresholds.high_max_rank ?? 50) + 1}–${thresholds.elevated_max_rank ?? "—"}. Measurement support describes Ookla coverage, not model confidence.</p></details>`;
    return;
  }
  if (!card) return;

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
  if (!el) return;
  if (state.finals) {
    const meta = state.finals.metadata;
    const test = meta.final_test || {};
    const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
    el.innerHTML = `<p><b>What this means:</b> The held-out test shows how well the model ordered investigation candidates for ${meta.forecast_quarter ?? "the next quarter"}.</p><table class="table table-sm"><thead><tr><th>Review budget</th><th>Observed emerging issues</th><th>Captured</th></tr></thead><tbody>${(test.top_k || []).map(row => `<tr><td>Top ${row.k}</td><td><b>${row.captured_events} / ${row.k}</b> (${pct(row.precision)})</td><td>${pct(row.recall)}</td></tr>`).join("")}</tbody></table><details><summary>Technical metrics</summary><p class="text-summary">PR-AUC <b>${test.pr_auc?.toFixed?.(3) ?? "—"}</b> · no-skill PR-AUC <b>${test.no_skill_pr_auc?.toFixed?.(3) ?? "—"}</b> · ROC-AUC <b>${test.roc_auc?.toFixed?.(3) ?? "—"}</b> · event rate ${pct(test.event_rate)}. These are ranking metrics, not accuracy or individual probabilities.</p></details>`;
    return;
  }
  if (!card) return;
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
  const candidates = state.lastCandidates || [];
  let peakDemand = 0;
  let populationAffected = 0;
  for (let c = 0; c < C; c++) {
    peakDemand = Math.max(peakDemand, cellValue("demand_pressure", t, c) ?? 0);
    if ((cellValue("degradation_class", t, c) ?? 0) >= 1) {
      populationAffected += state.gridPropsByH3.get(state.panel.h3_ids[c])?.population_day || 0;
    }
  }

  const qoeDelta = deltaStr(cur.avg_qoe, prev?.avg_qoe, false);
  const meta = state.finals?.metadata || {};
  const observed = meta.observed_layer || {};
  const highCutoff = meta.risk_thresholds?.high_max_rank;
  const elevatedCutoff = meta.risk_thresholds?.elevated_max_rank;
  let topQuarterly = null;
  let supported = 0;
  for (let c = 0; c < C; c++) {
    const rank = onsetValue("onset_risk_rank", c);
    if (rank != null && (!topQuarterly || rank < topQuarterly.rank)) topQuarterly = { rank, h3: state.panel.h3_ids[c], tier: onsetValue("onset_priority_tier", c) };
    if (onsetValue("measurement_support", c)) supported++;
  }
  const cards = [
    { label: "Observed underperformance", value: `${observed.underperforming_cells?.toLocaleString?.() ?? "—"}`, sub: `NOW · ${meta.predictor_quarter ?? "Current quarter"}` },
    { label: "High emerging risk", value: highCutoff?.toLocaleString?.() ?? "—", sub: `NEXT · ${meta.forecast_quarter ?? "Next quarter"}` },
    { label: "Elevated emerging risk", value: highCutoff != null && elevatedCutoff != null ? (elevatedCutoff - highCutoff).toLocaleString() : "—", sub: `NEXT · ${meta.forecast_quarter ?? "Next quarter"}` },
    { label: "Evaluable forecast cells", value: meta.eligible_cells?.toLocaleString?.() ?? "—", sub: "Currently normal with held-out next-quarter outcomes" },
    { label: "Measured cells", value: observed.measured_cells?.toLocaleString?.() ?? "—", sub: `${meta.predictor_quarter ?? "Current quarter"} Ookla coverage` },
    { label: "Quarterly measurement support", value: supported.toLocaleString(), sub: "Forecast cells with a reported Ookla support band" },
    { label: "Demo / Scenario cells", value: C.toLocaleString(), sub: "Separate hourly M-Lab / cell_time_dataset view" },
  ];

  el.innerHTML = cards.map(c => `
    <div class="exec-card">
      <div class="exec-label">${c.label}</div>
      <div class="exec-value">${c.value}</div>
      <div class="exec-sub">${c.sub}</div>
    </div>
  `).join("");

  const workspaceGrid = document.getElementById("workspaceOverviewGrid");
  if (workspaceGrid) {
    workspaceGrid.innerHTML = cards.slice(0, 3).map((c, index) => `<div class="overview-status-${index}"><span>${c.label}</span><b>${c.value}</b><small>${c.sub}</small></div>`).join("");
  }
  const additionalMetrics = document.getElementById("workspaceAdditionalMetrics");
  if (additionalMetrics) {
    const measured = observed.measured_cells ?? 0;
    const underperforming = observed.underperforming_cells ?? 0;
    const currentlyNormal = measured - underperforming;
    const evaluable = meta.eligible_cells ?? 0;
    const unavailable = Math.max(0, currentlyNormal - evaluable);
    additionalMetrics.innerHTML = `<div class="coverage-arithmetic"><div><span>Measured in ${meta.predictor_quarter}</span><b>${measured.toLocaleString()}</b></div><div><span>Observed underperforming</span><b>${underperforming.toLocaleString()}</b></div><div><span>Currently normal</span><b>${currentlyNormal.toLocaleString()}</b></div><div><span>Held-out ${meta.forecast_quarter} outcome available</span><b>${evaluable.toLocaleString()}</b></div><div><span>Outcome unavailable</span><b>${unavailable.toLocaleString()}</b></div></div><p class="coverage-note">${evaluable.toLocaleString()} evaluable forecast cells are the subset of currently-normal cells with held-out outcomes.</p>`;
  }
  const priorityEl = document.getElementById("workspaceTopPriority");
  if (priorityEl) {
    priorityEl.innerHTML = topQuarterly
      ? `<article class="top-priority-action risk-${topQuarterly.tier ?? "unscored"}"><div><span>Highest AI-ranked currently-normal cell</span><strong>${cellName(topQuarterly.h3)}</strong><p><b>${ONSET_TIER_LABELS[topQuarterly.tier] ?? topQuarterly.tier}</b> · <em>rank #${Math.round(topQuarterly.rank)}</em></p></div><button onclick="flyToCell('${topQuarterly.h3}')">Inspect cell <i class="bi bi-arrow-right"></i></button></article>`
      : `<span class="text-summary">No quarterly forecast candidate is available.</span>`;
  }
  if (window.Plotly && document.getElementById("workspacePulseChart")) {
    const xs = state.cityTS.map(d => new Date(d.timestamp));
    const ys = state.cityTS.map(d => d.avg_qoe);
    const demand = state.cityTS.map(d => d.avg_demand_pressure);
    const rangeStart = Math.max(0, t - 23);
    Plotly.react("workspacePulseChart", [
      { x: xs, y: ys, type: "scatter", mode: "lines", name: "QoE", fill: "tozeroy", fillcolor: "rgba(104,70,154,.10)", line: { color: "#68469a", width: 2.5 }, hovertemplate: "QoE: <b>%{y:.1f}</b><extra></extra>" },
      { x: xs, y: demand, type: "scatter", mode: "lines", name: "Scenario Pressure", yaxis: "y2", line: { color: "#f59e0b", width: 1.5, dash: "dot" }, hovertemplate: "Scenario Pressure: <b>%{y:.1f}</b><extra></extra>" }
    ], { margin: { l: 52, r: 52, t: 14, b: 42 }, height: 250, dragmode: "pan", hovermode: "x unified", xaxis: { title: { text: "Selected scenario hour", font: { size: 11 } }, range: [xs[rangeStart], xs[t]], rangeslider: { visible: true, thickness: .14 }, tickformat: "%H:%M<br>%d %b", hoverformat: "%H:%M, %d %b %Y", nticks: 5, tickfont: { size: 10 }, showgrid: false }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", yaxis: { title: { text: "QoE score", font: { size: 11 } }, tickfont: { size: 10 }, gridcolor: "#ece7f5" }, yaxis2: { title: { text: "Pressure", font: { size: 11 } }, tickfont: { size: 10 }, overlaying: "y", side: "right", showgrid: false }, showlegend: false }, { displayModeBar: false, responsive: true, scrollZoom: false });
    const pulsePlot = document.getElementById("workspacePulseChart");
    if (pulsePlot && !pulsePlot.dataset.syncBound) {
      pulsePlot.dataset.syncBound = "1";
      pulsePlot.on("plotly_click", event => { const idx = event.points?.[0]?.pointIndex; if (Number.isInteger(idx)) { state.currentT = idx; document.getElementById("timeSlider").value = idx; renderAll(); } });
      pulsePlot.on("plotly_relayout", event => {
        const from = event["xaxis.range[0]"] || event["xaxis.range"]?.[0];
        const to = event["xaxis.range[1]"] || event["xaxis.range"]?.[1];
        const label = document.getElementById("workspacePulseVisibleRange");
        if (label && from && to) label.textContent = `Showing: ${new Date(from).toUTCString().slice(5,22)} → ${new Date(to).toUTCString().slice(5,22)}`;
      });
    }
    const pulseMeta = document.getElementById("workspacePulseMeta");
    if (pulseMeta) pulseMeta.innerHTML = `<span><i class="pulse-dot"></i> QoE <b>${cur.avg_qoe?.toFixed(1) ?? "-"}</b></span><span>Pressure <b>${cur.avg_demand_pressure?.toFixed(1) ?? "-"}</b></span><span>${qoeDelta || "Stable versus last hour"}</span>`;
    const pulseTime = document.getElementById("workspacePulseSelectedTime");
    if (pulseTime) pulseTime.textContent = `Selected scenario hour: ${new Date(cur.timestamp).toUTCString().slice(0, 22)}`;
    const visibleRange = document.getElementById("workspacePulseVisibleRange");
    if (visibleRange) visibleRange.textContent = `Showing: ${new Date(xs[rangeStart]).toUTCString().slice(5,22)} → ${new Date(xs[t]).toUTCString().slice(5,22)}`;
    const overviewNote = document.getElementById("workspaceOverviewNote");
    if (overviewNote) overviewNote.innerHTML = `<b>Demo / Scenario only:</b> ${cur.severe_count?.toLocaleString?.() ?? 0} severe and ${cur.degraded_count?.toLocaleString?.() ?? 0} degraded cells at the selected hourly playback point. These values are separate from the quarterly real workflow above.`;
  }
}

function setQuarterlyPriorityView(view) {
  state.quarterlyPriorityView = ["future", "current", "cases"].includes(view) ? view : "future";
  state.quarterlyTriageFilter = "all";
  renderQuarterlyQueue();
}

function quarterlySupportLevel(support, tests) {
  const normalized = String(support || "").toLowerCase();
  if (["limited", "typical", "strong"].includes(normalized)) return normalized;
  if (tests == null) return "limited";
  if (Number(tests) >= 150) return "strong";
  if (Number(tests) >= 20) return "typical";
  return "limited";
}

// Transparent prototype policy: AI ranking and observed status remain unchanged.
// This layer only recommends the next human review pathway using existing
// quarterly evidence. It is configurable and not an industry-validated rule.
function quarterlyTriageRecommendation(row, observed = false) {
  const support = quarterlySupportLevel(row.support, row.tests);
  const recurrence = Number(row.historyRate ?? 0);
  if (observed) {
    if (support === "limited") return {
      key: "verify_first", label: "Verify first",
      reason: "The issue is observed, but quarterly measurement support is limited.",
      next: "Collect recent operator KPI or field measurements before escalation.",
      rule: "Observed issue + limited support",
    };
    return {
      key: "investigate_now", label: "Investigate now",
      reason: `Observed underperformance has ${support} quarterly measurement support.`,
      next: "Begin operator or field validation of the current underperformance.",
      rule: "Observed issue + typical/strong support",
    };
  }
  if (row.tier === "high" && support !== "limited") return {
    key: "investigate_now", label: "Investigate now",
    reason: `High AI priority is supported by ${support} quarterly measurement coverage.`,
    next: "Review the quarterly measurements and model contributors, then begin targeted validation.",
    rule: "High risk + typical/strong support",
  };
  if (row.tier === "elevated" && support === "strong" && recurrence >= 0.5) return {
    key: "investigate_now", label: "Investigate now",
    reason: "Elevated AI priority is supported by strong measurements and recurring historical underperformance.",
    next: "Review recurring performance history and begin targeted operator validation.",
    rule: "Elevated risk + strong support + ≥50% historical recurrence",
  };
  if (["high", "elevated"].includes(row.tier)) return {
    key: "verify_first", label: "Verify first",
    reason: `${row.tier === "high" ? "High" : "Elevated"} AI priority needs additional measurement verification${support === "limited" ? " because support is limited" : " before escalation"}.`,
    next: "Collect additional quarterly, operator KPI or field evidence before escalation.",
    rule: `${row.tier === "high" ? "High" : "Elevated"} risk without sufficient corroboration`,
  };
  return {
    key: "monitor", label: "Monitor",
    reason: "The cell is outside the High and Elevated immediate-review bands.",
    next: "Continue routine quarterly monitoring and review if measurements deteriorate.",
    rule: "Lower risk / routine review",
  };
}

function quarterlyRecordForH3(h3) {
  const c = state.h3IndexOf.get(h3);
  if (c == null) return null;
  return {
    h3, c,
    rank: onsetValue("onset_risk_rank", c),
    status: onsetValue("current_status", c),
    tier: onsetValue("onset_priority_tier", c),
    dlPercentile: onsetValue("dl_percentile_in_quarter", c),
    dl: onsetValue("ookla_dl_mbps", c),
    ul: onsetValue("ookla_ul_mbps", c),
    rtt: onsetValue("ookla_rtt_ms", c),
    population: onsetValue("population_day", c),
    support: onsetValue("measurement_support", c),
    tests: onsetValue("ookla_tests", c),
    historyRate: onsetValue("below_area_norm_rate_to_date", c),
    recentDl: onsetValue("rolling_mean_4q_dl", c),
    previousDl: onsetValue("lag_1q_dl", c),
    driver: onsetValue("risk_driver_1_label", c),
  };
}

function setQuarterlyTriageFilter(value) {
  state.quarterlyTriageFilter = ["all", "investigate_now", "verify_first", "monitor"].includes(value) ? value : "all";
  renderQuarterlyQueue();
}

function createInvestigationCase(h3) {
  const row = quarterlyRecordForH3(h3);
  if (!row) return;
  const observed = row.status === "currently_underperforming";
  const triage = quarterlyTriageRecommendation(row, observed);
  const c = row.c;
  const contributors = [1, 2, 3].map(n => ({
    label: onsetValue(`risk_driver_${n}_label`, c),
    value: onsetValue(`risk_driver_${n}_value`, c),
    family: onsetValue(`risk_driver_${n}_family`, c),
  })).filter(item => item.label);
  const protective = [1, 2].map(n => ({
    label: onsetValue(`protective_${n}_label`, c),
    value: onsetValue(`protective_${n}_value`, c),
    family: onsetValue(`protective_${n}_family`, c),
  })).filter(item => item.label);
  const now = new Date().toISOString();
  state.investigationCases[h3] = {
    schemaVersion: 1,
    caseId: `GQP-${Date.now().toString(36).toUpperCase()}`,
    h3,
    location: cellName(h3),
    createdAt: now,
    snapshotAt: now,
    modelVersion: state.finals?.metadata?.model_version ?? "—",
    observedQuarter: state.finals?.metadata?.predictor_quarter ?? "2026Q1",
    forecastQuarter: state.finals?.metadata?.forecast_quarter ?? "2026Q2",
    observedStatus: observed ? "Underperforming" : row.status === "currently_not_underperforming" ? "Not underperforming" : "Not measured",
    riskBand: row.rank != null ? (ONSET_TIER_LABELS[row.tier] ?? row.tier) : "Not forecast-eligible",
    riskRank: row.rank == null ? null : Math.round(row.rank),
    triage,
    measurements: { downloadMbps: row.dl, uploadMbps: row.ul, latencyMs: row.rtt, downloadPercentile: row.dlPercentile },
    measurementSupport: quarterlySupportLevel(row.support, row.tests),
    ooklaTests: row.tests,
    performanceHistory: { underperformanceRate: row.historyRate, fourQuarterAverageDownloadMbps: row.recentDl, previousQuarterDownloadMbps: row.previousDl },
    contributors,
    protectiveContributors: protective,
    context: {
      populationDay: row.population,
      populationDensityPerKm2: onsetValue("population_density_per_km2", c),
      towerDensityPerKm2: onsetValue("tower_density_per_km2", c),
      builtUpRatio: onsetValue("built_up_ratio", c),
      viirsHumanActivityProxy: onsetValue("viirs_rad_quarter", c),
    },
    workflowStatus: "open",
    verificationDate: "",
    evidenceSource: "",
    nextReviewDate: "",
    notes: "",
    updatedAt: now,
    disclaimer: "Prototype operational handoff. Triage is rule-derived and configurable; model contributors do not establish engineering cause.",
  };
  saveInvestigationCases();
  renderQuarterlyQueue();
}

function updateInvestigationCase(h3, field, value) {
  const record = state.investigationCases[h3];
  if (!record || !["workflowStatus", "verificationDate", "evidenceSource", "nextReviewDate", "notes"].includes(field)) return;
  record[field] = value;
  record.updatedAt = new Date().toISOString();
  if (field === "workflowStatus") {
    if (value === "investigating") state.tracking[h3] = "investigating";
    else if (value === "resolved") state.tracking[h3] = "done";
    else delete state.tracking[h3];
    saveTracking();
  }
  saveInvestigationCases();
  if (field !== "notes") renderQuarterlyQueue();
}

function investigationBriefText(record) {
  const fmt = (value, unit = "") => value == null ? "Not available" : `${Number(value).toFixed(1)}${unit}`;
  const contributors = record.contributors?.length ? record.contributors.map(item => `- ${item.label}: ${formatModelObservedValue(item.label, item.value)}`).join("\n") : "- None available";
  return `GeoQoE Pulse Investigation Brief\nCase: ${record.caseId}\nLocation: ${record.location}\nH3 cell: ${record.h3}\nCreated: ${record.createdAt}\n\nQuarterly context\nObserved ${record.observedQuarter}: ${record.observedStatus}\nForecast ${record.forecastQuarter}: ${record.riskBand}${record.riskRank != null ? ` · rank #${record.riskRank}` : ""}\nSystem triage: ${record.triage.label}\nReason: ${record.triage.reason}\nSuggested next step: ${record.triage.next}\n\nQuarterly measurements\nDownload: ${fmt(record.measurements.downloadMbps, " Mbps")}\nUpload: ${fmt(record.measurements.uploadMbps, " Mbps")}\nLatency: ${fmt(record.measurements.latencyMs, " ms")}\nMeasurement support: ${record.measurementSupport}${record.ooklaTests != null ? ` · ${record.ooklaTests} Ookla tests` : ""}\n\nModel contributors\n${contributors}\n\nHuman verification\nStatus: ${record.workflowStatus}\nEvidence source: ${record.evidenceSource || "Not recorded"}\nVerification date: ${record.verificationDate || "Not recorded"}\nNext review: ${record.nextReviewDate || "Not recorded"}\nNotes: ${record.notes || "None"}\n\nModel version: ${record.modelVersion}\n${record.disclaimer}`;
}

async function copyInvestigationBrief(h3) {
  const record = state.investigationCases[h3];
  if (!record) return;
  const text = investigationBriefText(record);
  try {
    await navigator.clipboard.writeText(text);
    window.alert("Investigation brief copied to the clipboard.");
  } catch {
    const area = document.createElement("textarea");
    area.value = text; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
    window.alert("Investigation brief copied to the clipboard.");
  }
}

function downloadInvestigationCase(h3) {
  const record = state.investigationCases[h3];
  if (!record) return;
  const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = `${record.caseId}-${h3}.json`; link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function printInvestigationBrief(h3) {
  const record = state.investigationCases[h3];
  if (!record) return;
  const popup = window.open("", "_blank", "noopener,noreferrer");
  if (!popup) { window.alert("Allow pop-ups to print the investigation brief."); return; }
  const body = escapeHtml(investigationBriefText(record)).replace(/\n/g, "<br>");
  popup.document.write(`<!doctype html><html><head><title>${escapeHtml(record.caseId)} · GeoQoE Pulse</title><style>body{font:14px/1.55 Arial,sans-serif;max-width:820px;margin:36px auto;padding:0 24px;color:#261c33}h1{color:#68469a}button{margin-top:20px;padding:8px 14px}@media print{button{display:none}}</style></head><body><h1>GeoQoE Pulse Investigation Brief</h1><div>${body}</div><button onclick="window.print()">Print / Save as PDF</button></body></html>`);
  popup.document.close();
}

function renderInvestigationCase(record, compact = false) {
  if (!record) return "";
  const statusOptions = [["open","Open"],["investigating","Investigating"],["confirmed_issue","Confirmed issue"],["not_confirmed","Not confirmed"],["insufficient_evidence","Insufficient evidence"],["monitoring","Monitoring"],["resolved","Resolved"]];
  const evidenceOptions = ["","Operator KPI","Field measurement","Drive test","Customer complaint","Network alarm","Additional Ookla measurement","Desktop review","Other"];
  const statusLabel = escapeHtml(statusOptions.find(item => item[0] === record.workflowStatus)?.[1] || record.workflowStatus);
  const summary = compact
    ? `<summary><span><b>Investigation case</b><em>${escapeHtml(record.caseId)}</em></span><span>Status: <b>${statusLabel}</b><small>Evidence snapshot saved</small></span><strong>View case</strong></summary>`
    : `<summary><span>Investigation case</span><b>${escapeHtml(record.caseId)} · ${statusLabel}</b></summary>`;
  return `<details class="investigation-case ${compact ? "compact-case" : ""}">${summary}<div class="investigation-case-body"><div class="case-snapshot-note"><i class="bi bi-camera"></i><span>Evidence snapshot created ${escapeHtml(new Date(record.createdAt).toLocaleString())}. Later model updates do not alter this saved brief.</span></div><div class="case-field-grid"><label><span>Outcome / workflow status</span><select onchange="updateInvestigationCase('${record.h3}','workflowStatus',this.value)">${statusOptions.map(([value,label]) => `<option value="${value}" ${record.workflowStatus === value ? "selected" : ""}>${label}</option>`).join("")}</select></label><label><span>Evidence source</span><select onchange="updateInvestigationCase('${record.h3}','evidenceSource',this.value)">${evidenceOptions.map(value => `<option value="${escapeHtml(value)}" ${record.evidenceSource === value ? "selected" : ""}>${escapeHtml(value || "Not recorded")}</option>`).join("")}</select></label><label><span>Verification date</span><input type="date" value="${escapeHtml(record.verificationDate)}" onchange="updateInvestigationCase('${record.h3}','verificationDate',this.value)"></label><label><span>Next review date</span><input type="date" value="${escapeHtml(record.nextReviewDate)}" onchange="updateInvestigationCase('${record.h3}','nextReviewDate',this.value)"></label></div><label class="case-notes"><span>Reviewer notes</span><textarea rows="3" placeholder="Record operator or field findings…" onchange="updateInvestigationCase('${record.h3}','notes',this.value)">${escapeHtml(record.notes)}</textarea></label><div class="case-actions"><button onclick="copyInvestigationBrief('${record.h3}')"><i class="bi bi-clipboard"></i> Copy summary</button><button onclick="downloadInvestigationCase('${record.h3}')"><i class="bi bi-download"></i> JSON</button><button onclick="printInvestigationBrief('${record.h3}')"><i class="bi bi-printer"></i> Print / PDF</button></div><p class="case-disclaimer">Human-entered outcomes are stored locally and do not automatically retrain or update the model.</p></div></details>`;
}

function renderQuarterlyQueue() {
  const rows = state.panel.h3_ids.map((h3, c) => ({
    h3, c,
    rank: onsetValue("onset_risk_rank", c),
    status: onsetValue("current_status", c),
    tier: onsetValue("onset_priority_tier", c),
    dlPercentile: onsetValue("dl_percentile_in_quarter", c),
    dl: onsetValue("ookla_dl_mbps", c),
    ul: onsetValue("ookla_ul_mbps", c),
    rtt: onsetValue("ookla_rtt_ms", c),
    population: onsetValue("population_day", c),
    support: onsetValue("measurement_support", c),
    tests: onsetValue("ookla_tests", c),
    historyRate: onsetValue("below_area_norm_rate_to_date", c),
    driver: onsetValue("risk_driver_1_label", c)
  }));
  const issues = rows.filter(r => r.status === "currently_underperforming").sort((a, b) =>
    (a.dlPercentile ?? Infinity) - (b.dlPercentile ?? Infinity) ||
    (b.population ?? 0) - (a.population ?? 0)
  );
  const forecast = rows.filter(r => r.rank != null && ["high", "elevated"].includes(r.tier)).sort((a,b) => a.rank - b.rank);
  const highCount = forecast.filter(r => r.tier === "high").length;
  const elevatedCount = forecast.filter(r => r.tier === "elevated").length;
  const view = ["current", "cases"].includes(state.quarterlyPriorityView) ? state.quarterlyPriorityView : "future";
  const investigationCases = Object.values(state.investigationCases || {}).sort((a, b) => {
    const order = { investigating: 0, open: 1, insufficient_evidence: 2, monitoring: 3, confirmed_issue: 4, not_confirmed: 5, resolved: 6 };
    return (order[a.workflowStatus] ?? 9) - (order[b.workflowStatus] ?? 9) || String(b.updatedAt || b.createdAt).localeCompare(String(a.updatedAt || a.createdAt));
  });
  const fmt = (value, unit, decimals = 1) => value == null ? "Not available" : `${Number(value).toFixed(decimals)}${unit ? ` ${unit}` : ""}`;
  const metricTone = (kind, value) => {
    if (value == null) return "metric-unknown";
    if (kind === "download") return value < 25 ? "metric-critical" : value < 100 ? "metric-watch" : "metric-strong";
    if (kind === "upload") return value < 5 ? "metric-critical" : value < 20 ? "metric-watch" : "metric-strong";
    if (kind === "latency") return value > 75 ? "metric-critical" : value > 30 ? "metric-watch" : "metric-strong";
    return "metric-unknown";
  };
  const card = (r, observed) => {
    const tier = observed ? "observed" : (r.tier || "lower");
    const priority = observed ? "P1 · Observed issue · 2026Q1" : tier === "high" ? "High AI risk" : "Elevated AI risk";
    const evidence = observed ? "Observed quarterly underperformance" : (r.driver || "Review the model contributors");
    const why = observed
      ? `Already underperforming in ${state.finals.metadata.predictor_quarter}${r.dlPercentile != null ? `; download is at the ${Number(r.dlPercentile).toFixed(1)} percentile` : ""}.`
      : `${ONSET_TIER_LABELS[tier] ?? "Emerging risk"} · rank #${Math.round(r.rank)} among evaluable currently-normal cells.`;
    const triage = quarterlyTriageRecommendation(r, observed);
    const savedCase = state.investigationCases[r.h3];
    const queueMarker = observed ? "OBS" : `#${Math.round(r.rank)}`;
    const conciseReason = triage.key === "verify_first" && quarterlySupportLevel(r.support, r.tests) === "limited"
      ? "Limited quarterly measurement support"
      : triage.reason;
    const supportingEvidence = `Quarterly download ${fmt(r.dl, "Mbps")}, upload ${fmt(r.ul, "Mbps")} and latency ${fmt(r.rtt, "ms")}.`;
    return `<article class="quarterly-queue-item queue-${tier} ${observed ? "" : "future-risk-card"}">
      <div class="queue-item-head"><span class="queue-card-rank">${queueMarker}</span><div class="queue-card-identity"><h3>${cellName(r.h3)}</h3><span class="quarterly-priority-badge">${priority}</span></div><button class="btn btn-sm btn-outline-primary" onclick="flyToCell('${r.h3}'); setInsightTab('${observed ? "overview" : "ai"}')">Inspect cell</button></div>
      <div class="quarterly-measurements"><span class="${metricTone("download", r.dl)}"><small>Download</small><b>${fmt(r.dl, "Mbps")}</b></span><span class="${metricTone("upload", r.ul)}"><small>Upload</small><b>${fmt(r.ul, "Mbps")}</b></span><span class="${metricTone("latency", r.rtt)}"><small>Latency</small><b>${fmt(r.rtt, "ms")}</b></span></div>
      <section class="triage-recommendation triage-${triage.key}"><div class="triage-pathway"><span>Recommended pathway</span><b>${triage.label}</b></div><div class="triage-summary-lines"><span>Reason</span><b>${conciseReason}</b><span>Next check</span><b>${triage.next}</b></div></section>
      <p class="quarterly-why"><span>Why prioritised</span><b>${why}</b></p>
      <details class="queue-evidence-detail"><summary>Evidence &amp; recommendation details</summary><div class="queue-detail-list"><div><span>${observed ? "Evidence state" : "Top model contributor"}</span><b>${evidence}</b></div><div><span>Quarterly measurement support</span><b>${r.support ?? "Not reported"}${r.tests != null ? ` · ${Number(r.tests).toLocaleString()} tests` : ""}</b></div><div><span>Population context</span><b>${r.population != null ? Math.round(r.population).toLocaleString() : "Not available"}</b></div><div><span>Supporting evidence</span><b>${supportingEvidence}</b></div><div><span>Recommendation rationale</span><b>${triage.rule}</b></div></div><p class="queue-policy-note">Relative prioritisation and transparent triage guidance; neither establishes an engineering cause or changes the AI rank.</p></details>
      ${savedCase ? renderInvestigationCase(savedCase, !observed) : `<div class="quarterly-handoff"><button class="btn btn-sm btn-outline-primary" onclick="createInvestigationCase('${r.h3}')"><i class="bi bi-file-earmark-plus"></i>Create investigation case</button><small>Save an evidence snapshot for handoff and outcome tracking</small></div>`}
    </article>`;
  };
  const switcher = `<div class="quarterly-view-toggle" role="group" aria-label="Choose investigation queue">
    <button type="button" class="future ${view === "future" ? "active" : ""}" aria-pressed="${view === "future"}" onclick="setQuarterlyPriorityView('future')"><span>Future risk shortlist</span><b>${forecast.length.toLocaleString()}</b></button>
    <button type="button" class="current ${view === "current" ? "active" : ""}" aria-pressed="${view === "current"}" onclick="setQuarterlyPriorityView('current')"><span>Observed issues · 2026Q1</span><b>${issues.length.toLocaleString()}</b></button>
    <button type="button" class="cases ${view === "cases" ? "active" : ""}" aria-pressed="${view === "cases"}" onclick="setQuarterlyPriorityView('cases')"><span>Investigation cases</span><b>${investigationCases.length.toLocaleString()}</b></button>
  </div>`;
  const sourceRows = view === "current" ? issues : view === "future" ? forecast : [];
  const triageCounts = { investigate_now: 0, verify_first: 0, monitor: 0 };
  sourceRows.forEach(row => { triageCounts[quarterlyTriageRecommendation(row, view === "current").key]++; });
  const triageFilters = `<div class="quarterly-triage-toolbar"><span>Recommended pathway</span><div><button class="${state.quarterlyTriageFilter === "all" ? "active" : ""}" onclick="setQuarterlyTriageFilter('all')">All <b>${sourceRows.length}</b></button><button class="triage-investigate ${state.quarterlyTriageFilter === "investigate_now" ? "active" : ""}" onclick="setQuarterlyTriageFilter('investigate_now')">Investigate now <b>${triageCounts.investigate_now}</b></button><button class="triage-verify ${state.quarterlyTriageFilter === "verify_first" ? "active" : ""}" onclick="setQuarterlyTriageFilter('verify_first')">Verify first <b>${triageCounts.verify_first}</b></button><button class="triage-monitor ${state.quarterlyTriageFilter === "monitor" ? "active" : ""}" onclick="setQuarterlyTriageFilter('monitor')">Monitor <b>${triageCounts.monitor}</b></button></div></div><p class="triage-scope-note"><i class="bi bi-info-circle"></i>Triage is a configurable, rule-based prototype layer. AI risk rank remains unchanged.</p>`;
  const filteredIssues = issues.filter(row => state.quarterlyTriageFilter === "all" || quarterlyTriageRecommendation(row, true).key === state.quarterlyTriageFilter);
  const filteredForecast = forecast.filter(row => state.quarterlyTriageFilter === "all" || quarterlyTriageRecommendation(row, false).key === state.quarterlyTriageFilter);
  const currentQueue = `<div class="queue-section-heading current"><span>OBSERVED ISSUES · 2026Q1</span><b>Validate existing underperformance first</b></div>
    <p class="panel-hint">Ordered by lowest quarterly download percentile, then population context. These cells are not forecast-ranked.</p>
    ${filteredIssues.length ? filteredIssues.slice(0,5).map(r => card(r,true)).join("") : `<div class="queue-empty"><b>No matching observed issues</b><span>Choose another triage pathway.</span></div>`}
    ${filteredIssues.length > 5 ? `<details class="quarterly-more"><summary>More current issues (next 15)</summary>${filteredIssues.slice(5,20).map(r => card(r,true)).join("")}</details>` : ""}`;
  const futureQueue = `<div class="queue-section-heading future"><span>FUTURE RISK SHORTLIST</span><b>${highCount.toLocaleString()} High · ${elevatedCount.toLocaleString()} Elevated</b></div>
    <p class="panel-hint">Only High and Elevated AI bands are included here. Use Anticipate to explore the complete ${rows.filter(r => r.rank != null).length.toLocaleString()}-cell forecast ranking.</p>
    ${filteredForecast.length ? filteredForecast.slice(0,5).map(r => card(r,false)).join("") : `<div class="queue-empty"><b>No matching future-risk cells</b><span>Choose another triage pathway.</span></div>`}
    ${filteredForecast.length > 5 ? `<details class="quarterly-more"><summary>More future priorities (next 15)</summary>${filteredForecast.slice(5,20).map(r => card(r,false)).join("")}</details>` : ""}`;
  const statusLabel = value => ({ open: "Open", investigating: "Investigating", confirmed_issue: "Confirmed issue", not_confirmed: "Not confirmed", insufficient_evidence: "Insufficient evidence", monitoring: "Monitoring", resolved: "Resolved" }[value] || value);
  const caseQueue = `<div class="queue-section-heading cases"><span>INVESTIGATION CASES</span><b>${investigationCases.filter(record => record.workflowStatus !== "resolved").length.toLocaleString()} active · ${investigationCases.filter(record => record.workflowStatus === "resolved").length.toLocaleString()} resolved</b></div>
    <p class="panel-hint">Saved cases are collected here so reviewers can resume investigations without finding the cell in either source queue.</p>
    ${investigationCases.length ? investigationCases.map(record => `<article class="case-queue-card status-${escapeHtml(record.workflowStatus)}"><div class="case-queue-head"><div><span>${escapeHtml(record.caseId)}</span><h3>${escapeHtml(record.location)}</h3><small>${record.riskRank != null ? `${escapeHtml(record.riskBand)} · AI rank #${record.riskRank}` : `Observed issue · ${escapeHtml(record.observedQuarter)}`}</small></div><div class="case-queue-actions"><b>${escapeHtml(statusLabel(record.workflowStatus))}</b><button class="btn btn-sm btn-outline-primary" onclick="flyToCell('${record.h3}'); setInsightTab('${record.riskRank != null ? "ai" : "overview"}')">Inspect cell</button></div></div><div class="case-queue-summary"><span><small>Recommended pathway</small><b>${escapeHtml(record.triage?.label || "Review")}</b></span><span><small>Measurement support</small><b>${escapeHtml(record.measurementSupport || "Not recorded")}</b></span><span><small>Last updated</small><b>${escapeHtml(new Date(record.updatedAt || record.createdAt).toLocaleDateString())}</b></span></div>${renderInvestigationCase(record, true)}</article>`).join("") : `<div class="queue-empty"><b>No investigation cases yet</b><span>Create a case from a Future risk or Observed issue card. It will appear here automatically.</span></div>`}`;
  document.getElementById("quarterlyPriorityList").innerHTML = state.finals
    ? `${switcher}${view === "cases" ? caseQueue : `${triageFilters}${view === "current" ? currentQueue : futureQueue}`}`
    : `<p>Quarterly forecast data is unavailable.</p>`;
}

function openSelectedCellReveal() {
  if (!state.selectedH3) return;
  state.insightTab = "overview";
  renderSelectedAreaInsights();
  setWorkspaceTool("cell");
}

function setWorkspaceTool(tool) {
  const labTools = new Set(["lab", "pulse", "compare", "hotspots", "spatial", "timeseries", "whatif", "rootcause", "actions", "comparison"]);
  const inLab = labTools.has(tool) || (tool === "cell" && document.body.classList.contains("in-scenario-lab"));
  document.body.classList.toggle("in-scenario-lab", inLab);
  document.querySelectorAll(".scenario-back-link").forEach(el => el.remove());
  if (!inLab) {
    if (state.playing) document.getElementById("playBtn").click();
    if (state.whatIfEnabled) document.getElementById("whatIfEnabled").click();
    if (state.weatherEnabled) document.getElementById("toggleWeather").click();
    state.analysisContext = "quarterly";
  }
  if (tool === "explain") {
    if (!state.selectedH3) {
      const first = state.panel.h3_ids.find((h3, c) => onsetValue("onset_risk_rank", c) === 1);
      if (first) flyToCell(first);
    }
    state.insightTab = "ai";
    renderSelectedAreaInsights();
  }
  if (tool === "overview" && state.selectedH3) {
    state.insightTab = "overview";
    renderSelectedAreaInsights();
    syncWorkspaceSelection();
  }
  if (tool === "prioritise") renderQuarterlyQueue();
  if (tool === "pulse") document.getElementById("hourlyPulsePanel").appendChild(document.querySelector(".workspace-hourly-secondary"));

  state.workspaceTool = tool;
  if (tool !== "more") document.getElementById("moreToolsPopover")?.classList.add("d-none");
  document.querySelectorAll(".workspace-tool").forEach(btn => btn.classList.toggle("active", tool === "cell" ? btn.dataset.workspaceTool === "overview" : btn.dataset.workspaceTool === tool));
  document.body.className = document.body.className.replace(/workspace-tool-[a-z]+/g, "").trim();
  document.body.classList.add(`workspace-tool-${tool}`);
  const workspacePanels = document.querySelectorAll(".workspace-analysis-panel");
  workspacePanels.forEach(panel => panel.style.setProperty("display", "none", "important"));
  const panelByTool = {
    lab: "scenarioLabPanel",
    pulse: "hourlyPulsePanel",
    explain: "selectedAreaPanel",
    prioritise: "quarterlyPriorityPanel",
    overview: "workspaceOverviewPanel",
    cell: "selectedAreaPanel",
    compare: "comparePanel",
    hotspots: "hotspotPanel",
    ai: "aiModelPanel",
    rootcause: "rootCausePanel",
    actions: null,
    trust: "trustPanel",
    more: "workspaceMorePanel",
    timeseries: "timeseriesPanel",
    comparison: "detectionComparisonPanel",
    whatif: "workspaceMorePanel",
    spatial: "hotspotPanel",
  };
  const activePanel = tool === "actions"
    ? document.querySelector(".workspace-actions-host")
    : document.getElementById(panelByTool[tool]);
  const analysisStage = document.getElementById("analysisStage");
  if (activePanel && analysisStage && activePanel.parentElement !== analysisStage) analysisStage.appendChild(activePanel);
  activePanel?.style.setProperty("display", "block", "important");
  if (analysisStage) analysisStage.scrollTop = 0;
  if (inLab && tool !== "lab" && activePanel) {
    const back = document.createElement("button");
    back.className = "scenario-back-link btn btn-sm btn-outline-secondary";
    back.textContent = "← Scenario Lab";
    back.onclick = () => setWorkspaceTool("lab");
    activePanel.prepend(back);
  }
  const compare = tool === "compare";
  state.compareMode = compare;
  const compareBtn = document.getElementById("compareToggleBtn");
  if (compareBtn) {
    compareBtn.classList.toggle("btn-primary", compare);
    compareBtn.classList.toggle("btn-outline-primary", !compare);
    compareBtn.textContent = compare ? "Click hexes to compare…" : "Compare cells";
  }
  document.getElementById("comparePanel")?.classList.toggle("d-none", !compare);
  if (compare) renderComparePanel();
  const quarterlyTools = new Set(["overview", "ai", "explain", "prioritise"]);
  const scenarioTools = new Set(["lab", "pulse", "compare", "comparison", "hotspots", "spatial", "rootcause", "actions", "timeseries", "whatif"]);
  if (quarterlyTools.has(tool)) state.analysisContext = "quarterly";
  if (scenarioTools.has(tool)) state.analysisContext = "scenario";
  document.body.classList.toggle("quarterly-context", state.analysisContext === "quarterly");
  document.body.classList.toggle("scenario-context", state.analysisContext === "scenario");
  const mapContext = document.getElementById("mapDataContextBadge");
  if (mapContext) mapContext.textContent = state.analysisContext === "quarterly" ? "Observed + AI · quarterly" : "Demo / Scenario · hourly";
  const timelineHint = document.getElementById("scenarioTimelineHint");
  if (timelineHint) timelineHint.textContent = state.analysisContext === "quarterly" ? "Demo only · does not change quarterly forecast" : "Active · controls hourly scenario views";
  syncMapSelectedBar();
  syncMapWorkspaceInsight();
  if (!inLab || tool === "overview") setMapLens("onset");
  if (["lab", "pulse", "compare"].includes(tool)) setMapLens("qoe");
  document.querySelector('[data-workspace-tool="lab"]')?.classList.toggle("active", inLab);
  if (tool === "hotspots" || tool === "spatial") setMapLens("hotspot");
  if (["rootcause", "actions", "timeseries", "whatif"].includes(tool)) setMapLens("qoe");
  if (tool === "ai" || tool === "comparison") setMapLens("onset");
  if (tool === "ai") renderAiPanel();
  document.querySelectorAll(".workspace-tool").forEach(btn => {
    if (btn.classList.contains("active")) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
  const whatIfBox = document.querySelector(".whatif-box");
  if (whatIfBox && !state.whatIfParent) state.whatIfParent = whatIfBox.parentElement;
  if (tool === "whatif") {
    document.getElementById("workspaceMorePanel")?.appendChild(whatIfBox);
  } else if (whatIfBox && state.whatIfParent) {
    state.whatIfParent.appendChild(whatIfBox);
  }
  if (window.gsap && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    if (activePanel) gsap.fromTo(activePanel, { autoAlpha: 0, x: 8 }, { autoAlpha: 1, x: 0, duration: .24, ease: "power2.out", overwrite: "auto" });
  }
  window.setTimeout(() => {
    state.map?.invalidateSize();
    activePanel?.querySelectorAll(".js-plotly-plot").forEach(plot => window.Plotly?.Plots.resize(plot));
    syncOperatorChartTheme();
  }, 80);
}

function showLatestWorkspacePulse() {
  state.currentT = state.panel.num_timestamps - 1;
  document.getElementById("timeSlider").value = state.currentT;
  renderAll();
}

function setMapLens(mode) {
  if (mode === "onset" && !state.onset) return;
  state.colorBy = mode;
  document.getElementById("colorBySelect").value = mode;
  syncPrimaryMapControls();
  renderMapColors();
  renderLegend();
}

function setViewMode(mode) {
  setAppPage(mode === "command" ? "command" : "analyst");
}

function setAppPage(page) {
  const workspaceByPage = { command: "overview", map: "overview", hotspots: "hotspots", ai: "ai", interventions: "actions", analyst: "rootcause" };
  const workspaceTool = workspaceByPage[page] || "overview";
  state.appPage = "command";
  state.viewMode = "command";
  document.body.className = document.body.className.replace(/app-page-[a-z]+/g, "").trim();
  document.body.classList.add("app-page-command");
  document.getElementById("viewModeOverviewBtn")?.classList.add("active");
  document.getElementById("viewModeFullBtn")?.classList.remove("active");
  document.querySelectorAll(".full-dashboard-only").forEach(elm => elm.classList.remove("d-none"));
  document.getElementById("mainWorkspace").classList.add("command-mode");
  document.getElementById("mainWorkspace").classList.remove("analyst-mode");
  document.querySelectorAll("[data-page]").forEach(btn => btn.classList.toggle("active", btn.dataset.page === page));
  renderExecOverview();
  window.setTimeout(() => state.map?.invalidateSize(), 220);
  setWorkspaceTool(workspaceTool);
}

function setHotspotMapView() {
  state.colorBy = "hotspot";
  document.getElementById("colorBySelect").value = "hotspot";
  syncPrimaryMapControls();
  renderMapColors(); renderLegend();
  setAppPage("map");
}

function setMoranMapView() {
  setMapLens("moran");
  const advanced = document.querySelector(".map-overflow-popover");
  if (advanced) advanced.open = true;
}

function setWorkspaceWidth(mode) {
  const workspace = document.getElementById("mainWorkspace");
  workspace?.style.removeProperty("--analysis-ratio");
  workspace?.style.removeProperty("--analysis-size");
  workspace?.style.removeProperty("--analysis-offset");
  document.body.classList.remove("workspace-wide-analysis", "workspace-wide-map");
  if (mode === "analysis") document.body.classList.add("workspace-wide-analysis");
  if (mode === "map") document.body.classList.add("workspace-wide-map");
  window.setTimeout(() => {
    state.map?.invalidateSize();
    document.querySelectorAll(".js-plotly-plot").forEach(plot => window.Plotly?.Plots.resize(plot));
  }, 120);
}

function initWorkspaceResize() {
  const workspace = document.getElementById("mainWorkspace");
  const handle = document.getElementById("workspaceResizeHandle");
  if (!workspace || !handle) return;
  const applyRatio = clientX => {
    if (window.innerWidth <= 760) return;
    const rect = workspace.getBoundingClientRect();
    const railWidth = document.querySelector(".workspace-tool-rail").getBoundingClientRect().width;
    const usableWidth = rect.width - railWidth;
    const ratio = Math.min(.58, Math.max(.38, (clientX - rect.left - railWidth) / usableWidth));
    workspace.style.setProperty("--analysis-ratio", ratio.toFixed(4));
    workspace.style.setProperty("--analysis-size", `${(ratio * 100).toFixed(3)}%`);
    workspace.style.setProperty("--analysis-offset", `${(ratio * railWidth).toFixed(2)}px`);
    document.body.classList.remove("workspace-wide-analysis", "workspace-wide-map");
    state.map?.invalidateSize();
  };
  handle.addEventListener("pointerdown", event => {
    if (window.innerWidth <= 760) return;
    event.preventDefault();
    handle.classList.add("is-dragging");
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener("pointermove", event => {
    if (handle.hasPointerCapture(event.pointerId)) applyRatio(event.clientX);
  });
  const stop = event => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    handle.releasePointerCapture(event.pointerId);
    handle.classList.remove("is-dragging");
    window.setTimeout(() => state.map?.invalidateSize(), 80);
  };
  handle.addEventListener("pointerup", stop);
  handle.addEventListener("pointercancel", stop);
  handle.addEventListener("keydown", event => {
    if (window.innerWidth <= 760 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = Number.parseFloat(getComputedStyle(workspace).getPropertyValue("--analysis-ratio")) || .47;
    const target = event.key === "Home" ? .38 : event.key === "End" ? .58 : current + (event.key === "ArrowRight" ? .02 : -.02);
    const rect = workspace.getBoundingClientRect();
    const railWidth = document.querySelector(".workspace-tool-rail").getBoundingClientRect().width;
    applyRatio(rect.left + railWidth + (rect.width - railWidth) * target);
  });
}

function setAiWorkspaceTab(tab) {
  document.querySelectorAll(".ai-workspace-tab").forEach(btn => btn.classList.toggle("active", btn.dataset.aiTab === tab));
  document.body.classList.remove("ai-tab-risk", "ai-tab-explain", "ai-tab-comparison", "ai-tab-trust");
  document.body.classList.add(`ai-tab-${tab}`);
  const explainPanel = document.getElementById("aiExplainabilityPanel");
  if (tab === "explain") {
    state.insightTab = "ai";
    renderSelectedAreaInsights();
    document.getElementById("aiExplainabilityBody").innerHTML = document.getElementById("selectedAreaBody").innerHTML;
    explainPanel.classList.remove("d-none");
  } else {
    explainPanel.classList.add("d-none");
  }
}

function setAnalystTab(tab) {
  state.analystTab = tab;
  document.querySelectorAll(".analyst-tab").forEach(btn => btn.classList.toggle("active", btn.dataset.analystTab === tab));
  document.body.classList.remove("analyst-tab-map", "analyst-tab-trends", "analyst-tab-compare", "analyst-tab-model", "analyst-tab-trust");
  document.body.classList.add(`analyst-tab-${tab}`);
  if (tab === "compare") document.getElementById("comparePanel")?.classList.remove("d-none");
  if (tab === "map") window.setTimeout(() => state.map?.invalidateSize(), 80);
}

function syncPrimaryMapControls() {
  const control = document.querySelector(".map-primary-controls");
  if (control) control.dataset.activeLens = state.colorBy;
  document.querySelectorAll(".map-mode-btn").forEach(btn => {
    const selected = btn.dataset.mapMode === state.colorBy;
    btn.classList.toggle("active", selected);
    btn.setAttribute("aria-pressed", String(selected));
  });
  document.querySelectorAll(".quarterly-map-view-btn").forEach(btn => {
    const selected = btn.dataset.quarterlyView === state.quarterlyMapView;
    btn.classList.toggle("active", selected);
    btn.setAttribute("aria-pressed", String(selected));
  });
  document.getElementById("quarterlyMapViews")?.classList.toggle("d-none", state.colorBy !== "onset");
  const lensHud = document.getElementById("mapLensHud");
  if (lensHud) lensHud.textContent = state.colorBy === "onset" ? ({ observed: "Observed Q1", forecast: "Forecast Q2", combined: "Observed Q1 + Forecast Q2" }[state.quarterlyMapView]) : ({ qoe: "QoE", demand: "Scenario Pressure", hotspot: "Hotspots", moran: "Moran's I", risk: "Historical outcome", static_ai: "AI static DL", comparison: "AI comparison" })[state.colorBy] || "Map layer";
  const contextBadge = document.getElementById("mapDataContextBadge");
  if (contextBadge) {
    const quarterly = state.colorBy === "onset";
    contextBadge.textContent = quarterly ? (state.finalsActualRevealed && state.quarterlyMapView === "combined" ? "Held-out outcome · quarterly" : ({ observed: "Observed status · 2026Q1", forecast: "AI forecast · 2026Q2", combined: "Observed 2026Q1 + forecast 2026Q2" }[state.quarterlyMapView])) : "Demo / Scenario · hourly";
    contextBadge.classList.toggle("ai", quarterly);
    contextBadge.classList.toggle("sim", !quarterly);
  }
  const mapHud = document.querySelector(".map-workspace-hud");
  if (mapHud) mapHud.dataset.context = state.colorBy === "onset" ? ({ observed: "Observed status · 2026Q1", forecast: "AI forecast · 2026Q2", combined: state.finalsActualRevealed ? "Held-out outcome · quarterly" : "Observed 2026Q1 + forecast 2026Q2" }[state.quarterlyMapView]) : "Scenario playback · hourly";
}

function setQuarterlyMapView(view) {
  state.quarterlyMapView = ["observed", "forecast", "combined"].includes(view) ? view : "combined";
  state.colorBy = "onset";
  document.getElementById("colorBySelect").value = "onset";
  syncPrimaryMapControls();
  renderMapColors();
  renderLegend();
}

function renderPriorityActionCentre() {
  const el = document.getElementById("priorityActionList");
  if (!el) return;
  const candidates = (state.lastCandidates || []).slice(0, 3);
  if (!candidates.length) {
    el.innerHTML = `<p class="text-summary">No degraded cells match the current filters at this hour.</p>`;
    return;
  }
  el.innerHTML = candidates.map((cand, index) => {
    const signalScores = causeScoresForCell(state.currentT, cand.c, cand.h3);
    const signals = triggeredCauses(signalScores);
    const cause = signals[0] || "none";
    const support = scenarioSupport(cellValue("confidence", state.currentT, cand.c));
    const onsetIndex = state.h3IndexOf.get(cand.h3);
    const aiRank = state.onset ? onsetValue("onset_risk_rank", onsetIndex) : null;
    const severity = cand.cls === 2 ? "Critical" : "High";
    return `<article class="priority-action-item priority-${severity.toLowerCase()}">
      <div class="priority-kicker">${severity} priority</div>
      <h3>${cellName(cand.h3)}</h3>
      <div class="priority-metrics"><span>Scenario QoE <b>${cand.qoe?.toFixed(0) ?? "-"}</b></span><span>Scenario pressure <b>${cand.demand?.toFixed(0) ?? "-"}</b></span>${aiRank != null ? `<span class="ai-text">Quarterly AI rank <b>#${Math.round(aiRank)}</b></span>` : ""}</div>
      <p><b>Why flagged:</b> ${cand.cls === 2 ? "Severe service degradation" : "Degraded service quality"} with priority score from the existing investigation queue.</p>
      <p><b>Primary investigation hypothesis:</b> <span class="cheat-badge heuristic">DEMO / SCENARIO</span> ${CAUSE_LABELS[cause]}</p>
      ${signals.length > 1 ? `<p><b>Supporting signals:</b> ${signals.slice(1).map(key => CAUSE_LABELS[key]).join(" · ")}</p>` : ""}
      <p><b>Scenario evidence coverage:</b> ${support.label}</p>
      <p><b>Investigation option:</b> ${ACTIONS[cause]}</p>
      <button class="btn btn-sm btn-outline-primary" onclick="flyToCell('${cand.h3}')">View on map</button>
    </article>`;
  }).join("");
}

function renderSelectedAreaInsights() {
  const el = document.getElementById("selectedAreaBody");
  if (!el || !state.panel) return;
  const h3 = state.selectedH3;
  if (!h3) {
    el.innerHTML = `<div class="cell-intel-empty"><i class="bi bi-hexagon"></i><h3>Select a cell on the map</h3><p>Choose a hexagon to inspect its status, AI priority, evidence and recommended investigation.</p></div>`;
    return;
  }
  const c = state.h3IndexOf.get(h3);
  if (c == null) return;
  const t = state.currentT;
  const metrics = effectiveMetrics(t, c);
  const props = state.gridPropsByH3.get(h3) || {};
  const investigationScores = causeScoresForCell(t, c, h3);
  const cellSignals = triggeredCauses(investigationScores);
  const cause = cellSignals[0] || "none";
  const cellSupport = scenarioSupport(cellValue("confidence", t, c));
  const aiRank = state.onset ? onsetValue("onset_risk_rank", c) : null;
  const aiScore = state.onset ? onsetValue("next_hour_onset_risk_score", c) : null;
  const aiTier = state.onset ? onsetValue("onset_priority_tier", c) : null;
  const aiStatus = state.onset ? onsetValue("onset_operational_status", c) : null;
  const quarterlyStatus = onsetValue("current_status", c);
  const quarterlyStatusLabel = quarterlyStatus === "currently_underperforming" ? "Underperforming" : quarterlyStatus === "currently_not_underperforming" ? "Not underperforming" : "Not measured";
  const quarterlyDl = onsetValue("ookla_dl_mbps", c);
  const quarterlyUl = onsetValue("ookla_ul_mbps", c);
  const quarterlyRtt = onsetValue("ookla_rtt_ms", c);
  const heldOutResult = onsetValue("time_machine_result", c);
  const status = cellValue("degradation_class", t, c);
  const statusLabel = ["Normal", "Degraded", "Severe"][status] || "Unknown";
  const explanation = explainCell(c);
  const priorQoe = t > 0 ? cellValue("qoe", t - 1, c) : null;
  const qoeDelta = metrics.qoe != null && priorQoe != null ? metrics.qoe - priorQoe : null;
  const hotspot = state.panel.hotspot_code_labels?.[cellValue("hotspot_code", t, c)] || "Not classified";
  const demandLabel = metrics.demand == null ? "Not available" : metrics.demand >= 15 ? "High" : metrics.demand >= 11 ? "Medium" : "Normal";
  const safe = value => value == null || value === "" ? "—" : value;
  const metric = (label, value, tone = "") => `<div class="cell-intel-metric ${tone}"><span>${label}</span><b>${safe(value)}</b></div>`;
  const metricTone = (kind, value) => {
    if (value == null) return "";
    if (kind === "download" && quarterlyStatus === "currently_underperforming") return "quality-poor";
    if (kind === "download") return value < 25 ? "quality-poor" : value < 100 ? "quality-watch" : "quality-good";
    if (kind === "upload") return value < 5 ? "quality-poor" : value < 20 ? "quality-watch" : "quality-good";
    if (kind === "latency") return value > 75 ? "quality-poor" : value > 30 ? "quality-watch" : "quality-good";
    if (kind === "qoe") return value < 50 ? "quality-poor" : value < 70 ? "quality-watch" : "quality-good";
    if (kind === "jitter") return value > 30 ? "quality-poor" : value > 15 ? "quality-watch" : "quality-good";
    if (kind === "loss") return value > 2 ? "quality-poor" : value > 1 ? "quality-watch" : "quality-good";
    if (kind === "demand") return value >= 15 ? "quality-poor" : value >= 11 ? "quality-watch" : "quality-good";
    return "";
  };
  const evidence = (label, value, tone = "") => value == null || value === "" ? "" : `<li class="${tone}"><span>${label}</span><b>${value}</b></li>`;
  const dataRow = (label, value, note = "", tone = "") => value == null || value === "" ? "" : `<div class="cell-data-row ${tone}"><span>${label}</span><b>${value}</b>${note ? `<small>${note}</small>` : ""}</div>`;
  const timestampLabel = state.panel.timestamps_iso[t] ? new Date(state.panel.timestamps_iso[t]).toLocaleString(undefined, { weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "Scenario hour unavailable";
  const qoeSamples = Array.from({ length: Math.min(24, t + 1) }, (_, i) => {
    const index = t - Math.min(23, t) + i;
    return { value: cellValue("qoe", index, c), timestamp: state.panel.timestamps_iso[index] };
  }).filter(sample => sample.value != null);
  const qoeHistory = qoeSamples.map(sample => sample.value);
  const pulse = qoeHistory.length > 1 ? (() => {
    const low = Math.min(...qoeHistory), high = Math.max(...qoeHistory), span = Math.max(1, high - low);
    const first = qoeHistory[0], last = qoeHistory[qoeHistory.length - 1], delta = last - first;
    const chartWidth = Math.max(260, qoeHistory.length * 34 + 48);
    const plotRight = chartWidth - 18;
    const coords = qoeHistory.map((v, i) => ({ x: 34 + i * ((plotRight - 34) / (qoeHistory.length - 1)), y: 62 - ((v - low) / span) * 42 }));
    const points = coords.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
    const areaPoints = `34,67 ${points} ${plotRight},67`;
    const trend = Math.abs(delta) < .05 ? "Stable" : delta > 0 ? "Improving" : "Declining";
    const trendClass = Math.abs(delta) < .05 ? "stable" : delta > 0 ? "up" : "down";
    const timeLabels = qoeSamples.map((sample, i) => {
      const date = sample.timestamp ? new Date(sample.timestamp) : null;
      return {
        full: date ? date.toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : `Hour ${i + 1}`,
        short: date ? date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }) : `H${i + 1}`,
      };
    });
    const tickEvery = qoeHistory.length > 16 ? 4 : qoeHistory.length > 8 ? 2 : 1;
    const timeTicks = coords.map((p, i) => (i % tickEvery === 0 || i === coords.length - 1)
      ? `<line class="cell-pulse-x-mark" x1="${p.x.toFixed(1)}" y1="68" x2="${p.x.toFixed(1)}" y2="72"/><text class="cell-pulse-axis cell-pulse-x-tick" x="${p.x.toFixed(1)}" y="81" text-anchor="middle">${timeLabels[i].short}</text>`
      : "").join("");
    return `<div class="cell-pulse">
      <div class="cell-pulse-head"><div><span>Cell Pulse</span><small>Demo / Scenario · recent hourly QoE</small></div><div class="cell-pulse-trend ${trendClass}"><b>${trend}</b><small>${delta >= 0 ? "+" : ""}${delta.toFixed(1)} pts</small></div></div>
      <div class="cell-pulse-chart"><div class="cell-pulse-y-label">QoE score</div><div class="cell-pulse-scroll" style="--cell-pulse-width:${chartWidth}px" tabindex="0" aria-label="Scrollable 24-hour QoE chart"><svg viewBox="0 0 ${chartWidth} 86" role="img" aria-label="Recent hourly QoE trend from ${first.toFixed(1)} to ${last.toFixed(1)}">
        <defs><linearGradient id="cellPulseArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#7c55b5" stop-opacity=".28"/><stop offset="100%" stop-color="#7c55b5" stop-opacity=".03"/></linearGradient></defs>
        <rect class="cell-pulse-plot" x="33" y="10" width="${plotRight - 33}" height="58" rx="7"/>
        <line class="cell-pulse-grid" x1="34" y1="20" x2="${plotRight}" y2="20"/><line class="cell-pulse-grid" x1="34" y1="41" x2="${plotRight}" y2="41"/><line class="cell-pulse-grid" x1="34" y1="62" x2="${plotRight}" y2="62"/>
        <text class="cell-pulse-axis" x="29" y="22" text-anchor="end">${high.toFixed(0)}</text><text class="cell-pulse-axis" x="29" y="64" text-anchor="end">${low.toFixed(0)}</text>
        <polygon class="cell-pulse-area" points="${areaPoints}"/><polyline class="cell-pulse-line" points="${points}"/>
        ${timeTicks}
        ${coords.map((p, i) => `<circle class="cell-pulse-point${i === coords.length - 1 ? " current" : ""}" cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${i === coords.length - 1 ? 4 : 2.5}" data-qoe="${qoeHistory[i].toFixed(1)}" data-time="${timeLabels[i].full}" tabindex="0"><title>Time: ${timeLabels[i].full} · QoE score: ${qoeHistory[i].toFixed(1)} / 100</title></circle>`).join("")}
      </svg></div><div class="cell-pulse-hover" role="status" aria-live="polite"></div><div class="cell-pulse-x-label">Scenario time</div><div class="cell-pulse-time"><span>Drag to explore ${qoeHistory.length} hours</span><span>Latest · ${last.toFixed(1)}</span></div></div>
    </div>`;
  })() : "";
  const statusLine = `<div class="cell-intel-identity cell-intel-context"><div><h3>${cellName(h3)}</h3><span class="cell-current-issue ${quarterlyStatus === "currently_underperforming" ? "is-issue" : "is-normal"}"><i></i>${quarterlyStatus === "currently_underperforming" ? "Observed issue · 2026Q1" : `${quarterlyStatusLabel} · 2026Q1`}</span></div><button class="cell-clear-link" onclick="clearSelectedArea()">Clear</button></div>`;
  document.querySelectorAll(".insight-tab[data-insight-tab]").forEach(btn => btn.classList.toggle("active", btn.dataset.insightTab === state.insightTab));

  let content = "";
  if (state.insightTab === "ai") {
    const signals = explanation?.signals || [];
    const finalsDrivers = state.finals ? [1, 2, 3].map(n => ({ label: onsetValue(`risk_driver_${n}_label`, c), value: onsetValue(`risk_driver_${n}_value`, c), family: onsetValue(`risk_driver_${n}_family`, c) })).filter(row => row.label != null) : [];
    const finalsProtective = state.finals ? [1, 2].map(n => ({ label: onsetValue(`protective_${n}_label`, c), value: onsetValue(`protective_${n}_value`, c), family: onsetValue(`protective_${n}_family`, c) })).filter(row => row.label != null) : [];
    const signalRow = signal => `<li><i class="bi ${signal.word?.toLowerCase().includes("high") || signal.word?.toLowerCase().includes("elevated") ? "bi-exclamation-triangle" : "bi-arrow-left-right"}"></i><span>${signal.label}</span><b>${signal.value.toFixed(1)}${signal.unit} <em>${signal.word}</em></b></li>`;
    const finalsSignalRow = row => `<li class="model-signal-card risk-raising"><i class="bi bi-stars"></i><span><b>${row.label}</b><small>Observed value</small><em>${formatModelObservedValue(row.label, row.value)}</em></span><strong><small>Model effect</small><b>↑ higher risk</b></strong></li>`;
    const finalsProtectiveRow = row => `<li class="model-signal-card risk-lowering"><i class="bi bi-shield-check"></i><span><b>${row.label}</b><small>Observed value</small><em>${formatModelObservedValue(row.label, row.value)}</em></span><strong><small>Model effect</small><b>↓ lower risk</b></strong></li>`;
    const signalRows = state.finals ? finalsDrivers.map(finalsSignalRow).join("") : signals.slice(0, 3).map(signalRow).join("");
    const remainingSignalRows = signals.slice(3).map(signalRow).join("");
    const qoeEvidence = state.finals || qoeDelta == null ? "" : `<li><i class="bi ${qoeDelta >= 0 ? "bi-arrow-up" : "bi-arrow-down"}"></i><span>Recent QoE change</span><b>${qoeDelta >= 0 ? "+" : ""}${qoeDelta.toFixed(1)} <em>${qoeDelta >= 0 ? "Improving" : "Degrading"}</em></b></li>`;
    const quarterlyContext = state.finals ? [dataRow("Human-activity proxy (VIIRS)", formatModelObservedValue("Human-activity proxy (VIIRS)", onsetValue("viirs_rad_quarter", c)), "Context proxy; not measured network demand"), dataRow("Population density", onsetValue("population_density_per_km2", c) != null ? `${Number(onsetValue("population_density_per_km2", c)).toLocaleString(undefined, { maximumFractionDigits: 1 })} people/km²` : null), dataRow("Built-up ratio", formatModelObservedValue("Built-up ratio", onsetValue("built_up_ratio", c))), dataRow("Tower density", onsetValue("tower_density_per_km2", c) != null ? `${Number(onsetValue("tower_density_per_km2", c)).toFixed(2)} sites/km²` : null)].join("") : "";
    const explainContext = state.finals
      ? aiRank != null
        ? `<div class="cell-explain-context risk-${aiTier ?? "unscored"}"><span>Emerging-risk assessment · ${state.finals.metadata.forecast_quarter}</span><b>${ONSET_TIER_LABELS[aiTier] ?? aiTier} · rank #${Math.round(aiRank)}</b></div>`
        : `<div class="cell-explain-context ${quarterlyStatus === "currently_underperforming" ? "is-observed-issue" : "is-unscored"}"><span>Emerging-risk assessment · ${state.finals.metadata.forecast_quarter}</span><b>${quarterlyStatus === "currently_underperforming" ? "Not forecast-eligible" : "No quarterly AI ranking"}</b></div>`
      : `<div class="cell-explain-context risk-${aiTier ?? "unscored"}"><span>${aiStatus?.replaceAll("_", " ") ?? "Forecast context"}</span><b>${aiRank != null ? `${ONSET_TIER_LABELS[aiTier] ?? aiTier} · rank #${Math.round(aiRank)}` : "Not scored"}</b></div>`;
    const noQuarterlyExplanation = state.finals && aiRank == null;
    const explanationBody = noQuarterlyExplanation
      ? `<div class="cell-explain-unavailable"><i class="bi bi-info-circle"></i><div><b>No emerging-risk explanation for this cell</b><p>${quarterlyStatus === "currently_underperforming" ? `This cell is already underperforming in ${state.finals.metadata.predictor_quarter}, so it was excluded from the currently-normal population ranked for ${state.finals.metadata.forecast_quarter}.` : "This cell is outside the evaluable quarterly forecast population."}</p><button class="btn btn-sm btn-outline-primary" onclick="openSelectedCellReveal()">Review observed status and measurements</button></div></div>`
      : `<div class="cell-intel-note"><i class="bi bi-info-circle"></i>${state.finals ? "These factors are contributors to the model's risk assessment. They influenced its ranking and do not establish engineering cause. Risk rank is not an individual probability." : "AI values are a relative rank for prioritisation, not a probability of failure."}</div>
        ${signalRows || qoeEvidence ? `<h5>Risk-assessment contributors</h5><ul class="cell-evidence-list">${signalRows}${qoeEvidence}</ul>${finalsProtective.length ? `<h5>Protective contributors</h5><ul class="cell-evidence-list protective">${finalsProtective.map(finalsProtectiveRow).join("")}</ul>` : ""}${remainingSignalRows ? `<details class="cell-detail-disclosure"><summary>All available model signals</summary><ul class="cell-evidence-list">${remainingSignalRows}</ul></details>` : ""}` : `<p class="text-summary">No cell-level forecast evidence is available for this selected cell.</p>`}
        ${quarterlyContext ? `<details class="cell-detail-disclosure"><summary>Geographic and activity context</summary><div class="cell-data-group">${quarterlyContext}</div></details>` : ""}
        ${!state.finals && aiScore != null ? `<details><summary>Model scoring detail</summary><p>Relative score: <b>${aiScore.toFixed(2)}</b>. Snapshot: ${state.onset?.snapshot_timestamp ?? "—"} UTC. The score should be used to order investigation, not as a calibrated likelihood.</p></details>` : ""}
        <div class="cell-intel-flow"><span>Model explanation</span><i class="bi bi-arrow-right"></i><button class="btn btn-sm btn-outline-primary" onclick="openSelectedCellReveal()">Review quarterly measurements</button></div>`;
    content = `<section class="cell-intel-section ai-insight"><div class="cell-section-heading"><div><div class="cell-title-row"><span class="cell-intel-kicker">EXPLAIN</span><span class="cheat-badge ai">QUARTERLY AI</span></div><h4>${noQuarterlyExplanation ? "Quarterly AI explanation" : "Why the model flagged this cell"}</h4></div></div>${explainContext}${state.finalsActualRevealed && heldOutResult ? `<div class="cell-validation-result compact"><span>Held-out ${state.finals.metadata.forecast_quarter} result</span><b>${timeMachineLabel(heldOutResult)}</b></div>` : ""}${explanationBody}<button class="cell-secondary-link" onclick="setWorkspaceTool('ai')">Open full quarterly forecast</button></section>`;
  } else if (state.insightTab === "cause") {
    const causeEvidenceItems = [
      evidence("QoE", metrics.qoe != null ? `${metrics.qoe.toFixed(1)} / 100` : null, metricTone("qoe", metrics.qoe)),
      evidence("Scenario Pressure", metrics.demand != null ? `${metrics.demand.toFixed(1)} · ${demandLabel}` : null, metricTone("demand", metrics.demand)),
      evidence("RTT", metrics.rtt != null ? `${metrics.rtt.toFixed(1)} ms` : null, metricTone("latency", metrics.rtt)),
      evidence("Jitter", metrics.jitter != null ? `${metrics.jitter.toFixed(1)} ms` : null, metricTone("jitter", metrics.jitter)),
      evidence("Packet loss", metrics.loss != null ? `${metrics.loss.toFixed(2)}%` : null, metricTone("loss", metrics.loss)),
      qoeDelta == null ? "" : evidence("QoE movement", `${qoeDelta >= 0 ? "+" : ""}${qoeDelta.toFixed(1)} points vs prior hour`, qoeDelta <= -10 ? "quality-poor" : qoeDelta < 0 ? "quality-watch" : "quality-good"),
      evidence("Tower sites", props.tower_count != null ? `${props.tower_count}` : null),
    ].filter(Boolean);
    const causeEvidence = causeEvidenceItems.slice(0, 4).join("");
    const remainingCauseEvidence = causeEvidenceItems.slice(4).join("");
    const supportingSignals = cellSignals.slice(1).map(key => `<span>${CAUSE_LABELS[key]} · ${(investigationScores[key] * 100).toFixed(0)}th percentile</span>`).join("");
    content = `<section class="cell-intel-section cause-insight"><div class="cell-title-row"><span class="cell-intel-kicker">EXPLAIN</span><span class="cheat-badge heuristic">DEMO / SCENARIO</span></div><h4>Primary investigation hypothesis: ${CAUSE_LABELS[cause]}</h4><p class="cell-intel-lead">${explanation?.likelyTrigger ?? "No investigation signal reaches the citywide top quartile for this scenario hour."}</p>${supportingSignals ? `<div class="cell-alternative-drivers"><b>Supporting signals</b>${supportingSignals}</div>` : ""}<div class="cell-measurement-support"><span>Scenario evidence coverage</span><b>${cellSupport.label}</b><small>${cellValue("confidence", t, c) != null ? `${(cellValue("confidence", t, c) * 100).toFixed(0)}% confidence` : "Unavailable"} · separate from investigation signals</small></div><h5>Scenario measurements</h5><ul class="cell-evidence-list">${causeEvidence || "<li>No supporting measurements are available for this hour.</li>"}</ul>${remainingCauseEvidence ? `<details class="cell-detail-disclosure"><summary>Additional scenario measurements</summary><ul class="cell-evidence-list">${remainingCauseEvidence}</ul></details>` : ""}<div class="cell-intel-note"><i class="bi bi-search"></i> Signals mark the citywide top quartile for this scenario hour. They are relative investigation thresholds, not engineering failure thresholds. Cells may show multiple supporting signals.</div><div class="cell-intel-flow"><span>Evidence suggests checks</span><i class="bi bi-arrow-right"></i><button class="btn btn-sm btn-outline-primary" onclick="setInsightTab('action')">Set investigation priority</button></div><button class="cell-secondary-link" onclick="setWorkspaceTool('rootcause')">Open investigation evidence</button></section>`;
  } else if (state.insightTab === "action") {
    const actionStatus = state.tracking[h3] || "";
    const investigationPriority = quarterlyStatus === "currently_underperforming" ? "Investigate observed current issue" : aiRank != null && aiRank <= (state.finals.metadata.risk_thresholds?.elevated_max_rank ?? 200) ? `Prioritise next-quarter review · rank #${Math.round(aiRank)}` : "Monitor through routine review";
    content = `<section class="cell-intel-section action-insight"><span class="cell-intel-kicker">PRIORITISE</span><h4>${investigationPriority}</h4><div class="cell-action-context"><span>Quarterly basis</span><b>${quarterlyStatusLabel}${aiRank != null ? ` · ${ONSET_TIER_LABELS[aiTier]} #${Math.round(aiRank)}` : ""}</b><small>Observed status and emerging-risk rank determine where to investigate first.</small></div><details class="cell-detail-disclosure"><summary>Demo / Scenario investigation option</summary><p>${ACTIONS[cause] ?? "Review this cell with network operations."}</p><p>Hourly scenario: QoE ${metrics.qoe?.toFixed(1) ?? "—"} · pressure ${demandLabel} · RTT ${metrics.rtt?.toFixed(0) ?? "—"} ms.</p></details><div class="cell-action-timeline" aria-label="Action status"><span class="${!actionStatus ? "current" : ""}">Open</span><span class="${actionStatus === "investigating" ? "current" : ""}">Investigating</span><span class="${actionStatus === "planned" ? "current" : ""}">Planned</span><span class="${actionStatus === "done" ? "current" : ""}">Resolved</span></div><label class="cell-status-control"><span>Status</span><select onchange="setTracking('${h3}', this.value || null)"><option value="">Not set</option><option value="planned" ${actionStatus === "planned" ? "selected" : ""}>Planned</option><option value="investigating" ${actionStatus === "investigating" ? "selected" : ""}>Investigating</option><option value="done" ${actionStatus === "done" ? "selected" : ""}>Resolved</option></select></label><div class="cell-intel-note"><i class="bi bi-person-workspace"></i> Validate contributors and measurements before selecting an engineering response. Tracking is saved locally in this browser.</div></section>`;
  } else if (state.insightTab === "data") {
    const finalsForecastRows = state.finals ? [dataRow("Observed current status", quarterlyStatusLabel), dataRow("Quarterly AI rank", aiRank != null ? `#${Math.round(aiRank)}` : "Not forecast-eligible"), dataRow("Risk band", aiRank != null ? (ONSET_TIER_LABELS[aiTier] ?? aiTier) : null), dataRow("Quarterly measurement support", onsetValue("measurement_support", c)), dataRow("Ookla tests", onsetValue("ookla_tests", c)), dataRow("Download", quarterlyDl != null ? `${quarterlyDl.toFixed(1)} Mbps` : null), dataRow("Upload", quarterlyUl != null ? `${quarterlyUl.toFixed(1)} Mbps` : null), dataRow("Latency", quarterlyRtt != null ? `${quarterlyRtt.toFixed(1)} ms` : null), dataRow("Download percentile", onsetValue("dl_percentile_in_quarter", c) != null ? `${onsetValue("dl_percentile_in_quarter", c).toFixed(1)}` : null), state.finalsActualRevealed ? dataRow("Actual outcome", timeMachineLabel(onsetValue("time_machine_result", c))) : ""].join("") : `${dataRow("AI rank", aiRank != null ? `#${Math.round(aiRank)}` : null)}${dataRow("Priority tier", aiTier ? (ONSET_TIER_LABELS[aiTier] ?? aiTier) : null)}${dataRow("Operational status", aiStatus ? aiStatus.replaceAll("_", " ") : null)}`;
    content = `<section class="cell-intel-section cell-data-section"><h4>Technical details</h4><p class="cell-intel-lead">Quarterly real and hourly scenario data remain explicitly separated.</p><div class="cell-data-group"><h5>Hourly service condition <span class="cheat-badge sim">DEMO / SCENARIO</span></h5>${[
      dataRow("QoE", metrics.qoe != null ? `${metrics.qoe.toFixed(1)} / 100` : null), dataRow("Download", metrics.dl != null ? `${metrics.dl.toFixed(1)} Mbps` : null), dataRow("Upload", metrics.ul != null ? `${metrics.ul.toFixed(1)} Mbps` : null), dataRow("RTT", metrics.rtt != null ? `${metrics.rtt.toFixed(1)} ms` : null), dataRow("Jitter", metrics.jitter != null ? `${metrics.jitter.toFixed(1)} ms` : null), dataRow("Packet loss", metrics.loss != null ? `${metrics.loss.toFixed(2)}%` : null), dataRow("Scenario Pressure", metrics.demand != null ? metrics.demand.toFixed(1) : null), dataRow("Scenario evidence coverage", cellValue("confidence", t, c) != null ? `${(cellValue("confidence", t, c) * 100).toFixed(0)}%` : null)
    ].join("") || "<p class='text-summary'>No hourly values available.</p>"}</div><div class="cell-data-group"><h5>Static GIS context <span class="cheat-badge real">REAL STATIC GIS</span></h5>${[
      dataRow("Population (day)", props.population_day != null ? Math.round(props.population_day).toLocaleString() : null), dataRow("Tower sites", props.tower_count), dataRow("Built-up ratio", props.built_up_ratio != null ? `${(props.built_up_ratio * 100).toFixed(1)}%` : null), dataRow("Nearest tower", props.dist_to_nearest_tower_m != null ? `${Math.round(props.dist_to_nearest_tower_m)} m` : null)
    ].join("") || "<p class='text-summary'>No static context available.</p>"}</div>${state.onset ? `<div class="cell-data-group"><h5>${state.finals ? "Finals quarterly forecast" : "Forecast output"} <span class="cheat-badge ai">AI MODEL</span></h5>${finalsForecastRows}${state.finals ? `<p class="text-summary">Quarterly measurement support reflects Ookla coverage, not model confidence. Quarterly rank is not a probability.</p>` : ""}</div>` : ""}</section>`;
  } else {
    const driverRows = [1, 2, 3].map(n => ({ label: onsetValue(`risk_driver_${n}_label`, c), value: onsetValue(`risk_driver_${n}_value`, c), family: onsetValue(`risk_driver_${n}_family`, c) })).filter(row => row.label);
    const protectiveRows = [1, 2].map(n => ({ label: onsetValue(`protective_${n}_label`, c), value: onsetValue(`protective_${n}_value`, c), family: onsetValue(`protective_${n}_family`, c) })).filter(row => row.label);
    const drivers = driverRows.map(row => row.label);
    const currentMeaning = quarterlyStatus === "currently_underperforming" ? "This cell is already underperforming and should be investigated now." : aiRank != null ? `Currently normal, with ${String(ONSET_TIER_LABELS[aiTier] ?? aiTier).toLowerCase()} for ${state.finals.metadata.forecast_quarter}.` : "No quarterly priority is available for this cell.";
    const priorityMeaning = quarterlyStatus === "currently_underperforming" ? "Investigate the observed current issue" : aiRank != null && aiRank <= (state.finals.metadata.risk_thresholds?.elevated_max_rank ?? 200) ? "Prioritise next-quarter investigation" : "Monitor through routine review";
    const support = onsetValue("measurement_support", c);
    const tests = onsetValue("ookla_tests", c);
    const driverSummary = drivers.length ? drivers.slice(0, 2).join(" · ") : "";
    const modelExplanation = aiRank != null ? `<details class="cell-summary-disclosure"><summary><span>Model explanation</span><small>AI MODEL · contributors</small></summary><div class="cell-disclosure-body"><p>These factors are contributors to the model's risk assessment. They influenced its ranking and do not establish engineering cause.</p><h5>Risk contributors</h5><ul class="cell-contributor-list">${driverRows.map(row => modelContributorHtml(row, "higher")).join("")}</ul>${protectiveRows.length ? `<h5>Protective contributors</h5><ul class="cell-contributor-list protective">${protectiveRows.map(row => modelContributorHtml(row, "lower")).join("")}</ul>` : ""}<div class="cell-data-row"><span>Risk band and rank</span><b>${ONSET_TIER_LABELS[aiTier]} · #${Math.round(aiRank)}</b></div>${dataRow("Risk percentile", onsetValue("risk_percentile", c) != null ? `${onsetValue("risk_percentile", c).toFixed(1)}%` : null)}</div></details>` : "";
    const supportBlock = support ? `<div class="cell-summary-row"><div><span>Quarterly measurement support</span><b>${support}</b><small>${tests != null ? `${tests} Ookla tests · evidence coverage, not model confidence` : "Evidence coverage, not model confidence"}</small></div><span class="cheat-badge real">REAL OOKLA</span></div>` : "";
    const forecastBlock = aiRank != null ? metric("Next-quarter AI status", `${ONSET_TIER_LABELS[aiTier]} · rank #${Math.round(aiRank)}`) : "";
    const whyBlock = aiRank != null ? `<div class="cell-summary-row cell-model-summary"><div><span>Why the model flagged it</span><b>${driverSummary || "Review model contributors"}</b><small>Contributors to the risk assessment</small></div><span class="cheat-badge ai">AI MODEL</span></div>` : "";
    const trackingStatus = state.tracking[h3] || "";
    const fmtNumber = (value, digits = 1) => value == null ? null : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    const fmtUnit = (value, unit, digits = 1) => value == null ? null : `${fmtNumber(value, digits)} ${unit}`;
    content = `<section class="cell-intel-section cell-overview cell-summary-first"><div class="cell-meaning"><span>What this means</span><p>${currentMeaning}</p></div><div class="cell-status-pair">${metric("Current observed status", quarterlyStatusLabel, quarterlyStatus === "currently_underperforming" ? "critical" : "")}${forecastBlock}</div><div class="cell-metric-section"><div class="cell-section-label"><span>Quarterly network measurements</span><span class="cheat-badge real">REAL OOKLA</span></div><div class="cell-metric-strip">${metric("Download", quarterlyDl != null ? `${quarterlyDl.toFixed(1)} Mbps` : null)}${metric("Upload", quarterlyUl != null ? `${quarterlyUl.toFixed(1)} Mbps` : null)}${metric("Latency", quarterlyRtt != null ? `${quarterlyRtt.toFixed(1)} ms` : null)}</div></div><div class="cell-summary-stack">${whyBlock}${supportBlock}<div class="cell-summary-row cell-priority-summary"><div><span>Investigation priority</span><b>${priorityMeaning}</b><small>${quarterlyStatus === "currently_underperforming" ? "REVEAL → PRIORITISE" : "ANTICIPATE → EXPLAIN → PRIORITISE"}</small></div><i class="bi bi-arrow-right-circle"></i></div></div><div class="cell-disclosures"><details class="cell-summary-disclosure"><summary><span>Area context</span><small>Location and contextual variables</small></summary><div class="cell-disclosure-body">${dataRow("H3 cell", h3)}${dataRow("District population", fmtNumber(onsetValue("district_population", c), 0))}${dataRow("Population density", fmtUnit(onsetValue("population_density_per_km2", c), "people/km²", 1))}${dataRow("Population (day)", fmtNumber(onsetValue("population_day", c) ?? props.population_day, 0))}${dataRow("Tower density", fmtUnit(onsetValue("tower_density_per_km2", c), "sites/km²", 2))}${dataRow("Nearest tower", fmtUnit(onsetValue("dist_to_nearest_tower_m", c) ?? props.dist_to_nearest_tower_m, "m", 0))}${dataRow("Building density", fmtUnit(onsetValue("building_density_per_km2", c), "buildings/km²", 1))}${dataRow("Built-up ratio", onsetValue("built_up_ratio", c) != null ? `${(onsetValue("built_up_ratio", c) * 100).toFixed(1)}%` : props.built_up_ratio != null ? `${(props.built_up_ratio * 100).toFixed(1)}%` : null)}${dataRow("Road density", fmtUnit(onsetValue("road_density_km_per_km2", c), "km/km²", 2))}${dataRow("Intersection density", fmtUnit(onsetValue("intersection_density_per_km2", c), "per km²", 1))}${dataRow("Nearest transit", fmtUnit(onsetValue("dist_to_nearest_transit_m", c), "m", 0))}${dataRow("Human-activity proxy (VIIRS)", fmtNumber(onsetValue("viirs_rad_quarter", c), 1), "Proxy only; not measured network demand")}${dataRow("VIIRS year-on-year change", fmtNumber(onsetValue("viirs_rad_yoy_log_ratio", c), 2))}</div></details>${modelExplanation}<details class="cell-summary-disclosure"><summary><span>Investigation tracking</span><small>Status and next check</small></summary><div class="cell-disclosure-body"><p><b>Primary investigation hypothesis:</b> ${CAUSE_LABELS[cause]}</p>${cellSignals.length > 1 ? `<p><b>Supporting signals:</b> ${cellSignals.slice(1).map(key => CAUSE_LABELS[key]).join(" · ")}</p>` : ""}<p><b>Scenario evidence coverage:</b> ${cellSupport.label}</p><p><b>Investigation option:</b> ${ACTIONS[cause]}</p><label class="cell-status-control"><span>Status</span><select onchange="setTracking('${h3}', this.value || null)"><option value="">Not set</option><option value="planned" ${trackingStatus === "planned" ? "selected" : ""}>Planned</option><option value="investigating" ${trackingStatus === "investigating" ? "selected" : ""}>Investigating</option><option value="done" ${trackingStatus === "done" ? "selected" : ""}>Resolved</option></select></label></div></details><details class="cell-summary-disclosure"><summary><span>Demo / Scenario hourly playback</span><small>Separate from quarterly real data</small></summary><div class="cell-disclosure-body"><span class="cheat-badge sim">DEMO / SCENARIO</span><small class="cell-scenario-time">${timestampLabel}</small>${dataRow("Scenario status", statusLabel)}${dataRow("QoE", metrics.qoe != null ? `${metrics.qoe.toFixed(1)} / 100` : null)}${dataRow("Download", metrics.dl != null ? `${metrics.dl.toFixed(1)} Mbps` : null)}${dataRow("Upload", metrics.ul != null ? `${metrics.ul.toFixed(1)} Mbps` : null)}${dataRow("RTT", metrics.rtt != null ? `${metrics.rtt.toFixed(1)} ms` : null)}${dataRow("Jitter", metrics.jitter != null ? `${metrics.jitter.toFixed(1)} ms` : null)}${dataRow("Packet loss", metrics.loss != null ? `${metrics.loss.toFixed(2)}%` : null)}${dataRow("Scenario Pressure", metrics.demand != null ? metrics.demand.toFixed(1) : null)}${dataRow("Scenario evidence coverage", cellSupport.label, cellValue("confidence", t, c) != null ? `${(cellValue("confidence", t, c) * 100).toFixed(0)}% confidence` : "Unavailable")}${dataRow("Hotspot classification", hotspot)}${pulse}</div></details><details class="cell-summary-disclosure"><summary><span>Technical details</span><small>Quarterly fields and interpretation</small></summary><div class="cell-disclosure-body">${dataRow("Observed current status", quarterlyStatusLabel)}${dataRow("Forecast quarter", onsetValue("forecast_for_quarter", c))}${dataRow("AI risk band", aiRank != null ? ONSET_TIER_LABELS[aiTier] : null)}${dataRow("AI risk rank", aiRank != null ? `#${Math.round(aiRank)}` : null)}${dataRow("Risk percentile", onsetValue("risk_percentile", c) != null ? `${onsetValue("risk_percentile", c).toFixed(1)}%` : null)}${dataRow("Download percentile", onsetValue("dl_percentile_in_quarter", c) != null ? onsetValue("dl_percentile_in_quarter", c).toFixed(1) : null)}${dataRow("Historical underperformance rate", formatModelObservedValue("Historical underperformance rate", onsetValue("below_area_norm_rate_to_date", c)))}${dataRow("Recent average download", fmtUnit(onsetValue("rolling_mean_4q_dl", c), "Mbps", 1))}${dataRow("Previous-quarter download", fmtUnit(onsetValue("lag_1q_dl", c), "Mbps", 1))}${state.finalsActualRevealed ? dataRow("Next-quarter observed status", onsetValue("actual_next_quarter_status", c)?.replaceAll("_", " ")) : ""}${state.finalsActualRevealed ? dataRow("Time Machine result", timeMachineLabel(onsetValue("time_machine_result", c))) : ""}${dataRow("Score is probability", onsetValue("score_is_probability", c) === false ? "No" : null)}${dataRow("Interpretation note", onsetValue("interpretation_note", c))}</div></details></div></section>`;
  }
  if (state.insightTab === "overview") {
    const overviewDrivers = [1, 2, 3].map(n => ({ label: onsetValue(`risk_driver_${n}_label`, c), value: onsetValue(`risk_driver_${n}_value`, c), family: onsetValue(`risk_driver_${n}_family`, c) })).filter(row => row.label);
    const overviewProtective = [1, 2].map(n => ({ label: onsetValue(`protective_${n}_label`, c), value: onsetValue(`protective_${n}_value`, c), family: onsetValue(`protective_${n}_family`, c) })).filter(row => row.label);
    const overviewPriority = quarterlyStatus === "currently_underperforming" ? "Investigate the observed current issue" : aiRank != null && aiRank <= (state.finals.metadata.risk_thresholds?.elevated_max_rank ?? 200) ? "Prioritise next-quarter investigation" : "Monitor through routine review";
    const overviewPriorityNote = quarterlyStatus === "currently_underperforming" ? "Review current network performance and gather supporting evidence." : aiRank != null ? "Review the model contributors and quarterly measurement support." : "Continue routine quarterly monitoring.";
    const overviewSupport = onsetValue("measurement_support", c);
    const overviewTests = onsetValue("ookla_tests", c);
    const overviewTracking = state.tracking[h3] || "";
    const overviewNumber = (value, digits = 1) => value == null ? null : Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
    const overviewUnit = (value, unit, digits = 1) => value == null ? null : `${overviewNumber(value, digits)} ${unit}`;
    const historyRateTone = value => value == null ? "" : value >= .5 ? "quality-poor" : value >= .2 ? "quality-watch" : "quality-good";
    const contextGroup = (title, rows) => `<section class="cell-context-group"><h5>${title}</h5>${rows.filter(Boolean).join("") || `<p class="text-summary">No ${title.toLowerCase()} context available.</p>`}</section>`;
    const aiSummary = aiRank != null ? `<section class="cell-ai-summary risk-${aiTier ?? "unscored"}"><div><span>AI emerging risk</span><b>${ONSET_TIER_LABELS[aiTier] ?? aiTier}</b></div><strong>Rank ${Math.round(aiRank)} <small>of ${state.finals?.metadata?.forecast_eligible_cells?.toLocaleString?.() ?? state.finals?.metadata?.eligible_cell_count?.toLocaleString?.() ?? "2,186"}</small></strong><span class="cheat-badge ai">AI MODEL</span></section>` : quarterlyStatus === "currently_underperforming" ? `<p class="cell-ai-not-required"><i class="bi bi-info-circle"></i>AI emerging-risk forecasting is not required because this issue is already observed.</p>` : "";
    const modelExplanation = aiRank != null ? `<details class="cell-summary-disclosure"><summary><span>Why the model flagged it</span><small>AI model contributors</small></summary><div class="cell-disclosure-body"><p>These factors are contributors to the model's risk assessment. They influenced its ranking and do not establish engineering cause.</p><ul class="cell-contributor-list">${overviewDrivers.map(row => modelContributorHtml(row, "higher")).join("")}</ul>${overviewProtective.length ? `<h5>Protective contributors</h5><ul class="cell-contributor-list protective">${overviewProtective.map(row => modelContributorHtml(row, "lower")).join("")}</ul>` : ""}${overviewSupport ? `<div class="cell-measurement-support"><span>Quarterly measurement support</span><b>${overviewSupport}</b><small>${overviewTests != null ? `${overviewTests} Ookla tests` : "Quarterly evidence coverage"} · not model confidence</small></div>` : ""}</div></details>` : "";
    const historicalRate = onsetValue("below_area_norm_rate_to_date", c);
    const recentAverageDl = onsetValue("rolling_mean_4q_dl", c);
    const previousQuarterDl = onsetValue("lag_1q_dl", c);
    const performanceHistoryRows = `${dataRow("Historical underperformance rate", historicalRate != null ? formatModelObservedValue("Historical underperformance rate", historicalRate) : null, "Share of past quarters below the area norm", historyRateTone(historicalRate))}${dataRow("Four-quarter average download", recentAverageDl != null ? overviewUnit(recentAverageDl, "Mbps", 1) : null, "Quarterly performance history", metricTone("download", recentAverageDl))}${dataRow("Previous-quarter download", previousQuarterDl != null ? overviewUnit(previousQuarterDl, "Mbps", 1) : null, "Previous observed quarter", metricTone("download", previousQuarterDl))}`;
    const performanceHistory = performanceHistoryRows ? `<details class="cell-summary-disclosure"><summary><span>QoE / performance history</span><small>Quarterly trend context</small></summary><div class="cell-disclosure-body">${performanceHistoryRows}</div></details>` : "";
    content = `<section class="cell-intel-section cell-overview cell-summary-first">
      ${aiSummary}
      <div class="cell-metric-section"><div class="cell-section-label"><span>Quarterly network measurements</span><span class="cheat-badge real">REAL OOKLA</span></div><div class="cell-metric-strip">${metric("Download", quarterlyDl != null ? `${quarterlyDl.toFixed(1)} Mbps` : null, metricTone("download", quarterlyDl))}${metric("Upload", quarterlyUl != null ? `${quarterlyUl.toFixed(1)} Mbps` : null, metricTone("upload", quarterlyUl))}${metric("Latency", quarterlyRtt != null ? `${quarterlyRtt.toFixed(1)} ms` : null, metricTone("latency", quarterlyRtt))}</div><div class="cell-metric-guide" aria-label="Measurement colour key"><span class="quality-good"><i></i>Strong / responsive</span><span class="quality-watch"><i></i>Watch</span><span class="quality-poor"><i></i>Slow / high latency</span></div><details class="cell-value-guide"><summary>How to read these values</summary><div><p><b>Download and upload</b> show data transfer speed; higher is generally better.</p><p><b>Latency, jitter and packet loss</b> describe responsiveness and stability; lower is generally better.</p><p><b>Historical underperformance rate</b> is the share of available past quarters below the area norm.</p><p><b>Model contributors</b> affect the risk ranking. Protective contributors reduce assessed risk; they are not guarantees or verified engineering explanations.</p></div></details><section class="cell-priority-card"><span>Investigation priority</span><b>${overviewPriority}</b><p>${overviewPriorityNote}</p></section></div>
      <div class="cell-disclosures">
        ${modelExplanation}
        ${performanceHistory}
        <details class="cell-summary-disclosure"><summary><span>Area context</span><small>Local context</small></summary><div class="cell-disclosure-body cell-context-groups">${contextGroup("Population", [dataRow("Population density", overviewUnit(onsetValue("population_density_per_km2", c), "people/km²", 1)), dataRow("Day population", overviewNumber(onsetValue("population_day", c) ?? props.population_day, 0))])}${contextGroup("Infrastructure", [dataRow("Tower density", overviewUnit(onsetValue("tower_density_per_km2", c), "sites/km²", 2)), dataRow("Nearest tower", overviewUnit(onsetValue("dist_to_nearest_tower_m", c) ?? props.dist_to_nearest_tower_m, "m", 0)), dataRow("Nearest transit", overviewUnit(onsetValue("dist_to_nearest_transit_m", c), "m", 0))])}${contextGroup("Built environment", [dataRow("Building density", overviewUnit(onsetValue("building_density_per_km2", c), "buildings/km²", 1)), dataRow("Built-up ratio", onsetValue("built_up_ratio", c) != null ? `${(onsetValue("built_up_ratio", c) * 100).toFixed(1)}%` : props.built_up_ratio != null ? `${(props.built_up_ratio * 100).toFixed(1)}%` : null), dataRow("Road density", overviewUnit(onsetValue("road_density_km_per_km2", c), "km/km²", 2)), dataRow("Intersection density", overviewUnit(onsetValue("intersection_density_per_km2", c), "per km²", 1))])}${contextGroup("Activity", [dataRow("Human-activity proxy (VIIRS)", overviewNumber(onsetValue("viirs_rad_quarter", c), 1)), dataRow("VIIRS year-on-year change", overviewNumber(onsetValue("viirs_rad_yoy_log_ratio", c), 2))])}<p class="cell-proxy-note">VIIRS is a human-activity proxy, not measured network demand.</p></div></details>
        <details class="cell-summary-disclosure"><summary><span>Investigation tracking</span><small>Tracking workflow</small></summary><div class="cell-disclosure-body"><div class="cell-investigation-workflow"><div><span>Primary investigation hypothesis</span><b>${CAUSE_LABELS[cause]}</b></div><div><span>Scenario evidence coverage</span><b>${cellSupport.label}</b></div><div><span>Recommended next check</span><b>${ACTIONS[cause]}</b></div><label class="cell-status-control"><span>Status</span><select onchange="setTracking('${h3}', this.value || null)"><option value="">Not set</option><option value="planned" ${overviewTracking === "planned" ? "selected" : ""}>Planned</option><option value="investigating" ${overviewTracking === "investigating" ? "selected" : ""}>Investigating</option><option value="done" ${overviewTracking === "done" ? "selected" : ""}>Resolved</option></select></label></div></div></details>
        <details class="cell-summary-disclosure"><summary><span>Demo / Scenario hourly playback <span class="cheat-badge sim">DEMO / SCENARIO</span></span><small>Hourly demo</small></summary><div class="cell-disclosure-body"><small class="cell-scenario-time">${timestampLabel}</small>${dataRow("QoE", metrics.qoe != null ? `${metrics.qoe.toFixed(1)} / 100` : null, "", metricTone("qoe", metrics.qoe))}${dataRow("Download", metrics.dl != null ? `${metrics.dl.toFixed(1)} Mbps` : null, "", metricTone("download", metrics.dl))}${dataRow("Upload", metrics.ul != null ? `${metrics.ul.toFixed(1)} Mbps` : null, "", metricTone("upload", metrics.ul))}${dataRow("RTT", metrics.rtt != null ? `${metrics.rtt.toFixed(1)} ms` : null, "", metricTone("latency", metrics.rtt))}${dataRow("Jitter", metrics.jitter != null ? `${metrics.jitter.toFixed(1)} ms` : null, "", metricTone("jitter", metrics.jitter))}${dataRow("Packet loss", metrics.loss != null ? `${metrics.loss.toFixed(2)}%` : null, "", metricTone("loss", metrics.loss))}${dataRow("Scenario Pressure", metrics.demand != null ? metrics.demand.toFixed(1) : null, "", metricTone("demand", metrics.demand))}${dataRow("Hotspot classification", hotspot)}${pulse}</div></details>
        <details class="cell-summary-disclosure"><summary><span>Technical details</span><small>Data and methodology</small></summary><div class="cell-disclosure-body">${dataRow("H3 cell", h3)}${dataRow("Observed quarter", state.finals?.metadata?.predictor_quarter)}${dataRow("Forecast quarter", onsetValue("forecast_for_quarter", c))}${dataRow("AI risk band", aiRank != null ? ONSET_TIER_LABELS[aiTier] : null)}${dataRow("AI risk rank", aiRank != null ? `#${Math.round(aiRank)}` : null)}${dataRow("Risk percentile", onsetValue("risk_percentile", c) != null ? `${onsetValue("risk_percentile", c).toFixed(1)}%` : null)}${dataRow("Download percentile", onsetValue("dl_percentile_in_quarter", c) != null ? onsetValue("dl_percentile_in_quarter", c).toFixed(1) : null)}${dataRow("Historical underperformance rate", overviewNumber(onsetValue("below_area_norm_rate_to_date", c), 3), "Share of past quarters below the area norm", historyRateTone(onsetValue("below_area_norm_rate_to_date", c)))}${dataRow("Recent average download", overviewUnit(onsetValue("rolling_mean_4q_dl", c), "Mbps", 1), "Four-quarter performance history", metricTone("download", onsetValue("rolling_mean_4q_dl", c)))}${dataRow("Previous-quarter download", overviewUnit(onsetValue("lag_1q_dl", c), "Mbps", 1), "Previous observed quarter", metricTone("download", onsetValue("lag_1q_dl", c)))}${state.finalsActualRevealed ? dataRow("Next-quarter observed status", onsetValue("actual_next_quarter_status", c)?.replaceAll("_", " ")) : ""}${state.finalsActualRevealed ? dataRow("Time Machine result", timeMachineLabel(onsetValue("time_machine_result", c))) : ""}${dataRow("Score is probability", onsetValue("score_is_probability", c) === false ? "No" : null)}${dataRow("Interpretation note", onsetValue("interpretation_note", c))}</div></details>
      </div>
    </section>`;
  }
  el.innerHTML = `${statusLine}${content}`;
  initCellPulseInteractions(el);
}

function initCellPulseInteractions(root) {
  const scroll = root?.querySelector(".cell-pulse-scroll");
  const tooltip = root?.querySelector(".cell-pulse-hover");
  if (!scroll || !tooltip) return;
  window.requestAnimationFrame(() => { scroll.scrollLeft = scroll.scrollWidth; });
  const showPoint = point => {
    if (!point) { tooltip.classList.remove("visible"); return; }
    const scrollRect = scroll.getBoundingClientRect();
    const pointRect = point.getBoundingClientRect();
    tooltip.innerHTML = `<span><em>Time</em>${point.dataset.time}</span><span><em>QoE score</em><b>${point.dataset.qoe} / 100</b></span>`;
    tooltip.style.left = `${Math.max(68, Math.min(scroll.clientWidth - 68, pointRect.left - scrollRect.left + pointRect.width / 2))}px`;
    tooltip.classList.add("visible");
  };
  scroll.addEventListener("pointermove", event => showPoint(event.target.closest(".cell-pulse-point")));
  scroll.addEventListener("pointerleave", () => showPoint(null));
  scroll.querySelectorAll(".cell-pulse-point").forEach(point => {
    point.addEventListener("focus", () => showPoint(point));
    point.addEventListener("blur", () => showPoint(null));
  });
  let dragging = false, startX = 0, startScroll = 0;
  scroll.addEventListener("pointerdown", event => {
    if (event.target.closest(".cell-pulse-point")) return;
    dragging = true; startX = event.clientX; startScroll = scroll.scrollLeft;
    scroll.classList.add("dragging"); scroll.setPointerCapture?.(event.pointerId);
  });
  scroll.addEventListener("pointermove", event => { if (dragging) scroll.scrollLeft = startScroll - (event.clientX - startX); });
  const stopDrag = event => { dragging = false; scroll.classList.remove("dragging"); scroll.releasePointerCapture?.(event.pointerId); };
  scroll.addEventListener("pointerup", stopDrag);
  scroll.addEventListener("pointercancel", stopDrag);
}

function setInsightTab(tab) {
  if (!document.body.classList.contains("in-scenario-lab") && tab === "action") { setWorkspaceTool("prioritise"); return; }
  if (!document.body.classList.contains("in-scenario-lab") && tab === "cause") tab = "ai";
  const body = document.getElementById("selectedAreaBody");
  state.insightTab = tab;
  renderSelectedAreaInsights();
  if (window.gsap && body && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    gsap.fromTo(body, { autoAlpha: 0, x: 8 }, { autoAlpha: 1, x: 0, duration: .22, ease: "power2.out", overwrite: "auto" });
  }
}

function advanceGuidedDemo() {
  const banner = document.getElementById("guidedDemoBanner");
  const popover = document.getElementById("guidedDemoPopover");
  const backdrop = document.getElementById("guidedDemoBackdrop");
  const steps = [
    { label: "REVEAL", tab: "Observed Q1", tool: "overview", mode: "onset", quarterlyView: "observed", use: "Review observed 2026Q1 underperformance and select a cell.", expect: "Only measured 2026Q1 status is encoded on the map." },
    { label: "ANTICIPATE", tab: "Forecast Q2", tool: "ai", mode: "onset", quarterlyView: "forecast", use: "Review 2026Q2 emerging-risk priorities among cells not underperforming in 2026Q1.", expect: "High and elevated bands order investigation candidates; ranks are not probabilities." },
    { label: "EXPLAIN", tab: "Selected cell", tool: "cell", mode: "onset", quarterlyView: "forecast", use: "Inspect the selected cell's model contributors and measurement support.", expect: "SHAP contributors explain the assessment without claiming causation." },
    { label: "PRIORITISE", tab: "Combined workflow", tool: "prioritise", mode: "onset", quarterlyView: "combined", use: "Compare observed issues with the future-risk shortlist before setting a tracking status.", expect: "Observed 2026Q1 issues and selected 2026Q2 risks remain clearly labelled." },
  ];
  state.guidedDemoStep = (state.guidedDemoStep ?? -1) + 1;
  if (state.guidedDemoStep >= steps.length) state.guidedDemoStep = 0;
  const step = steps[state.guidedDemoStep];
  if (step.mode !== "onset" || state.onset) {
    state.colorBy = step.mode;
    if (step.quarterlyView) state.quarterlyMapView = step.quarterlyView;
    document.getElementById("colorBySelect").value = step.mode;
    syncPrimaryMapControls(); renderMapColors(); renderLegend();
  }
  setWorkspaceTool(step.tool);
  document.querySelectorAll(".workspace-tool").forEach(btn => {
    const targeted = btn.dataset.workspaceTool === step.tool;
    btn.classList.toggle("guided-demo-target", targeted);
    btn.toggleAttribute("aria-current", targeted);
  });
  const top = state.lastCandidates?.[0];
  if (step.focusPriority && top) flyToCell(top.h3);
  banner.classList.remove("d-none");
  banner.innerHTML = `<strong>${step.label}</strong><span>${step.use} ${step.expect}</span><button class="btn btn-sm btn-outline-primary" onclick="advanceGuidedDemo()">Next</button>`;
  popover.classList.remove("d-none");
  backdrop?.classList.remove("d-none");
  popover.innerHTML = `<div class="guided-demo-top"><span>Guided demo · ${state.guidedDemoStep + 1}/${steps.length}</span><button onclick="closeGuidedDemo()" aria-label="Close guided demo">×</button></div><h3>${step.label}</h3><div><b>Open</b><p><span class="demo-tab-chip">${step.tab}</span></p></div><div><b>Do this</b><p>${step.use}</p></div><div><b>You should see</b><p>${step.expect}</p></div><button class="btn btn-sm btn-primary" onclick="advanceGuidedDemo()">${state.guidedDemoStep === steps.length - 1 ? "Restart demo" : "Next demo step"}</button>`;
  document.getElementById("guidedDemoBtn").textContent = "Next demo step";
}

function closeGuidedDemo() {
  document.getElementById("guidedDemoPopover")?.classList.add("d-none");
  document.getElementById("guidedDemoBackdrop")?.classList.add("d-none");
  document.querySelectorAll(".workspace-tool.guided-demo-target").forEach(btn => {
    btn.classList.remove("guided-demo-target");
    btn.removeAttribute("aria-current");
  });
  document.getElementById("guidedDemoBanner")?.classList.add("d-none");
  state.guidedDemoStep = -1;
  document.getElementById("guidedDemoBtn").textContent = "Start guided demo";
}

function renderTrustPanel() {
  const s = state.summary;
  const disclaimerParts = (s.disclaimer || "").split(/(?<=[.!?])\s+/);
  const shortDisclaimer = disclaimerParts.slice(0, 2).join(" ");
  const remainingDisclaimer = disclaimerParts.slice(2).join(" ");
  document.getElementById("trustDisclaimer").innerHTML = `<strong>Scenario provenance</strong><p>The hourly QoE panel is a documented Demo/Scenario dataset built on real static GIS context. It is not observed operator telemetry.</p><details><summary>Read complete note</summary><p>${[shortDisclaimer, remainingDisclaimer].filter(Boolean).join(" ")}</p><p>${s.ookla_anchored_pct}% of cells have scenario QoE re-centred on a real quarterly Ookla average; this describes simulation construction, not operator telemetry coverage.</p></details>`;

  const sourceRows = [
    ["Ookla Open Data", "ookla", "Real quarterly network measurements for observed status and the finals forecast dataset; also used in scenario construction."],
    ["Google Open Buildings", "buildings", "Real building footprints used for local context."],
    ["OpenCellID", "towers", "Real tower locations used for coverage context."],
    ["OpenStreetMap", "osm", "Real roads and Points of Interest."],
    ["WorldPop", "worldpop", "Real population estimates used for impact sizing."],
    ["OpenDOSM", "dosm_population", "Official Malaysian population statistics from population_malaysia.csv."],
    ["Transit ridership", "ridership", "Headline ridership figures from ridership_headline.csv, used as a transit-demand proxy."],
  ];
  document.getElementById("sourceBadges").innerHTML = `<div class="source-table-wrap"><table class="source-table"><thead><tr><th>Source</th><th>Status</th><th>Meaning</th></tr></thead><tbody>${sourceRows.map(([name, key, meaning]) => {
    const present = Boolean(s.source_datasets?.[key]);
    return `<tr><td>${name}</td><td><span class="source-status ${present ? "real" : "missing"}">${present ? "REAL SOURCE" : "Unavailable"}</span></td><td>${meaning}</td></tr>`;
  }).join("")}${[
    ["Ookla quarterly measurements", "REAL QUARTERLY", "Observed quarterly status and frozen forecast dataset."],
    ["Copernicus DEM GLO-30", "REAL SOURCE", "Terrain context from a surface model including buildings and vegetation; not bare-earth height."],
    ["ESA WorldCover", "REAL SOURCE", "Land-cover fractions, including built-up land context."],
    ["VIIRS nighttime lights", "ACTIVITY PROXY", "Human-activity proxy; not measured mobile-network demand."],
    ["Sunway / Subang Jaya QoE", "THIRD-PARTY MEASURED", "Local field-measured case study with limited geographic coverage; not area-wide validation."],
    ["BQoES / MCMC reference material", "LITERATURE ONLY", "Supports user/application relevance and application-profile weighting; not a quarterly forecasting feature."],
    ["Hourly cell_time_dataset", "DEMO / SCENARIO", "Simulated hourly demonstration workflow; not observed operator telemetry."],
    ["Frozen quarterly emerging-risk model", "AI MODEL", "Relative ranking of currently-normal cells; not calibrated probability."],
    ["Investigation signals", "HEURISTIC", "Independent relative scenario signals for investigation; not diagnosed causes."]
  ].map(([name, badge, meaning]) => `<tr><td>${name}</td><td><span class="source-status" data-provenance="${badge}">${badge}</span></td><td>${meaning}</td></tr>`).join("")}</tbody></table></div>`;

  document.getElementById("anchoringStat").innerHTML = `<p class="text-summary">Current finals study area: Klang Valley, Malaysia. Malaysia is the primary implementation. Other regions are future scalability or external-validation opportunities; no ASEAN-wide validation is claimed.</p><details><summary>Hourly scenario construction</summary>
    <div class="kpi-value" style="font-size:22px">${s.ookla_anchored_pct}%</div>
    <div class="anchor-bar"><div class="anchor-bar-fill" style="width:${s.ookla_anchored_pct}%"></div></div>
    <div class="text-summary">of cells have their simulated hourly QoE re-centred on a real quarterly
    Ookla average. The remaining ${(100 - s.ookla_anchored_pct).toFixed(1)}% are simulated-only —
    per the pipeline's own validation notes, almost all "degraded" hours occur on anchored cells.</div></details>
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
  let peakDemand = 0;
  const rows = [];
  for (let c = 0; c < C; c++) {
    const h3 = state.panel.h3_ids[c];
    const cls = cellValue("degradation_class", t, c);
    const demand = cellValue("demand_pressure", t, c) ?? 0;
    peakDemand = Math.max(peakDemand, demand);
    if (cls >= 1) popAffected += state.gridPropsByH3.get(h3)?.population_day || 0;
    rows.push({ h3, qoe: cellValue("qoe", t, c), demand, hotspot: cellValue("hotspot_code", t, c) });
  }
  const worst5 = [...rows].sort((a, b) => a.qoe - b.qoe).slice(0, 5);
  const priorities = (state.lastCandidates || []).slice(0, 3);
  const operationalRead = `${cur.severe_count} severe and ${cur.degraded_count} degraded cells are present. ${state.emergingCount ?? 0} emerging hotspot${(state.emergingCount ?? 0) === 1 ? "" : "s"} warrant early attention.`;

  const rootCauseImg = await Plotly.toImage("chartRootCause", { format: "png", width: 500, height: 220 });
  const qoeImg = await Plotly.toImage("chartQoe", { format: "png", width: 500, height: 180 });

  const win = window.open("", "_blank");
  win.document.write(`
    <html><head><title>GeoQoE Pulse — Scenario Report</title>
    <style>
      body { font-family: Arial, sans-serif; color: #393442; padding: 32px; max-width: 850px; margin: auto; }
      h1 { color: #282330; font-size: 22px; margin-bottom: 0; }.eyebrow { color:#7258b6; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
      h2 { margin:24px 0 8px; color:#282330; font-size:15px; }.sub { color: #746d7c; font-size: 12px; margin-top: 4px; }
      .read { margin:18px 0; padding:12px 14px; border-left:4px solid #7258b6; background:#f6f3fc; color:#504963; font-size:12px; line-height:1.45; }
      .kpis { display: grid; grid-template-columns:repeat(5,1fr); gap: 8px; margin: 18px 0; }
      .kpi { border: 1px solid #e7e2ec; border-radius: 9px; padding: 10px; }.kpi:nth-child(2) .value { color:#cf4658; }.kpi:nth-child(3) .value { color:#c87a18; }.kpi:nth-child(4) .value { color:#21846d; }
      .kpi .label { font-size: 9px; color: #746d7c; text-transform: uppercase; }.kpi .value { margin-top:4px; font-size: 19px; font-weight: 700; color: #282330; }
      table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0; }
      th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #e2e8f0; }
      img { max-width: 100%; margin: 10px 0; }
      .priorities { display:grid; gap:7px; }.priority { padding:10px 12px; border:1px solid #e7e2ec; border-radius:8px; font-size:11px; }.priority b { color:#282330; }.priority small { color:#746d7c; }.tag { display:inline-block; padding:2px 5px; margin-right:6px; border-radius:4px; color:#9d3141; background:#ffebee; font-size:9px; font-weight:700; }
      .trust { padding:10px 12px; border-radius:8px; background:#f4fbf8; color:#326656; font-size:11px; line-height:1.45; }.disclaimer { font-size:10px; color: #746d7c; margin-top: 24px; border-top: 1px solid #e2e8f0; padding-top: 10px; }
      @media print { button { display: none; } }
    </style></head><body>
      <div class="eyebrow">Demo / Scenario report</div><h1>GeoQoE Pulse</h1>
      <div class="sub">${state.summary.study_area} &middot; Hour: ${new Date(cur.timestamp).toUTCString().replace(" GMT", " UTC")}</div>

      <div class="read"><b>Operational read:</b> ${operationalRead}</div>

      <div class="kpis">
        <div class="kpi"><div class="label">Overall QoE</div><div class="value">${cur.avg_qoe?.toFixed(0)}</div></div>
        <div class="kpi"><div class="label">Degraded Cells</div><div class="value">${degradedPct.toFixed(1)}%</div></div>
        <div class="kpi"><div class="label">Peak Scenario Pressure</div><div class="value">${peakDemand.toFixed(0)}</div></div>
        <div class="kpi"><div class="label">Population Affected</div><div class="value">${Math.round(popAffected).toLocaleString()}</div></div>
        <div class="kpi"><div class="label">Total Cells</div><div class="value">${C.toLocaleString()}</div></div>
      </div>

      <h2>Recommended operational focus</h2>
      <div class="priorities">${priorities.length ? priorities.map((p, i) => `<div class="priority"><span class="tag">P${i + 1}</span><b>${cellName(p.h3)}</b><br/><small>${CAUSE_LABELS[p.cause] ?? "Network concern"} · ${ACTIONS[p.cause] ?? "Review evidence and validate in the field."}</small></div>`).join("") : "<div class=\"priority\">No degraded cells meet the current investigation filter.</div>"}</div>

      <h3>Top 5 worst-QoE cells this hour</h3>
      <table>
        <tr><th>Location</th><th>QoE</th><th>Scenario Pressure</th><th>Hotspot class</th></tr>
        ${worst5.map(r => `<tr><td>${cellName(r.h3)}</td><td>${r.qoe?.toFixed(0)}</td><td>${r.demand?.toFixed(0)}</td><td>${state.panel.hotspot_code_labels[r.hotspot]}</td></tr>`).join("")}
      </table>

      <h3>QoE trend (selected range)</h3>
      <img src="${qoeImg}">

      <h3>Investigation signal distribution (degraded scenario cells)</h3>
      <img src="${rootCauseImg}">

      ${state.finals ? `
      <h2>AI forecast context</h2>
      <p>Finals model <code>${state.finals.metadata.model_version ?? "—"}</code> uses ${state.finals.metadata.predictor_quarter ?? "—"} context to rank ${state.finals.metadata.eligible_cells?.toLocaleString?.() ?? "—"} evaluable H3 cells for emerging underperformance in ${state.finals.metadata.forecast_quarter ?? "—"}. High risk is ranks 1–${state.finals.metadata.risk_thresholds?.high_max_rank ?? "—"} and elevated risk continues through rank ${state.finals.metadata.risk_thresholds?.elevated_max_rank ?? "—"}. This is a relative investigation ranking, not an individual probability or verified engineering explanation.</p>` : state.onset ? `
      <h2>AI forecast context</h2>
      <p>Onset model <code>${state.onset.model_name}</code> (frozen threshold ${state.onset.model_frozen_threshold.toFixed(3)}) scored at
      ${state.onset.snapshot_timestamp} UTC. Static spatial baseline: ${state.onset.static_observed_cells.toLocaleString()} cells with real
      Ookla labels, ${state.onset.static_ai_estimated_cells.toLocaleString()} AI-estimated. Onset risk scores are relative rankings within a
      simulated scenario, not calibrated probabilities — see the AI Model Info panel for validated precision/recall.</p>` : ""}

      <h2>Data confidence</h2><div class="trust"><b>${state.summary.ookla_anchored_pct}%</b> of cells have scenario QoE re-centred on real quarterly Ookla averages. GIS context uses the documented real static sources; hourly performance and Scenario Pressure remain scenario outputs.</div>

      <div class="disclaimer">${state.summary.disclaimer}<br/><br/>
        Investigation-signal percentages are a client-side heuristic computed from real static features
        (tower distance, built-up ratio, latency versus Scenario Pressure, sample confidence) — the pipeline
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
  renderPriorityActionCentre();
  renderSelectedAreaInsights();
  syncWorkspaceSelection();
  syncMapSelectedBar();
  syncMapWorkspaceInsight();
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
  document.addEventListener("keydown", event => {
    if ((event.key === "Enter" || event.key === " ") && event.target.matches(".ai-queue-item[role=button]")) {
      event.preventDefault();
      event.target.click();
    }
  });
  boot().then(() => { renderTrustPanel(); syncOperatorChartTheme(); }).catch(err => {
    console.error(err);
    document.body.insertAdjacentHTML("afterbegin",
      `<div style="background:#fee2e2;color:#7f1d1d;padding:12px;text-align:center">
        Failed to load dashboard data: ${err.message}. Make sure you're serving this
        folder over HTTP (e.g. <code>python3 -m http.server</code>) and that
        <code>data/</code> was generated by build_dashboard_data.py.
      </div>`);
  });
});
