# GeoQoE Pulse dashboard — user guide

## What this dashboard is for

GeoQoE Pulse helps a regulator, operator, or analyst answer four questions:

1. **REVEAL** — Which cells are already underperforming in the current real quarter?
2. **ANTICIPATE** — Which currently-normal cells are ranked as emerging-risk candidates for the next quarter?
3. **EXPLAIN** — Which measured patterns, geographic context, activity proxies, and model contributors support an investigation?
4. **PRIORITISE** — Which locations should be investigated first?

The dashboard combines two deliberately separate time bases:

- **Quarterly validated/finals data:** real quarterly network measurements, current observed status, and the frozen quarterly AI forecast.
- **Demo / Scenario data:** an hourly scenario used for map playback, hotspot exploration, time-series charts, and intervention examples. It is not live network demand and must not be read as another quarterly measurement.

The dashboard supports investigation and prioritisation. It does not diagnose a root cause, prove causality, or replace field/operator validation.

## Before you start

Open the dashboard through a local web server, not by double-clicking the HTML file. For example:

```bash
cd dashboard/build
python -m http.server 3000
```

Then open `http://localhost:3000`.

## The header and global controls

- **GeoQoE Pulse** — returns your attention to the dashboard identity and study area.
- **Region badge** — shows the study geography, currently Klang Valley, Malaysia.
- **Scenario timeline** — controls the selected hour for Demo / Scenario views. Moving the slider changes hourly scenario maps, charts, hotspots, and selected-cell scenario values. It does not retrain or rerun the quarterly AI model.
- **Play** — automatically moves through the hourly scenario timeline.
- **Start guided demo** — steps through the main workflow using the dashboard’s existing guided sequence.
- **Export briefing** — creates a briefing/report from the currently loaded dashboard information.
- **Dark-mode button** — switches the visual theme.
- **Status text** — shows the latest scenario timestamp currently loaded.

Use the help strip near the top if you need a short reminder of the data boundary. The quarterly labels are the trusted/finals context; the amber **DEMO / SCENARIO** labels identify hourly scenario content.

## Left navigation: the main tabs

### Reveal

Reveal is the executive starting point. It answers: **What is happening now, and where should I look first?**

The overview contains:

- **Observed underperformance** — cells classified as currently degraded or severe in the real quarterly observed-status layer.
- **High emerging risk** and **Elevated emerging risk** — counts of currently-normal cells ranked by the quarterly AI model for the next forecast quarter.
- **Coverage metadata** — measured cells, measurement-support coverage, and Demo / Scenario cell counts. These describe data coverage, not extra risk.
- **Top Priority** — the highest-ranked forecast-eligible location, its risk band, rank, and an **Inspect cell** button.
- **Hourly Pulse** — a secondary Demo / Scenario view showing hourly scenario behavior. Treat it as exploratory rather than validated quarterly evidence.
- **Selected area** — appears after choosing a map hexagon; when no cell is selected, the compact helper tells you to select one.

Use **Inspect cell** or click a hexagon to open the Cell Intelligence panel. Start with the current observed status, then read the quarterly AI priority and measurements.

### Compare

Compare lets you select multiple cells and compare them side by side.

How to use it:

1. Click a hexagon on the map.
2. Use the compare action when available to add the cell.
3. Select additional cells, up to the displayed limit.
4. Read the comparison table and the Scenario QoE trend.

The table separates **hourly scenario metrics** from **quarterly AI rank**. A cell can have strong current-hour scenario QoE while still having a high quarterly emerging-risk rank; these are different questions and time bases.

Use the remove controls in the comparison table to drop a selected cell. The map and selected-cell state stay synchronized.

### Hotspots

Hotspots is an hourly **DEMO / SCENARIO** view. It answers: **Which places look spatially unusual or stressed at the selected scenario hour?**

Controls and sections:

- **Gi* map** — displays the Getis-Ord hotspot classification.
- **Local Moran’s I map** — displays clusters and spatial outliers relative to neighboring cells.
- **List view** — returns to ranked lists.
- **Spatial Hotspot Summary** — gives the Local Moran’s I interpretation. It is not simply a list of low QoE values.
- **Lowest scenario QoE** — ranks cells with the lowest hourly scenario QoE.
- **Highest scenario pressure** — ranks cells with the highest hourly pressure proxy.
- **Persistent hotspot watch** — highlights sustained high-high scenario hotspots.
- **Emerging** — highlights sharp recent scenario deterioration before confirmed hotspot status.
- **Show more** — expands a ranking by 20 additional rows at a time.

The labels, colors, and counts are scenario outputs. They do not replace the quarterly observed-status layer.

### AI Forecast

AI Forecast is the quarterly **AI MODEL** view. It answers: **Among currently-normal cells, which are ranked for emerging underperformance next quarter?**

The page shows:

- forecast period and model context;
- filter chips for **High**, **Elevated**, and **Lower** risk bands with data-driven counts;
- a ranked investigation queue;
- **Time Machine**, which reveals held-out actual outcomes when available;
- a collapsed **How this forecast works** section.

Important interpretation rules:

- ranks order candidates; they are **not probabilities**;
- currently-underperforming cells are not forecast-eligible;
- the forecast is based on the frozen quarterly finals package;
- a high rank means “review earlier,” not “failure is certain.”

Select a ranked row to focus its map cell. After using **Reveal Actual**, the queue/map can show correctly anticipated, false alert, missed, or correctly lower-risk outcomes where the held-out fields exist.

### Explain

Explain is the hourly **DEMO / SCENARIO** investigation-evidence view. It answers: **What supporting signals are worth checking?**

It shows:

- degraded scenario-cell count;
- **Scenario measurement support**: No support, Limited, Moderate, or Stronger, based on confidence;
- independent signal bars for:
  - Sparse mapped-tower context;
  - Elevated relative latency;
  - Elevated activity-pressure proxy;
  - Built-environment intensity;
- the number of degraded cells with no strong relative signal;
- a concise key takeaway.

Signals use the citywide top quartile for the selected scenario hour. They are relative investigation thresholds, not engineering failure thresholds. Percentages do not need to total 100% because one cell may show multiple signals.

Click an evidence bar to filter the Intervention Priorities queue to cells carrying that signal. Measurement support is separate from the possible-driver signals.

### Prioritise

Prioritise is the hourly **DEMO / SCENARIO** investigation queue. It answers: **Where should an operator or regulator investigate first?**

Filters:

- **Priority:** All, High, Medium, Low.
- **Workflow:** All, Active, Investigating, Resolved.
- Evidence-category filters can be applied from Explain.
- Counts are calculated from the current scenario cells; they are not hard-coded.
- The toolbar remains available while scrolling.

Each card shows, in order:

1. rank and location;
2. scenario priority;
3. primary investigation hypothesis, if one is supported;
4. key evidence chips such as QoE, pressure, RTT, and population;
5. why the cell was prioritised;
6. one recommended next step.

Open **Operational detail** for measurement support, supporting signals, percentiles, quarterly AI context, persistent hotspot status, contact placeholder, and illustrative cost notes. These are investigation aids, not diagnosed causes or guaranteed actions.

Workflow buttons:

- **Active** — available for review;
- **Investigating** — someone is actively checking it;
- **Resolved** — the workflow item is closed locally in this browser.

### Trust

Trust explains where the information comes from and what each source is allowed to support.

Start with the compact summary:

- **REAL** — quarterly network measurements and real geographic sources;
- **SIMULATED / DEMO** — hourly scenario data;
- **AI MODEL** — frozen quarterly model outputs;
- **HEURISTIC / interpretation** — client-side investigation signals.

The source table identifies roles for Ookla, OpenCellID, terrain, built environment, population, roads/POIs, transit, VIIRS activity proxy, literature, the hourly scenario dataset, the AI model, and investigation heuristics.

Open the detailed provenance sections when you need methodology, validation boundaries, or scenario-construction notes. VIIRS is a human-activity proxy; it is not measured network demand.

## More tools

The **More tools** menu contains secondary views:

- **Time-series analysis** — hourly and daily scenario charts. Use FROM and TO to choose the window, then inspect QoE, scenario pressure, and severity trends. Hover points for details and click a point to jump the map to that hour.
- **AI vs Scenario Hotspots** — compares the quarterly AI ranking with hourly spatial hotspot detection. The two outputs have different purposes and must not be compared as if they were the same metric.
- **What-if simulation** — applies illustrative scenario assumptions such as extra capacity. Results are planning approximations, not field-validated intervention predictions.

## Map controls

- **Search place or cell** — type a place name or cell identifier and press Enter; select a match to move/focus the map.
- **Layers** — opens optional overlays such as quarterly status, QoE demo, demand, hotspots, H3 boundaries, and labels.
- **More map settings (⋯)** — contains detailed lens and QoE profile options. Profiles change the scenario lens; they do not change quarterly model outputs.
- **QoE / Demand / Hotspots / Quarterly** — mutually exclusive map lenses. Quarterly is the default validated mode.
- **Leaflet zoom controls** — zoom in/out without changing the selected cell.
- **Reveal Actual** — recolors forecast cells by held-out outcome when actual fields are available.

Clicking a hexagon opens the Cell Intelligence panel. The panel keeps the following order:

1. selected location and current observed status;
2. quarterly AI emerging-risk band and rank;
3. real quarterly Download, Upload, and Latency measurements;
4. why the model flagged it, including SHAP contributors where available;
5. measurement support and investigation evidence;
6. investigation priority and next check.

Hourly scenario values appear only inside clearly labelled **DEMO / SCENARIO** sections. Quarterly values appear under labels such as **REAL OOKLA** and **QUARTERLY AI**.

## Reading the selected-cell panel

- **Current status** tells you whether the real quarterly observed layer is normal, degraded, or severe.
- **AI emerging risk** gives a next-quarter rank/band only for currently-normal cells.
- **Download / Upload** are speed measurements; higher is generally better.
- **Latency / RTT** describe responsiveness; lower is generally better.
- **Percentiles and historical rates** provide context against the relevant quarterly distribution or history.
- **SHAP contributors** are factors contributing to the model’s risk assessment. They are not root causes.
- **Measurement support** describes available scenario measurement confidence/coverage, separately from the investigation signals.
- **Investigation priority** is a review order, not a final engineering decision.

Use **How to read these values** and expandable technical sections when you need definitions. Keep the main card for the conclusion first; use details to support a conversation with operators or field teams.

## A practical review workflow

1. Start in **Reveal** and note observed issues, high/elevated emerging risk, and the top priority.
2. Click a map cell or use **Inspect cell**.
3. Read current quarterly status and real quarterly measurements first.
4. Open **AI Forecast** for the next-quarter rank, remembering that rank is not probability.
5. Open **Explain** to see relative scenario signals and measurement support.
6. Open **Prioritise** and filter the queue by priority, workflow state, or evidence category.
7. Use **Hotspots**, **Compare**, and **Time-series analysis** to inspect hourly scenario context separately.
8. Use **Trust** before quoting a number externally.
9. Record field/operator validation in the workflow status; do not treat a heuristic signal as a confirmed cause.

