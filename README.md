# GeoQoE Pulse

**A mobile network quality decision-support prototype for the Klang Valley (Kuala Lumpur & surrounding urban Selangor).**

GeoQoE Pulse takes real, open geospatial and network data, combines it into
one map, and helps a telecom operator or regulator answer two questions:

1. **Where is mobile internet quality currently bad, and why?**
2. **Where is it likely to get bad next, and what should we do about it?**

This document is the entry point for judging — what the prototype is,
how the pieces fit together, and the fastest way to open it.

---

## 1. What am I looking at?

This submission has three layers that build on each other:

| Layer | What it does | Where it lives |
|---|---|---|
| **① Data pipeline** | Pulls real open data (cell towers, buildings, roads, population, transit, and real Ookla speed-test data) for ~4,900 hexagonal map cells across the Klang Valley, and turns it into one clean, analysis-ready dataset. | `src/`, `data/`, `run_pipeline.py` |
| **② AI models** | Two trained machine-learning models sit on top of that data: one estimates network quality in cells with no direct measurement, and one ranks cells by risk of degrading in the next hour. | `Trained AI Model/` |
| **③ Interactive dashboard** | A map-based web app where a judge can explore quality, hotspots, AI predictions, and recommended interventions, hour by hour. | `dashboard/` |

**The dashboard (layer ③) is the prototype itself — that's what to open.**
Layers ① and ② are the data/AI engine behind it, included so judges can see
the full pipeline, not just the front end.

---

## 2. How it works, in plain terms

1. **Build the map grid.** The Klang Valley is divided into ~4,900
   hexagonal cells (H3 grid, about the size of a few city blocks each).
2. **Attach real-world context to every cell.** For each cell: how many
   cell towers, how dense the buildings are, how many roads/POIs/transit
   stops, how many people live and work there, and — where available — a
   real quarterly average internet speed from Ookla's open speed-test
   dataset (~60% of cells have this).
3. **Score network quality.** A custom 0–100 Quality-of-Experience (QoE)
   index is calculated from download/upload speed, latency, jitter, and
   packet loss, and cells are flagged as "degraded" against fixed
   thresholds (e.g. download < 10 Mbps or latency > 100 ms).
4. **Find hotspots.** Real spatial statistics (Getis-Ord Gi\*, Local
   Moran's I) identify statistically significant clusters of poor quality,
   not just individually bad cells.
5. **Predict risk.** Two trained XGBoost/logistic models — one estimating
   quality for cells with no direct measurement, one ranking cells by
   next-hour degradation risk — sit on top of the real data.
6. **Show it all on one dashboard.** A judge can scrub through time,
   change what the map is colored by, compare cells, simulate "what if we
   added X% more capacity," and see a ranked, prioritized list of
   recommended interventions.

---

## 3. How to open it (for judging)

The dashboard is a self-contained web app — **no Python environment or
installation required.** All the data it needs has already been generated
and sits in `dashboard/build/data/`.

```bash
cd dashboard/build
python3 -m http.server 3000
```

Then open **http://localhost:3000** in a browser.

No Python on hand? Two alternatives that need nothing else installed:

```bash
# Node.js (a zero-dependency server is already included)
node server.js 3000

# Docker
docker build -t geoqoe-pulse .
docker run --rm -p 3000:3000 geoqoe-pulse
```

> Opening `dashboard/build/index.html` directly by double-clicking it will
> **not** work — browsers block the page from loading its data files that
> way, hence the small local server above.

### Alternative: the Streamlit version

A second, Python-based view of the same underlying pipeline is available
at the repo root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This is a simpler, more analytical view (filters, tables, charts) built
directly from the pipeline output — useful for judges who want to see the
raw pipeline data outside of the map-first dashboard.

### What to click on first

- Start with the **"How to use this dashboard"** panel at the top — it's
  written for a first-time viewer.
- Use the **time slider** to scrub through a full simulated week.
- Try the **🤖 Trained AI Model** panel for the model's live risk rankings.
- Try **Color by → AI Onset Risk** on the map, and open a cell's popup.
- Check the **Data Quality & Trust** panel — every panel is badged
  REAL / SIMULATED / HEURISTIC so it's always clear what you're looking at.

---

## 4. What's real, and what's a demo stand-in (read before quoting a number)

We'd rather over-explain this than have it come up as a surprise question.

| Data | Status |
|---|---|
| Cell tower locations, buildings, roads/POIs/transit, population | ✅ **Real.** Sourced from OpenCellID, Google Open Buildings, OpenStreetMap, WorldPop, and OpenDOSM, joined with real GIS operations. |
| Quarterly average download/upload speed & latency (~60% of cells) | ✅ **Real.** From Ookla's Open Data program. |
| **Hour-by-hour** network quality for the 168-hour (1-week) panel shown in the dashboard's time slider | ⚠️ **Simulated.** No public source of real per-hour QoE exists for this area, so hourly values are generated by a documented diurnal-demand formula. Where a real Ookla number exists for a cell, the simulation is re-centered on it — but the hour-to-hour *shape* is always simulated. This is clearly labelled in the dashboard, never presented as measured. |
| The two AI models (static quality estimator + next-hour onset risk ranker) | ✅ **Really trained, on real held-out data.** The static model is trained/validated against the real Ookla labels. The onset-risk model is trained on the simulated hourly panel (see above) and evaluated on a real held-out split of it — its own validated precision/recall numbers are shown in the dashboard, including the honest trade-off (e.g. high recall, low precision, stated plainly rather than hidden). |
| Root-cause labels, cost estimates, operator contacts on intervention cards | 🔧 **Heuristic / illustrative placeholders**, computed transparently from real features — not pipeline-measured facts. Clearly labelled as such in the UI. |

**Bottom line for judges asking "is this real data?":** the static spatial
layer (towers, buildings, population, real Ookla speeds) is 100% real and
is what the AI static-quality model is trained and validated on. The
animated hour-by-hour view is an honestly-labelled simulation used to
demonstrate what the dashboard and forecasting pipeline would look like
running on live data.

---

## 5. Repository map

```
geoqoe_pulse/
├── README.md                  ← you are here
├── app.py                     ← Streamlit dashboard (alternative view)
├── config.py                  ← pipeline settings (CRS, thresholds, seed)
├── run_pipeline.py            ← runs the full data pipeline, 10 stages
├── data_loader.py             ← loads pipeline output for app.py
├── requirements.txt
│
├── data/                      ← pipeline inputs & outputs
│   ├── boundary/               study area definition
│   ├── processed/               H3 grid geometry
│   └── outputs/                 final datasets used by the dashboard & models
│
├── src/                       ← pipeline source code
│   ├── data_preprocessing/      geometry, joins, feature engineering, QoE targets
│   └── modelling/                baseline training scripts, reports, plots
│
├── Trained AI Model/          ← the two trained AI models + training code
│   └── GeoQoE_AI/
│       ├── code/                 training/evaluation scripts
│       ├── models/               saved model files
│       ├── competition_package/  frozen model card + latest predictions
│       └── docs/                 model training write-up
│
└── dashboard/build/           ← ★ the prototype — open this ★
    ├── index.html               entry point
    ├── dashboard.js / .css      app logic & styling
    ├── server.js, Dockerfile    ways to serve it locally
    ├── data/                     pre-built JSON/GeoJSON the dashboard reads
    ├── ai_model/                 frozen AI model outputs used by the dashboard
    └── build_dashboard_data.py   regenerates data/ from a fresh pipeline run
```

---

## 6. Known limitations

- No real per-hour network measurements exist for this area — the hourly
  panel is a labelled simulation, not observed data (see section 4).
- No real network capacity/traffic data exists — "demand pressure" is an
  activity proxy, not confirmed congestion.
- The custom 0–100 QoE index is our own composite, not an ITU-T G.107/P.1203
  standard.
- Root cause, cost, and operator-contact fields on intervention cards are
  editable illustrative placeholders, not measured pipeline outputs.
- `dashboard/build/data/panel.json` is ~54 MB and loads once on page load —
  fine for a local judged demo, a few seconds on a slow connection.
- The Docker path is written to standard conventions but was untested in
  the build environment (no Docker/network access there) — worth a quick
  local test before relying on it during judging.

Full technical detail on every pipeline stage, feature, and design decision
lives in `src/modelling/` reports and `dashboard/build/README.md` for
anyone who wants to go deeper than this overview.
