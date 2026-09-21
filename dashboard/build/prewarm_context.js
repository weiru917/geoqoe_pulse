// prewarm_context.js — retrieve and cache local context for the top priority
// cells, one at a time, before a demo.
//
// Each cell is one API call, exactly as one button click would be. Requests are
// strictly sequential with a pause between them: never concurrent, so this stays
// inside per-minute provider limits and a failure on one cell cannot cascade.
//
//   node prewarm_context.js                      # top 10 forecast cells
//   node prewarm_context.js --top 5              # fewer
//   node prewarm_context.js --delay 10000        # longer pause between calls
//   node prewarm_context.js --observed           # observed 2026Q1 issues instead
//   node prewarm_context.js --refresh            # re-search cells already cached
//   node prewarm_context.js --dry-run            # list the cells, call nothing
//
// The server does not need to be running — this uses the same module and writes
// the same cache files that the dashboard reads.

const fs = require("fs");
const path = require("path");
const { buildContextBrief } = require("./context_brief");

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] && !args[i + 1].startsWith("--") ? args[i + 1] : fallback;
};
const has = name => args.includes(`--${name}`);

const TOP = Math.max(1, Number(flag("top", 10)));
const DELAY_MS = Math.max(0, Number(flag("delay", 5000)));
const OBSERVED = has("observed");
const REFRESH = has("refresh");
const DRY_RUN = has("dry-run");

const DATA = path.join(__dirname, "data");
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function loadCells() {
  const read = name => JSON.parse(fs.readFileSync(path.join(DATA, name), "utf8"));
  const grid = new Map(read("h3_grid.geojson").features.map(f => [f.properties.h3_index, f.properties]));
  const meta = read("finals_metadata.json");

  // The two queues come from different files: finals_quarterly.geojson holds
  // only the forecast-eligible (currently-normal) cells, so observed issues are
  // not in it at all and must be read from the observed-quarter layer.
  const selected = OBSERVED
    ? read("quarterly_observed_2026Q1.geojson").features.map(f => f.properties)
        .filter(r => r.current_status === "currently_underperforming")
        // Not forecast-ranked; ordered the way the dashboard orders them —
        // worst download percentile first, then population.
        .sort((a, b) => (a.dl_percentile_in_quarter ?? Infinity) - (b.dl_percentile_in_quarter ?? Infinity)
          || (b.population_day ?? 0) - (a.population_day ?? 0))
    : read("finals_quarterly.geojson").features.map(f => f.properties)
        .filter(r => r.risk_rank != null && ["high", "elevated"].includes(r.risk_band))
        .sort((a, b) => a.risk_rank - b.risk_rank);

  return selected.slice(0, TOP).map(r => {
    const props = grid.get(r.h3_index) || {};
    return {
      h3: r.h3_index,
      cellName: props.place_name || r.h3_index.slice(-6),
      lat: props.centroid_lat,
      lon: props.centroid_lon,
      forecastQuarter: meta.forecast_quarter ?? "2026Q2",
      predictorQuarter: meta.predictor_quarter ?? "2026Q1",
      cell: {
        observedStatus: r.current_status === "currently_underperforming" ? "Underperforming"
          : r.current_status === "currently_not_underperforming" ? "Not underperforming" : "Not measured",
        riskBand: r.risk_band_label ?? null,
        riskRank: r.risk_rank == null ? null : Math.round(r.risk_rank),
        eligibleCells: meta.eligible_cells ?? null,
        downloadMbps: r.ookla_dl_mbps ?? null,
        uploadMbps: r.ookla_ul_mbps ?? null,
        latencyMs: r.ookla_rtt_ms ?? null,
        measurementSupport: r.measurement_support ?? null,
        populationDay: r.population_day ?? null,
        populationDensityPerKm2: r.population_density_per_km2 ?? null,
        towerDensityPerKm2: r.tower_density_per_km2 ?? null,
        builtUpRatio: r.built_up_ratio ?? null,
        topContributor: r.risk_driver_1_label ?? null,
      },
    };
  });
}

(async () => {
  const cells = loadCells();
  const label = OBSERVED ? "observed 2026Q1 issues" : "forecast priority cells";
  console.log(`Pre-warming ${cells.length} ${label}, sequentially, ${DELAY_MS} ms apart.`);
  if (DRY_RUN) {
    cells.forEach((c, i) => console.log(`  [${i + 1}] ${c.cellName}  (${c.h3})`));
    console.log("\n--dry-run: nothing was called.");
    return;
  }

  let found = 0, empty = 0;
  for (const [i, cell] of cells.entries()) {
    const t0 = Date.now();
    const brief = await buildContextBrief(cell, { refresh: REFRESH });
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    const n = brief.contextItems?.length ?? 0;
    if (n) found++; else empty++;
    const source = brief.source === "cache" ? "already cached" : `${n} item(s)`;
    console.log(`  [${i + 1}/${cells.length}] ${String(cell.cellName).padEnd(34)} ${source.padEnd(15)} (${secs}s)`);
    if (brief.note) console.log(`        note: ${brief.note}`);
    if (i < cells.length - 1) await sleep(DELAY_MS);
  }

  console.log(`\nDone. ${found} cell(s) with context, ${empty} without. Cache: data/context/`);
  if (empty === cells.length) {
    console.log("Every cell came back empty — check the server log hints above; this usually means");
    console.log("the provider rejected the requests rather than that there are genuinely no events.");
  }
  console.log("Verify each source_url before presenting these as corroborated context.");
})();
