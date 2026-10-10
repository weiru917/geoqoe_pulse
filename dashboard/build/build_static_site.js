// Package the existing dashboard and saved context for static hosting.
// No pipeline, server module, credentials or AI/search requests are involved.
const fs = require("node:fs");
const path = require("node:path");

const ROOT = __dirname;
const CLIENT_FILES = ["index.html", "dashboard.css", "operator-theme.css", "dashboard.js", "map-experience.css", "map-experience.js"];
const DATA_FILES = ["h3_grid.geojson", "panel.json", "city_timeseries.json", "summary_stats.json", "quarterly_observed_2026Q1.geojson", "finals_quarterly.geojson", "finals_metadata.json"];

function savedContextIndex() {
  const dir = path.join(ROOT, "data", "context");
  const briefs = {};
  for (const name of fs.readdirSync(dir).sort()) {
    if (!/^[0-9a-f]{15}\.json$/.test(name)) continue;
    const brief = JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
    const h3 = name.replace(/\.json$/, "");
    if (brief.h3 && brief.h3 !== h3) throw new Error(`Context cell does not match filename: ${name}`);
    briefs[h3] = { ...brief, h3, source: brief.source === "manual" ? "manual" : "cache" };
  }
  return briefs;
}

function buildStaticSite(output = path.join(ROOT, "..", "..", ".pages-site")) {
  output = path.resolve(output);
  if ([ROOT, path.resolve(ROOT, "..", "..")].includes(output)) throw new Error("Choose a separate publication directory, not the source directory.");
  const briefs = savedContextIndex();
  const json = JSON.stringify(briefs, null, 2) + "\n";
  // Commit this index so both the local preview and public site show the same
  // saved briefs. Rebuild whenever individual source briefs are updated.
  fs.writeFileSync(path.join(ROOT, "data", "context", "index.json"), json);
  fs.mkdirSync(path.join(output, "data", "context"), { recursive: true });
  CLIENT_FILES.forEach(name => fs.copyFileSync(path.join(ROOT, name), path.join(output, name)));
  DATA_FILES.forEach(name => fs.copyFileSync(path.join(ROOT, "data", name), path.join(output, "data", name)));
  fs.writeFileSync(path.join(output, "data", "context", "index.json"), json);
  fs.writeFileSync(path.join(output, ".nojekyll"), "");
  // A standalone public dashboard repo uses this workflow. Only the client
  // allowlist above is published; the main research repo can remain private.
  const workflowDir = path.join(output, ".github", "workflows");
  fs.mkdirSync(workflowDir, { recursive: true });
  fs.copyFileSync(path.join(ROOT, "pages-workflow.yml"), path.join(workflowDir, "pages.yml"));
  fs.writeFileSync(path.join(output, "README.md"), "# GeoQoE Pulse dashboard\n\nStatic demo with saved forecasts and context. Live AI/search is disabled.\n\nEnable Settings → Pages → Source: GitHub Actions, then run the Deploy dashboard workflow.\n\nBuildings and map imagery are loaded from their credited geographic providers. No API credentials are included.\n");
  console.log(`Packaged ${Object.keys(briefs).length} saved briefs in ${output}`);
  return { output, briefs };
}

if (require.main === module) buildStaticSite(process.argv[2]);
module.exports = { buildStaticSite, savedContextIndex };
