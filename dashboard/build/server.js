// server.js — minimal static file server for the GeoQoE Pulse dashboard.
// No dependencies beyond Node's standard library. Data is pre-converted to
// JSON by build_dashboard_data.py; this just serves the dashboard folder.
//
// Run:  node server.js [port]
// Then open http://localhost:3000

const http = require("http");
const fs = require("fs");
const path = require("path");
const { buildContextBrief, providerStatus } = require("./context_brief");

const PORT = process.argv[2] || process.env.PORT || 3000;
const ROOT = __dirname;

const MIME = {
  ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
  ".json": "application/json", ".geojson": "application/json",
  ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
};

const MAX_BODY_BYTES = 64 * 1024;

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", chunk => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) { req.destroy(); reject(new Error("Request body too large")); return; }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}")); }
      catch { reject(new Error("Invalid JSON body")); }
    });
    req.on("error", reject);
  });
}

// POST /api/context-brief — retrieves public scheduled-event context for one cell
// and returns ITU-referenced next-actions. The API key stays in this process.
async function handleContextBrief(req, res, query) {
  let body;
  try { body = await readJsonBody(req); }
  catch (err) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: err.message }));
    return;
  }
  if (!body.h3) {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "h3 is required" }));
    return;
  }
  // Retrieval never throws to the client: a failure returns the same neutral
  // "no corroborated context" brief a genuinely empty result would.
  const brief = await buildContextBrief(body, { refresh: query.get("refresh") === "1" });
  res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(brief));
}

// GET /api/context-cache — every brief already retrieved and cached on disk.
// The dashboard loads this once at startup so pre-warmed cells render their
// context immediately, with no click and no API call. Cells that are NOT in
// here keep their "Find local context" button.
function handleContextCache(res) {
  const dir = path.join(ROOT, "data", "context");
  const briefs = {};
  let files = [];
  try { files = fs.readdirSync(dir); } catch { /* no cache yet */ }
  for (const name of files) {
    // The static index aggregates cells; it is not itself a cell brief.
    if (!/^[0-9a-f]{15}\.json$/.test(name)) continue;
    try {
      const brief = JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
      const h3 = brief.h3 || name.replace(/\.json$/, "");
      // Hand-compiled briefs keep their own source so the badge stays honest.
      briefs[h3] = { ...brief, source: brief.source === "manual" ? "manual" : "cache" };
    } catch (err) {
      console.warn(`[context-cache] skipping ${name}: ${err.message}`);
    }
  }
  res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(briefs));
}

const server = http.createServer((req, res) => {
  const [rawPath, rawQuery] = req.url.split("?");
  let urlPath = decodeURIComponent(rawPath);

  if (urlPath === "/api/context-cache") {
    handleContextCache(res);
    return;
  }

  if (urlPath === "/api/context-brief") {
    if (req.method !== "POST") {
      res.writeHead(405, { "Content-Type": "application/json", "Allow": "POST" });
      res.end(JSON.stringify({ error: "Use POST" }));
      return;
    }
    handleContextBrief(req, res, new URLSearchParams(rawQuery || "")).catch(err => {
      console.error("[context-brief] unhandled:", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Internal error" }));
    });
    return;
  }

  if (urlPath === "/") urlPath = "/index.html";
  const filePath = path.join(ROOT, urlPath);

  // Basic path-traversal guard
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403); res.end("Forbidden"); return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === "ENOENT" && urlPath.startsWith("/data/")) {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end(`${urlPath} not found — did you run build_dashboard_data.py yet?`);
        return;
      }
      res.writeHead(404); res.end("Not found"); return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`GeoQoE Pulse dashboard running at http://localhost:${PORT}`);
  if (!fs.existsSync(path.join(ROOT, "data", "panel.json"))) {
    console.log(`\nWARNING: data/panel.json not found. Run this first:\n  python3 build_dashboard_data.py\n`);
  }
  const context = providerStatus();
  console.log(context.ok
    ? `Context retrieval: LIVE via ${context.detail}. Cached briefs in data/context/ are served first; add ?refresh=1 to re-search.`
    : `Context retrieval: CACHE-ONLY — ${context.detail}. Uncached cells return 'no corroborated context'. See .env.example.`);
});
