// server.js — minimal static file server for the GeoQoE Pulse dashboard.
// No dependencies beyond Node's standard library. Data is pre-converted to
// JSON by build_dashboard_data.py; this just serves the dashboard folder.
//
// Run:  node server.js [port]
// Then open http://localhost:3000

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.argv[2] || process.env.PORT || 3000;
const ROOT = __dirname;

const MIME = {
  ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
  ".json": "application/json", ".geojson": "application/json",
  ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
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
});
