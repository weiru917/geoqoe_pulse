# GeoQoE Pulse dashboard — deployment image.
#
# This image serves the ALREADY-BUILT static dashboard (index.html, dashboard.js,
# dashboard.css, data/*.json). It deliberately does NOT run the data pipeline or
# build_dashboard_data.py inside the container — those need pandas/geopandas/h3
# and the raw multi-GB source files, which don't belong in a lightweight demo
# image. Build the data once on your machine first:
#
#   python3 build_dashboard_data.py      # regenerates dashboard/data/*.json
#   docker build -t geoqoe-pulse .
#   docker run --rm -p 3000:3000 geoqoe-pulse
#
# Then open http://localhost:3000
#
# Live context retrieval needs credentials. The .env file is NOT baked into the
# image (it is gitignored and must not ship in a layer) — pass it at run time:
#
#   docker run --rm -p 3000:3000 --env-file .env geoqoe-pulse
#
# Without it the dashboard still runs: cached briefs in data/context/ are served
# and uncached cells return "no corroborated upcoming context found".

FROM node:22-slim

WORKDIR /app

# Only what's needed to serve the dashboard — no pip/pandas/geopandas here.
COPY index.html dashboard.css dashboard.js server.js ./
COPY operator-theme.css map-experience.css map-experience.js ./
# Server-side context retrieval. Without these the server cannot start:
# server.js require()s context_brief.js on the first line.
COPY context_brief.js regulator_actions.json ./
COPY data/ ./data/

RUN test -f ./data/panel.json || \
    (echo "ERROR: data/panel.json missing — run build_dashboard_data.py before docker build" && exit 1)

EXPOSE 3000
CMD ["node", "server.js", "3000"]
