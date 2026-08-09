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

FROM node:22-slim

WORKDIR /app

# Only what's needed to serve the dashboard — no pip/pandas/geopandas here.
COPY index.html dashboard.css dashboard.js server.js ./
COPY data/ ./data/

RUN test -f ./data/panel.json || \
    (echo "ERROR: data/panel.json missing — run build_dashboard_data.py before docker build" && exit 1)

EXPOSE 3000
CMD ["node", "server.js", "3000"]
