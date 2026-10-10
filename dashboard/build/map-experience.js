/* All presentation views use the existing data, filters and selection.
   Optional data-column heights encode metrics; buildings and terrain use map sources. */
(() => {
  let map, ready = false, mode = "2d", frame, popup, failed = false;
  let rows = [], signature = "", fallbackMessage = "";
  let buildingsEnabled = true, terrainEnabled = true, columnsEnabled = false;
  let columnOpacity = .18;
  const columnHeightScale = .22;
  let buildingsFailed = false, terrainFailed = false;
  let cameraReady = false;
  const cityCamera = { center: [101.7116, 3.1578], zoom: 15.5, pitch: 60, bearing: -28 };
  const $ = id => document.getElementById(id);
  const duration = () => matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 650;
  const clamp = v => Math.max(0, Math.min(1, Number(v) || 0));
  const heatSupported = () => !["moran", "comparison"].includes(state.colorBy) && !(state.colorBy === "onset" && state.finalsActualRevealed);

  function encoding(c) {
    const t = state.currentT;
    if (state.colorBy === "onset") {
      const status = onsetValue("current_status", c);
      if (state.finalsActualRevealed && state.quarterlyMapView === "combined") return { height: onsetValue("time_machine_result", c) ? 600 : 0, weight: 0 };
      if (state.quarterlyMapView !== "forecast" && status === "currently_underperforming") return { height: 900, weight: 1 };
      if (state.quarterlyMapView === "observed") return { height: status === "currently_not_underperforming" ? 35 : 0, weight: 0 };
      return ({ high: { height: 1400, weight: 1 }, elevated: { height: 600, weight: .45 }, lower: { height: 50, weight: 0 } })[onsetValue("onset_priority_tier", c)] || { height: 0, weight: 0 };
    }
    if (["qoe", "demand"].includes(state.colorBy)) {
      const metrics = effectiveMetrics(t, c), value = state.colorBy === "qoe" ? metrics.qoe : metrics.demand;
      const weight = value == null ? 0 : clamp((state.colorBy === "qoe" ? 100 - value : value) / 100);
      return { height: value == null ? 0 : 30 + weight * 1800, weight };
    }
    if (state.colorBy === "hotspot") {
      const code = cellValue("hotspot_code", t, c);
      return { height: [4, 5, 6].includes(code) ? 850 : 35, weight: ({ 4: .4, 5: .7, 6: 1 })[code] || 0 };
    }
    if (state.colorBy === "risk") {
      const value = cellValue("next_hour_degraded_historical", t, c);
      return { height: value == null ? 0 : value === 1 ? 900 : 35, weight: value === 1 ? 1 : 0 };
    }
    if (state.colorBy === "static_ai") {
      const value = onsetValue("display_dl_mbps", c), weight = value == null ? 0 : clamp(value / 300);
      return { height: value == null ? 0 : 30 + weight * 1800, weight };
    }
    return { height: 500, weight: 0 }; // equal-height categorical maps
  }

  function description() {
    if (state.colorBy === "onset") return state.quarterlyMapView === "observed" ? "observed Q1 underperformance" : state.quarterlyMapView === "forecast" ? "emerging-risk priority bands" : state.finalsActualRevealed ? "held-out outcome categories" : "observed issues and emerging-risk priorities";
    return ({ qoe: "scenario QoE shortfall", demand: "scenario pressure", hotspot: "scenario Gi* hot classifications", risk: "historical scenario degradation", static_ai: "download speed", moran: "Moran cluster categories", comparison: "detection categories" })[state.colorBy] || "network signals";
  }

  function chrome(message) {
    $("mapStage").dataset.style = mode;
    document.querySelectorAll("[data-map-style]").forEach(button => {
      const active = button.dataset.mapStyle === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
      button.disabled = button.dataset.mapStyle === "heat" ? !heatSupported() || failed : button.dataset.mapStyle === "3d" && failed;
      button.title = button.dataset.mapStyle === "heat" && !heatSupported() ? "Use Clean 2D or 3D to preserve outcome categories" : "";
    });
    $("mapCameraControls").hidden = mode !== "3d";
    $("mapEnvironmentControls").hidden = mode !== "3d";
    $("mapEnvironmentStatus").hidden = mode !== "3d";
    $("mapColumnOpacityControl").hidden = mode !== "3d" || !columnsEnabled;
    $("mapSceneTitle").textContent = mode === "3d" ? "Explore the city behind the network." : mode === "heat" ? "See the patterns beyond the cells." : "Every area. A clearer picture.";
    $("mapSceneHint").textContent = mode === "3d" ? "Drag to pan · Right-drag to orbit · Click the map for cell evidence" : mode === "heat" ? "Smoothed regional view · Click an area for exact evidence" : "Click an area to investigate · Scroll to zoom";
    let note = "A quiet basemap keeps geography readable. Colour shows the active metric; select an area for exact evidence.";
    if (mode === "3d" && !columnsEnabled) {
      note = "Blue/teal buildings show mapped height; ground colours show network signals. Terrain follows ground elevation. Building detail and heights vary with source coverage.";
    } else if (mode === "3d") {
      if (state.colorBy === "onset") note = state.quarterlyMapView === "observed" ? "Raised columns mark observed underperformance; normal cells stay near the ground." : state.finalsActualRevealed && state.quarterlyMapView === "combined" ? "Equal-height columns preserve outcome categories; colour identifies the result." : state.quarterlyMapView === "forecast" ? "High-risk bands rise tallest, then Elevated. Lower-risk areas stay flat; height encodes bands, not individual ranks." : "High-risk bands rise tallest, then Elevated. Existing observed issues have a separate fixed height.";
      else if (["moran", "comparison"].includes(state.colorBy)) note = "Equal-height columns preserve categories; colour identifies the category.";
      else if (state.colorBy === "hotspot") note = "Raised columns mark Gi* hot classifications; height does not measure their significance.";
      else note = `Taller columns = ${state.colorBy === "qoe" ? "lower scenario QoE" : state.colorBy === "static_ai" ? "higher download speed" : "stronger " + description()}.`;
      note += " Hollow, translucent columns keep the city visible; adjust Column opacity for contrast. Blue/teal buildings show mapped height. Column heights use a reduced display scale and are illustrative, not building heights or probabilities.";
    } else if (mode === "heat") note = `Smoothed, zoom-dependent concentration of ${description()}. A visual summary, not a new statistical hotspot test or probability map. Click for exact cell evidence.`;
    $("mapEncodingNote").innerHTML = `${mode === "heat" ? '<span class="heat-scale"><span></span>Less concentration → More</span>' : '<i class="bi bi-info-circle" aria-hidden="true"></i>'}<span>${escapeHtml(message || fallbackMessage || note)}</span>`;
    $("mapLegend").hidden = mode === "heat";
    ["toggleWeather", "toggleTowers"].forEach(id => { $(id).disabled = mode !== "2d"; $(id).closest("label").title = mode === "2d" ? "" : "Available in Clean 2D"; });
  }

  function style() {
    const labels = state.basemap === "labels";
    const satellite = state.basemap === "satellite";
    const base = { version: 8, sources: { basemap: { type: "raster", tileSize: 256, tiles: [labels ? "https://tile.openstreetmap.de/{z}/{x}/{y}.png" : satellite ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" : "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"], attribution: labels ? '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' : satellite ? "Imagery © Esri, Maxar, Earthstar Geographics" : "Tiles © Esri", maxzoom: labels || satellite ? 18 : 16 } }, layers: [{ id: "background", type: "background", paint: { "background-color": "#e7ecf0" } }, { id: "basemap", type: "raster", source: "basemap", paint: { "raster-saturation": satellite ? -.15 : -.65, "raster-opacity": satellite ? 1 : .88 } }], light: { anchor: "viewport", color: "#ffffff", intensity: .65, position: [1.5, 210, 35] } };
    if (!labels) {
      base.sources.placeLabels = { type: "raster", tileSize: 256, tiles: [satellite ? "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}" : "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"], maxzoom: satellite ? 18 : 16 };
      base.layers.push({ id: "place-labels", type: "raster", source: "placeLabels" });
    }
    return base;
  }

  function addLayers() {
    signature = "";
    map.addSource("city-buildings", { type: "vector", url: "https://tiles.openfreemap.org/planet" });
    map.addSource("terrain-dem", { type: "raster-dem", tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"], tileSize: 256, maxzoom: 14, encoding: "terrarium", attribution: 'Terrain: <a href="https://registry.opendata.aws/terrain-tiles/">Mapzen / AWS Open Data</a>; SRTM / GMTED2010: USGS; ETOPO1: NOAA' });
    const empty = { type: "FeatureCollection", features: [] };
    ["network-cells", "network-points", "network-selection", "network-column-shells", "network-column-selection"].forEach(id => map.addSource(id, { type: "geojson", data: empty }));
    map.addLayer({ id: "network-ground", source: "network-cells", type: "fill", paint: { "fill-color": ["get", "color"], "fill-opacity": ["case", [">", ["get", "height"], 50], .32, .12] } });
    map.addLayer({ id: "city-buildings-3d", source: "city-buildings", "source-layer": "building", type: "fill-extrusion", minzoom: 13.5, filter: ["!=", ["get", "hide_3d"], true], layout: { visibility: "none" }, paint: {
      "fill-extrusion-color": ["interpolate", ["linear"], ["coalesce", ["get", "render_height"], 0], 0, "#64748b", 40, "#426882", 100, "#22637f", 250, "#147d8e", 450, "#0c6677"],
      "fill-extrusion-height": ["coalesce", ["get", "render_height"], 0],
      "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],
      "fill-extrusion-opacity": 1, "fill-extrusion-vertical-gradient": true,
    } });
    map.addLayer({ id: "network-columns", source: "network-column-shells", type: "fill-extrusion", filter: [">", ["get", "height"], 50], paint: { "fill-extrusion-color": ["get", "color"], "fill-extrusion-height": ["*", ["get", "height"], columnHeightScale], "fill-extrusion-opacity": columnOpacity, "fill-extrusion-vertical-gradient": false } });
    map.addLayer({ id: "network-column-footprints", source: "network-cells", type: "line", paint: { "line-color": ["get", "color"], "line-opacity": .65, "line-width": 1.2 } });
    map.addLayer({ id: "network-heat", source: "network-points", type: "heatmap", paint: { "heatmap-weight": ["get", "weight"], "heatmap-intensity": 1, "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 8, 15, 11, 28, 14, 46, 17, 65], "heatmap-opacity": .82, "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(28,110,147,0)", .15, "#35a7bd", .35, "#7195d8", .55, "#bd71ca", .75, "#f28c63", 1, "#b92746"] } });
    map.addLayer({ id: "network-hit-targets", source: "network-cells", type: "fill", paint: { "fill-opacity": 0 } });
    map.addLayer({ id: "network-selected", source: "network-selection", type: "line", paint: { "line-color": "#5b21b6", "line-width": 3 } });
    map.addLayer({ id: "network-selected-top", source: "network-column-selection", type: "fill-extrusion", filter: [">", ["get", "height"], 50], paint: { "fill-extrusion-color": "#7c3aed", "fill-extrusion-height": ["+", ["*", ["get", "height"], columnHeightScale], 6], "fill-extrusion-base": ["*", ["get", "height"], columnHeightScale], "fill-extrusion-opacity": columnOpacity, "fill-extrusion-vertical-gradient": false } });
    ready = true;
    renderNow();
  }

  function environmentStatus() {
    if (!map || mode !== "3d" || !ready) return;
    let buildings = buildingsFailed ? "Building tiles unavailable" : !buildingsEnabled ? "Buildings off" : map.getZoom() < 13.5 ? "Zoom in for buildings, or choose KL skyline" : map.isSourceLoaded("city-buildings") ? "Mapped buildings loaded" : "Loading building tiles…";
    if (buildingsEnabled && !buildingsFailed && map.getZoom() >= 13.5 && map.isSourceLoaded("city-buildings") && !map.queryRenderedFeatures(undefined, { layers: ["city-buildings-3d"] }).length) buildings = "No mapped buildings in this view";
    const terrain = terrainFailed ? "Terrain tiles unavailable" : !terrainEnabled ? "Terrain off" : !map.getTerrain() ? "Preparing terrain…" : map.isSourceLoaded("terrain-dem") ? "Elevation tiles loaded · true scale" : "Loading elevation tiles…";
    $("mapEnvironmentStatus").textContent = `${buildings} · ${terrain}`;
  }

  function syncEnvironment() {
    if (!ready) return;
    const terrain = cameraReady && mode === "3d" && terrainEnabled && !terrainFailed;
    if (Boolean(map.getTerrain()) !== terrain) map.setTerrain(terrain ? { source: "terrain-dem", exaggeration: 1 } : null);
    map.setLayoutProperty("city-buildings-3d", "visibility", mode === "3d" && buildingsEnabled && !buildingsFailed ? "visible" : "none");
    environmentStatus();
  }

  function ensureMap() {
    if (map) return true;
    if (!window.maplibregl) { fallback("3D / heatmap could not load. Clean 2D remains available; check your connection and reload."); return false; }
    try {
      const center = state.map.getCenter();
      // MapLibre uses a 512px world tile; Leaflet uses 256px. Offset zoom
      // by one level to preserve geographic scale when switching renderers.
      map = new maplibregl.Map({ container: "mapSpatial", style: style(), center: [center.lng, center.lat], zoom: state.map.getZoom() - 1, pitch: mode === "3d" ? 60 : 0, bearing: mode === "3d" ? -28 : 0, maxPitch: 70, minZoom: 7, maxZoom: 18, attributionControl: false });
      map.addControl(new maplibregl.AttributionControl(), "bottom-right");
      map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
      map.addControl(new maplibregl.ScaleControl({ maxWidth: 100 }), "bottom-left");
      map.on("style.load", addLayers);
      map.on("load", () => { cameraReady = true; syncEnvironment(); });
      // A failed optional building source can prevent the overall load event.
      // Start terrain after the basemap has rendered in that case as well.
      map.on("render", () => {
        if (!cameraReady && ready && map.isSourceLoaded("basemap")) {
          cameraReady = true;
          requestAnimationFrame(syncEnvironment);
        }
      });
      map.on("error", event => {
        console.warn("Spatial map resource error", event.error);
        if (/webgl|context lost/i.test(event.error?.message || "")) fallback("Your browser could not render 3D. Clean 2D remains available.");
        else if (event.sourceId === "city-buildings" || /openfreemap/.test(event.error?.url || "")) { buildingsFailed = true; $("mapBuildingsEnabled").checked = false; $("mapBuildingsEnabled").disabled = true; syncEnvironment(); }
        else if (event.sourceId === "terrain-dem" || /elevation-tiles/.test(event.error?.url || "")) { terrainFailed = true; $("mapTerrainEnabled").checked = false; $("mapTerrainEnabled").disabled = true; syncEnvironment(); }
      });
      map.on("idle", environmentStatus);
      map.on("zoomend", environmentStatus);
      map.getCanvas().addEventListener("webglcontextlost", () => fallback("Graphics context lost. Switched to Clean 2D."));
      const hit = event => {
        if (!ready) return null;
        const features = map.queryRenderedFeatures(event.point, { layers: mode === "3d" && columnsEnabled ? ["network-columns", "network-hit-targets"] : ["network-hit-targets"] });
        return features.find(f => f.layer.id === "network-columns") || features[0];
      };
      map.on("click", event => { const feature = hit(event); if (!feature) return; popup?.remove(); if (state.compareMode) toggleCompareCell(feature.properties.h3); else selectArea(feature.properties.h3); });
      map.on("mousemove", event => {
        const feature = hit(event);
        map.getCanvas().style.cursor = feature ? "pointer" : "";
        if (!feature) { popup?.remove(); return; }
        popup ||= new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: "spatial-cell-tooltip", offset: 15, maxWidth: "290px" });
        popup.setLngLat(event.lngLat).setHTML(hoverHtml(feature.properties.h3)).addTo(map);
      });
      map.getCanvas().addEventListener("mouseleave", () => popup?.remove());
      return true;
    } catch (error) { console.warn("Spatial map unavailable", error); fallback("This browser cannot start the 3D map. Clean 2D remains available."); return false; }
  }

  function fallback(message) { failed = true; fallbackMessage = message; setMode("2d"); }
  function setMode(next) {
    if (next === "heat" && !heatSupported()) return;
    state.map.stop(); // cancel a 2D fly-to before its container is hidden
    const previous = mode;
    mode = next;
    $("map").hidden = mode !== "2d";
    $("mapSpatial").hidden = mode === "2d";
    popup?.remove();
    if (mode === "2d") {
      state.map.invalidateSize();
      if (previous !== "2d" && map) { const center = map.getCenter(); state.map.setView([center.lat, center.lng], map.getZoom() + 1, { animate: false }); }
    } else if (ensureMap()) {
      map.resize();
      const center = previous === "2d" ? state.map.getCenter() : map.getCenter();
      if (mode === "3d" && previous !== "3d") {
        const props = state.gridPropsByH3.get(state.selectedH3);
        map.easeTo({ ...cityCamera, ...(props ? { center: [props.centroid_lon, props.centroid_lat] } : {}), duration: duration() });
      } else map.easeTo({ center: [center.lng, center.lat], zoom: previous === "2d" ? state.map.getZoom() - 1 : map.getZoom(), pitch: mode === "3d" ? 60 : 0, bearing: mode === "3d" ? -28 : 0, duration: duration() });
    }
    chrome(); renderNow();
  }

  function renderNow() {
    if (!$("mapStage")) return;
    if (mode === "heat" && !heatSupported()) setMode("2d");
    chrome();
    if (!ready || mode === "2d") return;
    const key = [mode, state.colorBy, state.currentT, state.quarterlyMapView, state.finalsActualRevealed, state.timeMachineOutcomeFilter, state.profile, state.whatIfEnabled, state.whatIfPct].join("|");
    if (signature !== key) {
      rows = state.grid.features.map(feature => {
        const props = feature.properties, h3 = props.h3_index, c = state.h3IndexOf.get(h3);
        const filtered = state.colorBy === "onset" && state.finalsActualRevealed && state.timeMachineOutcomeFilter !== "all" && onsetValue("time_machine_result", c) !== state.timeMachineOutcomeFilter;
        const value = encoding(c);
        return { type: "Feature", id: h3, geometry: feature.geometry, properties: { h3, color: colorForMode(state.colorBy, state.currentT, c), height: filtered ? 0 : value.height, weight: filtered ? 0 : value.weight } };
      });
      signature = key;
      map.getSource("network-cells").setData({ type: "FeatureCollection", features: rows });
      map.getSource("network-column-shells").setData({ type: "FeatureCollection", features: rows.map(columnShell) });
      map.getSource("network-points").setData({ type: "FeatureCollection", features: rows.filter(row => row.properties.weight > 0).map(row => { const props = state.gridPropsByH3.get(row.properties.h3); return { ...row, geometry: { type: "Point", coordinates: [props.centroid_lon, props.centroid_lat] } }; }) });
    }
    const selected = rows.filter(row => row.properties.h3 === state.selectedH3 || state.compareCells.includes(row.properties.h3));
    map.getSource("network-selection").setData({ type: "FeatureCollection", features: selected });
    map.getSource("network-column-selection").setData({ type: "FeatureCollection", features: selected.map(columnShell) });
    map.setLayoutProperty("network-ground", "visibility", mode === "3d" ? "visible" : "none");
    ["network-columns", "network-selected-top", "network-column-footprints"].forEach(id => map.setLayoutProperty(id, "visibility", mode === "3d" && columnsEnabled ? "visible" : "none"));
    map.setLayoutProperty("network-heat", "visibility", mode === "heat" ? "visible" : "none");
    syncEnvironment();
  }

  function columnShell(feature) {
    const props = state.gridPropsByH3.get(feature.properties.h3);
    const ringShell = rings => {
      const scale = amount => rings[0].map(p => [props.centroid_lon + (p[0] - props.centroid_lon) * amount, props.centroid_lat + (p[1] - props.centroid_lat) * amount]);
      // Opposite winding creates an open centre, including the roof. Keep
      // full cell geometry in the ground layer and exact selection targets.
      return [scale(.88), scale(.85).reverse()];
    };
    const geometry = feature.geometry.type === "Polygon"
      ? { type: "Polygon", coordinates: ringShell(feature.geometry.coordinates) }
      : feature.geometry.type === "MultiPolygon"
        ? { type: "MultiPolygon", coordinates: feature.geometry.coordinates.map(ringShell) }
        : feature.geometry;
    return { ...feature, geometry };
  }

  function updateColumnOpacity() {
    $("mapColumnOpacityValue").textContent = `${Math.round(columnOpacity * 100)}%`;
    if (!ready) return;
    ["network-columns", "network-selected-top"].forEach(id => map.setPaintProperty(id, "fill-extrusion-opacity", columnOpacity));
  }

  window.GeoQoEMap = {
    init() {
      document.querySelectorAll("[data-map-style]").forEach(button => button.addEventListener("click", () => { signature = ""; setMode(button.dataset.mapStyle); }));
      $("mapOverviewBtn").addEventListener("click", () => { const b = state.geoLayer.getBounds(); if (mode === "2d") state.map.fitBounds(b, { padding: [32, 32], animate: duration() > 0 }); else map?.fitBounds([[b.getWest(), b.getSouth()], [b.getEast(), b.getNorth()]], { padding: 45, pitch: mode === "3d" ? 48 : 0, bearing: mode === "3d" ? -18 : 0, duration: duration() }); });
      $("mapRotateBtn").addEventListener("click", () => map?.easeTo({ bearing: map.getBearing() + 45, duration: duration() }));
      $("mapTiltBtn").addEventListener("click", () => map?.easeTo({ pitch: map.getPitch() > 45 ? 25 : 60, duration: duration() }));
      $("mapCityBtn").addEventListener("click", () => map?.flyTo({ ...cityCamera, duration: duration(), essential: false }));
      $("mapBuildingsEnabled").addEventListener("change", e => { buildingsEnabled = e.target.checked; syncEnvironment(); });
      $("mapTerrainEnabled").addEventListener("change", e => { terrainEnabled = e.target.checked; syncEnvironment(); });
      $("mapColumnsEnabled").addEventListener("change", e => { columnsEnabled = e.target.checked; signature = ""; renderNow(); });
      $("mapColumnOpacity").addEventListener("input", e => { columnOpacity = Number(e.target.value) / 100; updateColumnOpacity(); });
      new ResizeObserver(() => { if (mode === "2d") state.map.invalidateSize(); else map?.resize(); }).observe($("mapStage"));
      chrome();
    },
    render() { cancelAnimationFrame(frame); frame = requestAnimationFrame(renderNow); },
    isSpatial() { return mode !== "2d"; },
    flyTo(h3) { if (mode === "2d" || !map) return; const props = state.gridPropsByH3.get(h3); if (props) map.flyTo({ center: [props.centroid_lon, props.centroid_lat], zoom: mode === "3d" ? 15.5 : 13.5, duration: duration(), essential: false }); },
    basemap() {
      if (!map) return;
      if (!ready) { map.once("load", () => window.GeoQoEMap.basemap()); return; }
      popup?.remove();
      // Preserve terrain, its camera state and data layers while replacing
      // only imagery. A full style reset can invalidate active terrain tiles.
      ["place-labels", "basemap"].forEach(id => { if (map.getLayer(id)) map.removeLayer(id); });
      ["placeLabels", "basemap"].forEach(id => { if (map.getSource(id)) map.removeSource(id); });
      const next = style();
      Object.entries(next.sources).forEach(([id, source]) => map.addSource(id, source));
      next.layers.filter(layer => layer.id !== "background").forEach(layer => map.addLayer(layer, "network-ground"));
    },
  };
})();
