# GeoQoE Pulse — Data & GIS Methodology

**What this document is.** The reasoning behind the data layer: where every dataset came from, why it was chosen over the alternatives, what we decided *not* to do and why, and which limits are permanent rather than unfinished work.

`HANDOFF.md` is the contract — what the files contain and how to use them. This is the *why*. Read this before presenting, and before a judge asks "why did you use that?"

**Pipeline version:** 3.3.0 · **Study area:** Klang Valley, 4,918 H3 res-8 cells (~0.86 km² each)

---

## Contents

1. [The question the data has to answer](#1-the-question-the-data-has-to-answer)
2. [Every dataset — what, why, and why not something else](#2-every-dataset)
3. [How the datasets become one grid](#3-how-the-datasets-become-one-grid)
4. [Decisions we made along the way](#4-decisions-we-made-along-the-way)
5. [The France option — found, evaluated, dropped](#5-the-france-option)
6. [Why we dropped hourly](#6-why-we-dropped-hourly)
7. [What the MCMC sources give us](#7-what-the-mcmc-sources-give-us)
8. [Limits that cannot be solved, and why](#8-limits-that-cannot-be-solved)
9. [How we know the data is right](#9-how-we-know-the-data-is-right)

---

## 1. The question the data has to answer

The proposal asks: **does mobile network quality degrade where demand is growing fastest?**

That sentence dictates the entire data design. To answer it you need four things, and each one is a hard requirement:

| Requirement | Why it is non-negotiable |
|---|---|
| A **quality** measure | Otherwise there is no "degrade" |
| A **demand** measure | Otherwise there is no "where demand grows" |
| Both on the **same geography** | Otherwise you cannot compare them |
| Both over **time** | "Growing" and "degrading" are both changes, not states |

The fourth is the one that most projects get wrong. A single snapshot can only show *correlation between places* — "busy areas are slower" — which is nearly tautological and tells an operator nothing. Showing that quality *falls where demand rose* requires measuring both at two or more points in time.

Every dataset below exists to satisfy one of those four requirements, and the biggest single decision in this project — dropping hourly for quarterly — was made to satisfy the fourth honestly.

---

## 2. Every dataset

### 2.1 Ookla Open Data — the quality measure ⭐

| | |
|---|---|
| **What** | Crowdsourced mobile speed tests, aggregated to ~600 m tiles, published quarterly |
| **How we got it** | Public S3, no account: `ookla-open-data.s3.amazonaws.com/parquet/performance/type=mobile/year=YYYY/quarter=Q/…` |
| **How much** | **14 quarters (2023Q1–2026Q2), 3,959,197 real speed tests**, 3,566 cells |
| **Gives us** | `ookla_dl_mbps`, `ookla_ul_mbps`, `ookla_rtt_ms`, `ookla_tests`, `ookla_devices` |

**Why this and not something else.** It is the only free source that gives real, geolocated, repeated mobile quality measurements for Malaysia. The alternatives all fail on at least one requirement:

- **MCMC's own published statistics** — national or state level, no coordinates. Cannot join a hexagon grid.
- **Operator coverage maps** — modelled predictions, not measurements, and no history.
- **Published drive-test datasets** — this is the Sunway data (§2.8). Ten cells. Real and rigorous, but not a study area.

**The property that made the whole project possible.** Ookla's `quadkey` tile identifier is **stable across quarters** — the same tile ID refers to the same ground area every release. That is what turns 14 separate downloads into a genuine time series rather than 14 unrelated snapshots. Without it there would be no temporal axis and no answerable question.

**What we do about its bias.** Ookla is *consumer-initiated* — people run a speed test when the network feels slow, or when they have just upgraded. Coverage is therefore not random, and we measured exactly how non-random: GIS features predict which cells Ookla covers at **ROC-AUC 0.978**. That is handled in §3.4, not ignored.

**Raw files deleted after extraction** (~2.75 GB for a few hundred KB of useful extract). Every source URL is recorded in `data/outputs/source_manifest.json`, so any quarter is re-fetchable exactly.

---

### 2.2 NASA VIIRS Black Marble (VNP46A3) — demand that changes over time ⭐

| | |
|---|---|
| **What** | Monthly satellite nighttime-lights composites, ~500 m resolution |
| **How we got it** | NASA LAADS DAAC, free with an Earthdata login |
| **How much** | **43 monthly granules, 2022-12 → 2026-06** |
| **Gives us** | `viirs_rad_quarter`, `viirs_rad_yoy_log_ratio`, `viirs_monthly_std` |

**Why we need it.** This is the *only* demand signal in the project that varies **per cell and per quarter**. Everything else is either fixed per cell (population, POIs, buildings) or shared across a whole district. Without VIIRS, "where demand grew" could only be answered at 8-district resolution — too coarse for a hexagon-level claim.

**Why nighttime lights as a demand proxy.** Radiance responds to economic activity, construction and human presence. It is not mobile traffic, and we never say it is. The honest phrasing is *"associated with rising activity"*.

Two processing decisions worth defending:

- **Year-on-year, not quarter-on-quarter.** Klang Valley is equatorial with a monsoon; consecutive quarters differ by season as much as by activity. The same quarter one year earlier is seasonally comparable. The 4-quarter lag is validated against the real quarter gap, so a cell with a missing quarter never silently gets a 3- or 5-quarter comparison labelled "year on year".
- **Log-ratio, not percent change.** A dim cell going 0.1 → 1.3 reads as *+1,200%* — arithmetically true, analytically useless. The log ratio is symmetric (a halving and a doubling are equal and opposite) and tail-tamed. **Model on `viirs_rad_yoy_log_ratio`.**

Cloud cover is substantial at this latitude, so `viirs_n_months_in_quarter` records how many valid retrievals back each value, and growth is suppressed below two.

---

### 2.3 WorldPop — population

| | |
|---|---|
| **What** | 100 m gridded population, UN-adjusted, 2020 reference year |
| **How we got it** | worldpop.org, open licence |
| **Gives us** | `population_night`, `population_day`, `population_density_per_km2` |

**Why this and not the census directly.** Malaysia's census publishes counts by administrative boundary — mukim or district. Those are large, irregular, and do not align with hexagons. Splitting a district count across hexagons by area would assume uniform density, which is exactly wrong in a city. WorldPop has already done that disaggregation properly, using built-up structure and ancillary layers.

**Its limitation, stated plainly.** 2020 reference year — the oldest layer in the project, and static. It cannot tell you where population *grew*. That gap is what DOSM (§2.4) and VIIRS (§2.2) fill.

---

### 2.4 DOSM district population — real population change

| | |
|---|---|
| **What** | Department of Statistics Malaysia district population, annual 2020–2025 |
| **How we got it** | data.gov.my, open data |
| **Gives us** | `district_population`, `district_population_yoy_growth_pct` |

**Why we need it despite WorldPop.** WorldPop is a fixed 2020 snapshot. DOSM is a real, official, *annual* series — the only genuinely governmental measure of demand growth in the project. When we claim demand rose, this is the citable source.

**Its limitation.** Eight districts across 4,918 cells. Every cell in Petaling shares one value. It is real and temporal but spatially coarse — which is precisely why VIIRS matters, and why the two are kept as separate features rather than blended into one index.

---

### 2.5 OpenStreetMap — the built environment

| | |
|---|---|
| **What** | Buildings, roads, POIs, transit stops |
| **How we got it** | Geofabrik Malaysia extract, 2026-07-27 |
| **Gives us** | ~15 features: `built_up_ratio`, `road_density_km_per_km2`, `intersection_count`, `poi_*_count`, `dist_to_nearest_transit_m` |

**Why OSM.** Free, current, and the only source with POI *categories* — offices, retail, education, healthcare. Category matters because it distinguishes an area busy at 09:00 from one busy at 21:00. Malaysian OSM coverage in Klang Valley is dense.

**The known weakness:** volunteer-mapped, so completeness varies. Where it matters most — building heights — we quantify it rather than hide it: only **6.5% of buildings** carry a height tag, covering 21% of cells, clustered where mappers are active. `building_height_coverage_is_biased = True` ships with the data. **We never impute heights**, because imputing from the mapped subset would import exactly the bias we are flagging.

---

### 2.6 ESA WorldCover — land cover

| | |
|---|---|
| **What** | 10 m global land cover, 11 classes, 2021 |
| **How we got it** | ESA/Copernicus, free |
| **Gives us** | `landcover_built_up_pct`, `landcover_tree_cover_pct`, and 9 more, summing to exactly 100% per cell |

**Why we need it alongside OSM buildings.** They measure different things and disagree usefully. OSM's `built_up_ratio` is *mapped building footprints* — it inherits volunteer coverage gaps. WorldCover is *satellite-observed impervious surface* — complete everywhere, no human in the loop.

Their agreement (**ρ = +0.96**) is one of the strongest pieces of evidence that our spatial joins are correct: an independent satellite product and a volunteer map, processed through entirely separate code paths, landing in the same hexagons.

---

### 2.7 Copernicus DEM GLO-30 — terrain

| | |
|---|---|
| **What** | 30 m elevation, global |
| **How we got it** | Copernicus Data Space, free |
| **Gives us** | `elevation_mean_m`, `terrain_ruggedness_m`, `elevation_range_m`, 100% coverage |

**Why terrain matters for mobile.** Radio propagation is line-of-sight dependent. Hills block signal, valleys trap it, ruggedness predicts coverage holes. This is genuine physical explanation for why some cells underperform their demand profile.

**Two honest caveats.** It is a *surface* model — it includes buildings and tree canopy, so `elevation_mean_m` is not bare-ground height and correlates with the building layers by construction. And it produced the most instructive bug in the project: the tiles declare `nodata=None`, so reading outside a tile's footprint returned **0 and counted it as data** — 2,562 cells silently got an elevation of exactly 0.0 m. Caught because a median elevation of 0 m for Klang Valley is obviously impossible. Now fixed by selecting tiles geographically, and locked by a regression test.

---

### 2.8 Sunway multi-operator QoE dataset — the only true ground truth

| | |
|---|---|
| **What** | Published academic drive/walk-test dataset, 2 km² around Sunway University |
| **How we got it** | **Mendeley Data — third-party, not self-collected.** Kabeer, Nordin & Behjati (2025), *An Urban Multi-Operator QoE-Aware Dataset for Cellular Networks in Dense Environments*, v1, published 2025-06-17 · [DOI 10.17632/dx5xyyfz2y.1](https://doi.org/10.17632/dx5xyyfz2y.1) |
| **Collection method** | GNetTrack Pro on Samsung S21 Ultra; 3 anonymised commercial operators; 4G and 5G; walking and driving; 132 physical cell sites validated via OpenCellID and field inspection |
| **How much** | **30,925 cleaned, labelled records** falling in 10 of our cells; measurements span 2024-12-03 → 2025-04-02 |
| **Gives us** | 26 columns: RSRP, RSRQ, SNR, CQI, jitter, packet loss, per-operator signal |

**Attribution matters here.** This is a published dataset by other researchers, released under Mendeley Data. It must be **cited, never presented as our own fieldwork.** The shipped data records this — `sunway_source` reads `"REAL (Mendeley dx5xyyfz2y)"` on every row.

That it is peer-published is a strength, not a caveat: the methodology is documented, the equipment is named, the operators were anonymised by the authors, and the cell sites were independently validated. It is more defensible than an undocumented internal campaign would be.

**What only this can do.** Every other quality source in the project is crowdsourced or modelled. This is measured with known equipment on a known route at a known time. Two things nothing else here provides:

1. **The hole Ookla structurally cannot see.** One cell has **91.4% of readings below −100 dBm** — a genuine coverage failure. Ookla reports 147 Mbps for that same cell. Both are true: Ookla only sees people who *succeeded* in running a speed test, so it is blind to places where the signal is too weak to try. This is the clearest available illustration of crowdsourcing bias, and it comes from independent measurement rather than from argument.

2. **A like-for-like operator comparison.** One cell shows Operator A at −79 dBm and Operator B at −98 dBm — same place, same route, same handset, same window. A 19 dB gap is roughly an 80× difference in received power. Public Malaysian data essentially never permits this comparison.

**What it cannot do — the easiest mistake to make.** Sunway's throughput reads ~0.5 Mbps where Ookla reads ~150 Mbps in the same cells. Not a bug and not a unit error: resolved empirically against the file's own test phases, and **confirmed by the source description**, which states the traffic scenarios are HTTP, FTP and 1080p video streaming. Those are application transfers, not saturating speed tests — a different quantity from what Ookla measures. **Sunway cannot validate or calibrate Ookla.** Compare Sunway cells only with each other.

---

### 2.9 OpenCellID — tower locations

| | |
|---|---|
| **What** | Crowdsourced cell tower positions, MCC 502 (Malaysia) |
| **Gives us** | `tower_count`, `tower_density_per_km2`, `dist_to_nearest_tower_m` |

**Why it is here and what it is not.** It is the only free source of tower geography, and tower density is a reasonable proxy for infrastructure investment. But positions are *crowd-estimated from handset observations*, not surveyed — accurate to maybe a few hundred metres. And critically: **it carries no capacity information.** No spectrum, no bandwidth, no sector count. It tells you a tower is probably near, never how much traffic it can carry. See §8.1.

---

### 2.10 M-Lab NDT — the hourly shape

| | |
|---|---|
| **What** | Open internet measurement, 2.05M tests, aggregated to one citywide diurnal curve |
| **Gives us** | The hour-of-week shape used by the hourly demo panel |

**Why it is in the project at all.** The hourly panel needs a temporal shape. A hand-invented sine wave would be fiction. M-Lab gives a *real, measured* shape from real tests — one citywide curve, applied uniformly.

**The bug this source taught us.** M-Lab's BigQuery export is keyed in **UTC**. We applied it to local Malaysian timestamps without converting — an 8-hour phase error. The "worst" hour landed at 14:00, and the resulting `is_peak_period` feature pointed the network's busiest hours at its *fastest* period. Now converted at load, worst hour at **22:00**, and a validator hard-fails the build if the worst hour is ever outside the evening again.

That fix is why HANDOFF.md opens by telling your teammate their previous model learned an inverted daily pattern.

---

## 3. How the datasets become one grid

### 3.1 Why H3 hexagons

Every source arrives on a different geometry: Ookla on ~600 m square tiles, VIIRS on a ~500 m raster, WorldPop on 100 m, OSM as vectors, DOSM as districts. They must land on one common unit.

**H3 resolution 8** (~0.86 km²) was chosen because:

- **Hexagons have uniform adjacency.** Every neighbour is the same distance away, unlike squares where diagonals are 1.41× farther. This matters because we compute spatial statistics (Getis-Ord Gi\*) where a distorted neighbourhood distorts the result.
- **Res-8 matches Ookla's native tile size.** Finer would invent detail Ookla does not have; coarser would throw away detail it does.
- **Nesting gives free holdout keys.** `h3_parent_res6` groups cells into ~7× larger areas for spatial cross-validation, and res-5 for district roll-up. No extra geometry needed.

### 3.2 CRS discipline

**All storage in EPSG:4326 (WGS84). Every distance, area and length computed in EPSG:3375** (Peninsular Malaysia RSO).

This is not pedantry. Computing a centroid or a distance in degrees near the equator gives an error of several percent that varies with latitude — a systematic distortion that looks like a real spatial pattern. I made this exact error once during development (building centroids in WGS84) and caught it only from a GeoPandas warning.

### 3.3 Join methods, per source type

| Source type | Method | Why |
|---|---|---|
| Raster (VIIRS, WorldCover, DEM) | `rasterstats.zonal_stats` | Area-weighted, respects partial pixel overlap |
| Polygons (buildings) | `overlay` + area in EPSG:3375 | Splits footprints crossing hexagon edges |
| Points (POIs, towers) | `sjoin` | Straight containment |
| Lines (roads) | `overlay` + length in EPSG:3375 | Splits at boundaries; no double-counting |
| Nearest (transit) | `sjoin_nearest` in EPSG:3375 | Real metres, not degrees |
| Tiles (Ookla) | Test-count-weighted mean | A tile with 900 tests must outweigh one with 3 |

### 3.4 Handling Ookla's coverage bias

Ookla covers 2,950 of 4,918 cells, and **not at random** — GIS features predict coverage at ROC-AUC 0.978. Naively training on covered cells and predicting the rest is extrapolation, not interpolation.

We built three columns to make that explicit:

- `coverage_propensity` — out-of-fold probability that a cell is covered. Use as an inverse-propensity training weight.
- `in_common_support` — whether an uncovered cell resembles the covered ones at all.
- `coverage_tier` — the plain-language version:

```
measured    2,950   real measurement exists
estimable     280   similar enough to covered cells that a prediction is defensible
unknown     1,688   unlike anything measured — say "insufficient data"
```

**The honest reach of a spatial model is ~280 extra cells, not 1,968.** Presenting a prediction for the `unknown` tier at the same confidence as a measurement would be the single most misleading thing this project could do. The dashboard should grey them out.

---

## 4. Decisions we made along the way

### 4.1 One snapshot → 14 quarters

**Original state:** a single 2024Q4 Ookla file.
**Problem:** with one time point you cannot show change. The proposal's question becomes unanswerable, and the project collapses to "busy places are slower."
**Decision:** ingest all 14 available quarters, exploiting stable quadkeys.
**Cost:** ~2.75 GB of downloads, deleted after extraction.
**Result:** the project's central claim became testable — and it held. The fastest activity-growth quartile is **1.55× more likely** to fall below the city norm next quarter (14.98% vs 9.69%, p = 3e-16), with mean reversion ruled out as a confounder.

### 4.2 Absolute → relative degradation target

**Problem:** `is_degraded` uses absolute thresholds (<10 Mbps or >100 ms). Malaysian speeds rose from 141 to 192 Mbps over the study window, so the positive rate fell to **3.0% and shrinking** — a target that is disappearing as the network improves.
**Decision:** the primary target is `is_below_area_norm` — the bottom 25% of the citywide distribution *in that quarter*.
**Why it is better:** stable ~22–25% prevalence across all 14 quarters, and it asks the operationally useful question — "which areas are falling behind their city?" — rather than one that a nationwide upgrade silently answers for you.

### 4.3 Fixing the config that recommended the wrong target

Found during final verification: `config.PRIMARY_ML_TARGET` still named the *hourly* target, and the feature dictionary labelled it "RECOMMENDED PRIMARY ML TARGET" — while the same config file documented that target as reconstructable by memorisation. HANDOFF.md was right; config and the dictionary were not.

A teammate who opened either — both reasonable first stops — would have trained on a label a lookup table solves, scored near-perfectly, and had it collapse under the first question. Now corrected in all four places and locked by a test.

### 4.4 Six-month → 43-month VIIRS

The extraction held 43 months while the merged layer still carried 6-month averages — same column names, same row counts, plausible values, silently stale. Rebuilt, and a **freshness guard** now fails the build if any derived layer is older than its own inputs. The guard matters more than the rebuild.

### 4.5 Application profiles instead of one "degraded" flag

A single flag hides that a cell can be fine for one thing and unusable for another. A cell at 8 Mbps / 30 ms streams video badly but carries voice perfectly; 200 Mbps / 180 ms is the reverse.

Five profiles now ship (voice, browsing, video call, streaming, gaming), each pass/fail, plus `app_worst_failing` — "what is actually broken here" in user language. **Thresholds come from engineering guidance (ITU-T G.114, codec requirements); weights come from MCMC BQoES.** Deliberately not conflated: a satisfaction survey does not set technical requirements.

The finding: **the binding constraint is latency and upload, not download.** Web browsing and video calls fail most often. That is a more interesting story than "some places are slow."

---

## 5. The France option

### What it was

While looking for a way to close the capacity gap (§8.1), we found that **France publishes what Malaysia does not**: ANFR's site database gives per-site, per-operator technical detail, and ARCEP publishes independent measured mobile quality. Together they would have supplied the one input this project fundamentally lacks — a real, geolocated measure of network *capacity* alongside real quality measurements, per operator.

On data quality alone, it was clearly the more comprehensive option.

### Why we dropped it

**Because the data was never the point.** This is a Malaysian bootcamp project, judged on Malaysian relevance. Switching study area would have meant discarding:

- The **Sunway ground truth** — 30,925 independently measured records (Mendeley `dx5xyyfz2y`), the only true ground truth in the project, and specific to this study area
- The **MCMC framing** — BQoES weights, the regulator's own register, the entire "what Malaysian users care about" argument
- **DOSM** population, and every claim tied to Malaysian districts
- The relevance of the work to the audience it is for

We would have traded a project *about Klang Valley with an honest gap* for a technically richer project about a country nobody in the room works in.

**The decision was scope, not capability.** It is worth stating exactly that way if asked, because it is a stronger answer than pretending we never found a better dataset. "We found richer open data in France, evaluated it, and chose relevance over completeness — then documented the resulting gap and tested its impact" is a defensible engineering judgement. It also demonstrates we looked.

---

## 6. Why we dropped hourly

This was the most consequential decision in the project.

### The original design

`cell_time_dataset.parquet` — 826,224 rows, 4,918 cells × 168 hours. It looks like the centrepiece: a real temporal panel at hourly resolution.

### What we found when we tested it

**No hourly mobile QoE data exists for Malaysia. At any price we could reach.** Not from MCMC, not from operators, not from Ookla, not from any open source. So the panel was built as `level(cell) × shape(hour)` — each cell's real Ookla average multiplied by one shared citywide M-Lab diurnal curve.

Testing that structure produced an unambiguous result:

```
dl/rtt reconstruct from (cell, hour-of-week) with max absolute error 0.0
the stored target matches the reconstruction on 100% of anchored rows
```

The panel is **rank-one**. Three consequences follow, and none are fixable by better modelling:

1. **The forecast target is a lookup table.** Given cell identity and hour, it is fully determined. Any accuracy figure measures memorisation.
2. **No cell-specific temporal dynamics exist.** No model can learn that one area degrades earlier than another, because that information was never in the data.
3. **Hotspot recurrence is noise.** Gi\* is invariant to a global scale factor, so categories move only via the Gaussian noise on unanchored cells.

A separability test (`separability_test.json`) confirms the shared-shape assumption **fails** when tested against the real Sunway data.

### The decision

**Keep the panel as a labelled demonstration fixture; move every real claim to the quarterly dataset.**

This costs the demo its hourly animation as a *result* — it remains fine as an illustration. What it buys is that every number the project reports is defensible. A judge who asks "how do you have hourly data for Malaysia?" gets a straight answer instead of a collapse.

### Was it worth pursuing hourly further?

We evaluated this on value and impact, not sunk effort. **No** — and the reason is that the hourly axis does not serve the question. The proposal asks whether quality degrades where demand *grows*. Demand growth is a phenomenon of months and years — population shifts, new developments, changing land use. Hourly resolution answers a different question ("when is the network busy?"), one that operators already know the answer to.

The quarterly axis is not a downgrade from hourly. It is the correct resolution for the question being asked.

---

## 7. What the MCMC sources give us

### 7.1 BQoES 2024 — the survey

**What it is:** MCMC's Broadband Quality of Experience Survey, 2,304 respondents, national.

**What it cannot do, stated first:** it has no coordinates, no timestamps, no measurements, and its finest geography is urban vs rural. **It cannot join the H3 grid and it is not a pipeline input.** Anyone who tries to merge it is making a category error.

**What it is genuinely worth — two things, both in slides:**

**Justifying the index weights with regulator evidence.**

| Activity | Importance (1–5) | Satisfaction gap |
|---|---|---|
| Online voice calls | **4.31** | **−0.34** |
| Web browsing | 4.24 | −0.31 |
| Online video calls | 4.09 | −0.19 |
| Video streaming | 3.88 | +0.04 |
| Online gaming | 3.47 | +0.33 |

The two activities Malaysians value most are both **latency-driven**, and both are where operators fall furthest short. Gaming, by contrast, already exceeds expectations. This replaces "we chose these weights" with "the regulator's national survey chose them."

**Translating metrics into user language.** Video stalls 52.4% → throughput. Choppy voice 49.0% → jitter and loss. Waiting for pages 24.0% → latency. This is what `app_worst_failing` is built on, and it reads far better on a dashboard than a millisecond figure.

**State the caveat yourself before anyone asks:** 29.9% of Malaysia's mobile users are rural; this study area is entirely urban.

### 7.2 The Register of Apparatus Assignments — examined, and structurally unable to help

**What it is:** MCMC's public register of site-based spectrum licences — holder, site code, TX/RX frequency, sector, expiry.

**Why we hoped it would help:** it is the closest thing Malaysia publishes to per-site network capacity. If sector counts per cell tracked speed, we would have a real capacity feature — and a mechanism for *why* a cell degrades, not just *where*.

**We retrieved 30 sector rows across 10 sites (Hulu Langat and Sepang, LTE and 5G filters, 2026-08-30) and decoded every frequency pair against the 3GPP band plan.** Two independent blockers emerged, either of which alone is fatal.

#### (a) There is no geography

The `Location` field is an operator-internal site code — `LGAT(BT14 HULU LANGAT)`, `DBSEP1368_PEKANSEPANG` — with **no latitude or longitude anywhere in the record**. Nothing can be joined to a hexagon. The finest achievable match is district name: 8 districts across 4,918 cells, which is far too coarse to say anything about a 0.86 km² cell.

Worse, name matching silently crosses states. Searching the study area's `SEPANG` returns `DSBKI0302_TLKSEPANGGAR` — **Telok Sepanggar, Kota Kinabalu, roughly 1,600 km away in Sabah.** In one nine-row sample, a third of results came from the wrong end of the country. (Two-word searches like `HULU LANGAT` are safe; short names are not.)

#### (b) It contains none of the spectrum that carries throughput

| Holder | TX / RX (MHz) | Band | Role | Sites | Sectors |
|---|---|---|---|---|---|
| U Mobile | 788 / 733 | **Band 28 (700 MHz)** | coverage | 6 | 18 |
| YTL Communications | 800.5 / 841.5 | **Band 20 (800 MHz)** | coverage | 2 | 6 |
| Digital Nasional Berhad | 768 / 713 | **Band 28 (700 MHz)** | coverage | 2 | 6 |

```
Sub-1 GHz coverage spectrum : 100% of sectors
Capacity spectrum (1800+)   : 0 sectors
Maxis, CelcomDigi           : absent entirely
```

Sub-1 GHz is **coverage** spectrum — low frequencies that travel far and penetrate buildings. It determines *reach*, not *speed*. Throughput comes from 1800/2100/2600 MHz and 5G mid-band, and the two largest operators hold those under **national Spectrum Assignments with no site-level geography at all**.

*(YTL's TX being below its RX is not an error: Band 20 is reverse-duplex, downlink 791–821 below uplink 832–862. It is a genuine Band 20 signature.)*

#### Why (b) is the argument to lead with

Reason (a) could in principle be fixed — MCMC could publish coordinates. **Reason (b) could not.** Even given exact site positions, the register could not explain why one cell is faster than another, because the spectrum that decides throughput is simply not in it.

**Quote this when asked why there is no capacity data:**

> *"We went to the regulator's own register. It publishes no site coordinates — only internal site codes — so nothing joins to a spatial grid. And every sector we retrieved was 700 or 800 MHz coverage spectrum from U Mobile, YTL and DNB; Maxis and CelcomDigi don't appear. The bands that actually determine throughput are held under national assignments with no site-level geography, so even with coordinates the register couldn't explain a speed difference."*

**State it as a property of the data, never as "capacity doesn't matter."** Capacity absolutely drives speed — the register just doesn't contain it. Anyone who knows telecoms will catch the stronger claim immediately.

Evidence: raw rows in `data/raw/mcmc_register/`, decoding in `scripts/analysis/check_mcmc_register_bands.py`, finding in `data/outputs/mcmc_register_findings.json`. Reproducible from a single register search.

> **Note on an earlier version.** This supersedes a claim that 40 matched cells showed *"no relationship between sector count and speed (Spearman −0.065, p = 0.69)."* That was not wrong, but it rested on district-name matching and its per-cell counts were never persisted. A correlation invites *"your sample was too small"*; a structural finding does not. The shortlist file behind it has been removed rather than left in the repo as a number whose working cannot be reproduced.

---

## 8. Limits that cannot be solved

These are properties of what Malaysia publishes, not unfinished work. Each is bounded, documented, and — where possible — measured.

### 8.1 No capacity data

**Why it cannot be solved:** capacity spectrum is licensed nationally with no site-level geography. It is not published, at any price, by anyone. Operators treat it as commercially sensitive. §7.2 documents the test.

**What we do instead:** tower density and terrain proxy for infrastructure. We make **no capacity claim**.

### 8.2 No hourly QoE

**Why it cannot be solved:** no hourly mobile QoE source exists for Malaysia. Ookla publishes quarterly aggregates by design — the hourly detail is not withheld, it is never published. §6.

**What we do instead:** the quarterly axis, which is the right resolution for the question anyway.

### 8.3 Ookla is consumer-initiated

**Why it cannot be solved:** it is inherent to crowdsourcing. Fixing it would require a measurement campaign across 4,918 cells — which is precisely what Sunway is, at 10 cells.

**What we do instead:** measure the bias (ROC-AUC 0.978), ship `coverage_propensity`, `in_common_support` and `coverage_tier`, and weight by `ookla_tests`. **The bias is quantified rather than assumed away.**

### 8.4 Demand is a proxy, never traffic

**Why it cannot be solved:** real mobile traffic per cell is operator-internal data. It is never published anywhere.

**What we do instead:** VIIRS radiance and DOSM population, both labelled as *activity* proxies. The phrasing is always "associated with rising activity", never "caused by rising demand."

### 8.5 No jitter or packet loss outside Sunway

**Why it cannot be solved:** Ookla's open tiles publish download, upload and latency only. Jitter and loss are simply not in the product.

**What we do instead:** voice and video-call profiles are marked **partial and optimistic** — a cell can pass every measurable criterion and still sound broken. `app_n_evaluable` records how many criteria were actually checkable. The hourly panel's jitter/loss columns are simulated and must never be evaluated as if real.

### 8.6 Building heights are sparse and biased

**Why it cannot be solved:** OSM height tags depend on volunteers. 6.5% of buildings, 21% of cells, clustered where mappers are active.

**What we do instead:** `building_height_coverage_is_biased = True`, and **we never impute** — imputing from the mapped subset would import the very bias being flagged.

### 8.7 Mixed vintages

**Why it cannot be solved:** each publisher sets its own release cycle. Population 2020–2025, land cover 2021, OSM 2026, Ookla 2023–2026, VIIRS 2022–2026.

**What we do instead:** per-source `vintage` in `metadata.json`. The risk is real — a 2021 land cover layer will miss a development built in 2024 — and is stated rather than smoothed over.

---

## 9. How we know the data is right

The pipeline's own validators check internal consistency: columns exist, row counts match, nothing unexpectedly null. That catches a *broken* build. It cannot catch a build that is **wrong but self-consistent** — a CRS error shifting every cell equally, a join attaching correctly-shaped data to the wrong hexagons.

So `tests/test_output_integrity.py` (**21 tests, ~7 s**) asks a different question: *does this agree with the outside world?*

**Sources that never coordinated, agreeing:**

| | Spearman ρ |
|---|---|
| WorldPop vs NASA VIIRS | **+0.93** |
| ESA WorldCover vs OSM buildings | **+0.96** |
| WorldPop vs OSM roads | +0.85 |
| Tree cover vs built-up | −0.69 |

NASA, ESA, WorldPop and OSM cannot make the same spatial-join error. This is the strongest evidence available without ground truth.

**Landmarks match reality:**

| Location | pop/km² | radiance | built-up | elev |
|---|---|---|---|---|
| KLCC | 9,453 | 237.8 | 81% | 44 m |
| Sunway | 7,350 | 95.1 | 91% | 22 m |
| KLIA | 523 | 39.7 | 94% | 18 m |
| Hulu Langat | 130 | 2.7 | 2% | 182 m |

KLIA is the sharpest test — **94% built-up, 523 people/km²**, a vast structure where nobody sleeps. A grid offset by even one hexagon toward Sepang town would break that combination.

**No leakage:** highest feature-to-target correlation is **0.24**. Nothing is a disguised copy of the label.

**The data genuinely predicts.** Forward-in-time split, model never sees the future:

```
prevalence            0.220
model PR-AUC          0.720
persistence baseline  0.494   ← the number that matters
ROC-AUC               0.895
```

Persistence — "assume next quarter looks like this one" — is the honest benchmark. Beating chance is trivial; beating persistence is what shows the features carry real information.

**Build-time guards that fail the run:** dictionary completeness · diurnal timezone alignment · derived-layer passthrough · derived-layer freshness.

### What this does and does not prove

It proves the specific failures that would matter — misplaced joins, wrong CRS, leaked targets, stale layers, a memorisable benchmark — are each covered by a test that fails loudly.

It does not prove the code is free of every logic error. These test *outputs*, not every function. A mistake producing plausible numbers in an untested corner could still exist. Every bug we found was found by someone looking, not by a check firing — which is exactly why the checks now exist.

---

## Appendix — source summary

| Dataset | Source | Vintage | Role |
|---|---|---|---|
| Ookla Open Data | AWS S3, open | 2023Q1–2026Q2 (14 q) | **Quality target** ⭐ |
| VIIRS VNP46A3 | NASA LAADS DAAC | 2022-12–2026-06 (43 mo) | **Demand over time** ⭐ |
| WorldPop | worldpop.org | 2020 | Population |
| DOSM district | data.gov.my | 2020–2025 | Real population growth |
| OpenStreetMap | Geofabrik | 2026-07-27 | Built environment, POIs |
| ESA WorldCover | Copernicus | 2021 | Land cover |
| Copernicus DEM | Copernicus | GLO-30 | Terrain |
| Sunway QoE dataset | Mendeley `dx5xyyfz2y` (Kabeer et al. 2025) | 2024-12–2025-04 | **Ground truth**, 10 cells |
| OpenCellID | opencellid.org | 2026-07 | Tower geography |
| M-Lab NDT | M-Lab | Dec 2024+ | Hourly shape (demo only) |
| MCMC BQoES | MCMC | 2024 | Weights + language (**not a feature**) |
| MCMC Register | MCMC | 2026 | Tested, null result (**not a feature**) |

Full provenance with re-fetch URLs: `data/outputs/source_manifest.json` · `data/outputs/metadata.json`
