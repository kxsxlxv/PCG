# Research 01 — Geospatial data and building data fusion

Date: 2026-11-24 | Project: GeoWorld PCG / UNIGINE 2.22

Status: source and methodology audit; real Adler feature counts and height accuracy NOT measured. Sandworm is excluded from all architecture. This is an independent, offline-first compiler. All thresholds below are proposed research defaults, not locally calibrated values.

## 1. Decisions

- Primary consolidated footprint layer: Overture Buildings, supplemented with raw OSM tags, relations and history where relevant.
- Independent candidate layers: Microsoft ML footprints, GlobalBuildingAtlas, 3D-GloBFP. Do not append these layers directly to Overture.
- Russian residential enrichment: official GIS ЖКХ registry and Tochno dataset, preferably linked by reliable building/address IDs and coordinates.
- Store every observation with source lineage and uncertainty; choose canonical attributes separately.
- Use a regional bare-earth terrain estimate (FABDEM) and independently record DSM and building-height estimates. Building height is NOT absolute elevation.
- Use imagery only when acquisition rights allow analysis/derivatives. High-resolution imagery is optional for MVP; regional architecture inference is required regardless.
- No live provider calls in simulator runtime. Use pinned snapshots, local data, cached preprocessing, deterministic output.

## 2. Source catalog

| Source | Use | Specific evidence / caveats | URL |
|---|---|---|---|
| OSM PBF | Raw tags, relations, buildings/parts, roads, land use, POI | Geofabrik Southern Federal District ~303 MB at review. Preserve original PBF. | https://download.geofabrik.de/russia/south-fed-district/index.html |
| Overture 2026-11-18.0 | Consolidated building/part geometries | 2,544,125,628 buildings and 4,669,725 parts globally; OSM cutoff 2026-11-06; not local coverage. Buildings ODbL. | https://docs.overturemaps.org/blog/2026/11/18/release-notes/ |
| Overture schema | Height, floors, roof, facade fields and source lineage | Height/min_height and roof_direction conventions require normalization; field presence not completeness. | https://docs.overturemaps.org/schema/reference/buildings/building/ |
| Overture bridge files | OSM/source ID to GERS mapping | One-to-one matches only; source version suffix; provider partitioning since Aug 2026. | https://docs.overturemaps.org/gers/bridge-files/ |
| Microsoft GlobalMLBuildingFootprints | Coverage QA, additional height candidates | Nov 12 2026 update: missing height/confidence null instead of -1; capture epochs vary. CDLA Permissive 2.0 for current release; record exact artifact license. | https://github.com/microsoft/GlobalMLBuildingFootprints |
| GBA | Model-predicted 3m height grid, LoD1, polygons | 2.75B polygons, 97%+ height completeness, Europe LoD1 RMSE 4.1m; not local validation. Height/LoD1 CC BY-NC 4.0; separate ODbL polygons. | https://github.com/zhu-xlab/GlobalBuildingAtlas ; https://doi.org/10.5194/essd-17-6647-2025 |
| 3D-GloBFP | Secondary building heights | XGBoost estimate, 2020 nominal epoch, reported Russian RMSE 4.99m, original Russia archive 2.5 GB. | https://zenodo.org/records/15459025 ; https://zenodo.org/records/12674244 ; https://doi.org/10.5194/essd-16-5357-2024 |
| GHS-OBAT | Broad age class, use, height, compactness | GERS links to 2024-07-22.0; coarse GHSL-derived values, not cadastral year. | https://doi.org/10.2905/JRC.9ZB4R6G ; https://pmc.ncbi.nlm.nih.gov/articles/PMC12221504/ |
| OpenBuildingMap | Context/secondary estimates | GHSL-derived height, OSM POI/landuse function; not independent from OBAT. | https://www.openbuildingmap.org/ |
| GHSL / WSF 3D | Neighborhood-scale priors | GHSL ~100m and WSF 90m; not individual-building measurement. | https://human-settlement.emergency.copernicus.eu/ghs_buH2023.php ; https://web.geoservice.dlr.de/web/datasets/wsf_3d |
| GIS ЖКХ | Year, building type, construction details, possible floors/entrances | Official public register download; verify actual schema, completeness and matching. | https://cdn.dom.gosuslugi.ru/webhelp/topics/houses/och/t_download_archive-och.html |
| Tochno | Accessible Russian housing data | 956,147 records, 33 attributes, Aug 2026 snapshot, Sep 17 update; location, year, wall material, floor area; floors not verified. CC BY 4.0. | https://tochno.st/datasets/gisgkh |
| NextGIS Sochi | Local reference counts, purchasable joined data | 2025-03-03 AI buildings: MS 73,302; OSM polygons 68,057, points 57. 2026-04-27 housing: 2,233 polygons, 2,431 point records. Counts are not distinct physical buildings or coverage rates. | https://data.nextgis.com/ru/region/RU-CITY-046/msbld/ ; https://data.nextgis.com/ru/region/RU-CITY-046/gkh/ |
| FABDEM v1.2 | Approximate DTM | 1 arc-second, ML removal of building/vegetation bias, CC BY-NC-SA 4.0. Not survey-quality ground. | https://research-information.bris.ac.uk/en/datasets/fabdem-v1-2/ |
| Copernicus GLO-30 | DSM / comparison / terrain fallback | 1 arc-second, EGM2008 vertical reference for standard orthometric product. | https://registry.opendata.aws/copernicus-dem/ ; https://doi.org/10.5069/G9028PQB |
| OpenAerialMap | High-resolution licensed imagery catalog | STAC, CC BY 4.0; local availability not verified. | https://imagery.hotosm.org/ ; https://docs.imagery.hotosm.org/usage/using-imagery/ |
| Innoter | Evidence of local high-resolution imagery | Sochi orthophoto project completed May 2020 using WorldView-3 30 cm imagery; no public bulk download confirmed. | https://innoter.com/en/projects/creation-of-orthophotomaps-for-sochi-city/ |
| Sentinel-2 | Land cover and spatial context | 10m RGB/NIR; cannot resolve small individual roofs reliably. | https://www.esa.int/ga/articles/about-us/directorates/earth-observation-programmes/copernicus/sentinel-2/facts-and-figures |
| Mapillary / KartaView | Licensed street-level reference images | Check API terms and coverage; Mapillary imagery CC BY-SA per provider. | https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data ; https://wiki.openstreetmap.org/wiki/Street-level_imagery_services |
| EUBUCCO | Training/reference only | Main dataset EU, UK, Norway, Switzerland, not Russia. | https://docs.eubucco.com/ |

## 3. AOIs and reproducible acquisition

Proposed WGS84 bounding boxes, not administrative boundaries:

- Adler pilot: west=39.91 south=43.42 east=39.94 north=43.44 (~5.40 km², geodesic area; includes land and water if present).
- Adler extended: 39.87,43.38,40.03,43.51 (~187.07 km², not purely developed area).
- Add independently verified hillside, dense apartment, and industrial samples in later benchmark. Do not extrapolate pilot coverage to all Sochi.

Use a buffer for matching, tile ownership and cross-boundary relationships. Keep the original full polygons and count a feature once with a deterministic ownership rule. For the first audit the supplied Python script uses representative point in AOI; count invalid input before AOI filtering and distinguish parts. No geometry clipping or automatic merging.

### OSM

```sh
curl -L https://download.geofabrik.de/russia/south-fed-district-latest.osm.pbf -o south.osm.pbf
osmium extract -b 39.90,43.41,39.95,43.45 -s smart south.osm.pbf -o adler_buffer.osm.pbf
osmium tags-filter adler_buffer.osm.pbf w/building r/building w/building:part r/building:part r/type=building -o adler_buildings.osm.pbf
osmium export adler_buildings.osm.pbf -a type,id,version -u type_id -f geojson -o adler_osm.geojson
```

Commands proposed, not run here (network inaccessible). Verify references and relation completeness against original extract. `osmium export` is audit convenience, NOT canonical import: type=building relations, member roles, POI and address nodes must be read from PBF by pyosmium/libosmium. See https://docs.osmcode.org/osmium/latest/osmium-export.html and https://docs.osmcode.org/pyosmium/latest/user_manual/03-Working-with-Geometries/.

### Overture, quick access (latest)

```sh
pip install overturemaps
overturemaps download --bbox=39.90,43.41,39.95,43.45 --type=building -f geojson -o adler_overture.geojson
overturemaps download --bbox=39.90,43.41,39.95,43.45 --type=building_part -f geojson -o adler_parts.geojson
```

Record actual release; CLI automatically chooses latest. For repeatability use pinned Parquet URIs:

```sql
INSTALL spatial; LOAD spatial;
INSTALL httpfs; LOAD httpfs;
SET s3_region='us-west-2';
SELECT count(*) AS bbox_candidates,
       count_if(height > 0) AS positive_height,
       count_if(num_floors > 0) AS positive_floors,
       count(roof_shape) AS roof_shape_present
FROM read_parquet('s3://overturemaps-us-west-2/release/2026-11-18.0/theme=buildings/type=building/*', hive_partitioning=true)
WHERE bbox.xmax >= 39.91 AND bbox.xmin <= 39.94
  AND bbox.ymax >= 43.42 AND bbox.ymin <= 43.44;
```

These are bbox candidate counts, not exact AOI ownership counts. Use local geometry test for exact comparison. Prefer STAC-selected intersecting files rather than scanning all globbed files. Inspect schema and GeoParquet geometry type; do not assume all versions have identical fields. Documentation: https://docs.overturemaps.org/getting-data/ and https://docs.overturemaps.org/getting-data/overturemaps-py/.

### Other layers

- GBA: use representative index `lod1.geojson`, `height_zip.geojson`, `height_tif.geojson`, select intersections. 35–40E,40–45N and 40–45E,40–45N for extended AOI; polygons EPSG:3857 per upstream warning even if mislabelled. Verify coordinate magnitudes and metadata before transforming. Height raster and LoD1 must keep separate license/source manifest. https://mediatum.ub.tum.de/1782307
- Microsoft: use dataset-links.csv and tile geometry/quadkeys to select candidates; do not fetch whole planet. Preserve confidence and capture date if present; legacy -1 and current null both missing.
- GHS-OBAT: get RUS/admin subdivision for Krasnodar; match 2024 GERS IDs via registry/changelog or spatial fallback; never silently transfer to a split/rebuilt building.
- Tochno: download Parquet/CSV from provider; inspect column names and completeness before filtering by geographic AOI and municipality. Coordinate point is not verified footprint centroid; normalize Russian addresses and building/corpus/structure suffixes.
- GLO-30/FABDEM: choose intersecting one-degree cells; pilot lies in N43/E039, extended AOI also E040. Confirm file availability, raster CRS, NoData, vertical datum and acquisition epoch.
- OAM: use STAC search at https://api.imagery.hotosm.org/stac/search with bbox and datetime; record item, asset URL, capture epoch, GSD, sun/view geometry when available. Local availability not confirmed.

### Local audit

```sh
pip install 'shapely>=2' pyproj
python audit_building_data.py --osm adler_osm.geojson --overture adler_overture.geojson --overture-parts adler_parts.geojson --output adler_audit.json
```

Script syntax and synthetic fixture test passed in research environment; real network downloads and live local audit did not run. Reports field presence, positive numeric heights and integer floors, geometry quality, parts, primary source counts and IoU candidate matches. It does not resolve duplicates, reconstruct relations, validate all tag units, or estimate missing attributes. Do not treat counts as verified ground truth.

## 4. Coordinate and geometry policy

- Store canonical geographic geometry in WGS84 with explicit lon,lat order; use regional metric CRS UTM zone 37N (EPSG:32637) for Adler (39–40E), or local equal-area for area audit, ENU or ECEF doubles for georeferencing.
- Runtime: tile-local positions, double precision transforms to world and test coordinate axes/handedness in UNIGINE 2.22. No assumption that engine Z automatically equals orthometric height.
- Standard GLO-30 orthometric elevations: EGM2008. Record source vertical datum and convert explicitly when using ellipsoidal height, h=H+N. Check geoid grid availability; do not mix datums or assume geoid zero. DTM and scene terrain must use same processed surface.
- Maintain source footprint vs roofprint distinction. True orthophoto vs ordinary orthophoto: elevated objects can be horizontally displaced. For off-nadir angle theta, ideal flat-terrain approximate shift magnitude H*tan(theta); use actual viewing geometry/RPC if available; do not apply universal shifts. Roof overhang also changes roofprint size.
- Validate rings, holes, MultiPolygon, geometry dimensions, zero area, self-intersections, duplicate vertices. Keep original geometries; repair copies with GEOS make_valid; review severe changes, never blindly use convex hull. No global grid snapping without validation.

## 5. Conflation design

1. Immutable raw records, manifest, normalized units and geometries, source IDs.
2. Resolve exact ID/bridge matches with epoch and geometry sanity checks. Preserve explicit parent/part relation links separately.
3. Build STRtree/R-tree spatial candidate graph in metric CRS; start with 2–5m search expansion, calibrate larger tolerances for roofprint displacement only with evidence.
4. Features: intersection/union, overlap over smaller area, centroid distance normalized by object size, shape/orientation, address compatibility, date and source type.
5. Initial candidate rules: IoU >=0.5 high candidate; IoU >=0.3 plus centroid containment or intersection/min(area)>=0.8 review candidate. Not a calibrated probability. Do not automatically accept all candidates or apply transitive merging.
6. Resolve clean 1:1 separately; one-to-many and many-to-one require group/part treatment, aggregate geometry tests and original IDs. Distinguish duplicate observations from adjacent attached buildings. For overlapping roof/ground projection, align if evidence supports it.
7. Building outline is identity/coverage; parts are volumes. Do not extrude both blindly; preserve uncovered outline portion if parts incomplete, and support elevated volumes.
8. Keep unmatched ML records only after checking roads/water, size/type and source epoch. Avoid removing piers, stilt buildings and other valid exceptions solely by intersection. No semantic attribute inherited from arbitrary nearest building.
9. Record all decisions in a match ledger; support human/automatic review, rebuild and source updates. Use stable internal IDs and split/merge lineage. Persist seed independent of mutable GERS ID or exact footprint coordinates.

References: https://docs.overturemaps.org/guides/buildings/ ; https://github.com/OvertureMaps/workshop/blob/main/7-buildings-matching.md

## 6. BuildingDescriptor and attribute semantics

Example conceptual model (not an observed building):

```yaml
id: internal_stable_id
seed: stable_seed
geometry:
  footprint: polygon_ref
  roofprint: optional_polygon_ref
  parts: [part_id]
  crs: EPSG:4326
  kind: footprint_or_roofprint
observations:
  - field: floors_above_ground
    value: 5
    method: registry_reported
    source: gis_gkh
    source_record: example_only
    capture_date: null
    release_date: 2026-09-17
    match_quality: uncalibrated_high
    value_sigma: null
resolved:
  ground_elevation_m: null
  base_offset_m: 0
  eave_height_agl_m: null
  ridge_height_agl_m: null
  roof_shape: null
  roof_ridge_bearing_deg: null
  roof_slope_bearing_deg: null
  floors_above_ground: 5
  floor_height_m: null
  construction_epoch: null
  structural_wall_material: null
  facade_finish_material: null
  roof_color: null
  use_class: apartment
  architecture_profile: null
  confidence_class: unknown
```

Height reference MUST encode top/eave/ridge/median/mean/relative part extent/unknown. OSM height and min_height represent elevations above ground, while Overture schema describes height as distance lowest to highest and min_height as starting altitude above ground: verify conversion for parts against source and known examples. Do not copy numeric values across schemas blindly. OSM roof:direction describes slope/front-facing direction; Overture roof_direction schema describes ridge bearing. Keep separate, test along/across and angle normalization. `building:levels` excludes roof:levels. Construction year != renovation year. Registry structural material != visible cladding. Missing values are null, not zero. Confidence is not probability until calibrated. Preserve capture time separately from download/release timestamp.

## 7. Height and floor resolution

Preference per observation, not globally fixed ranking: verified surveyed height or matching authoritative source, reliable explicit tags/registry floors, locally validated ML heights (GBA, MS, 3D-GloBFP), neighborhood prior (GHSL/WSF), regional archetype priors. Conflicting explicit sources must be checked; never force unverified estimates over explicit valid values. Use robust residuals, empirical local calibration stratified by low/mid/high-rise, and avoid double counting shared source families. GHS-OBAT and OBM both use GHSL; Overture may already include Microsoft values. Derive candidate floor count, eave and ridge jointly under type-specific plausible floor-height ranges and roof geometry; preserve original observations and resolved confidence class. If floor count trusted, use ML height to constrain roof/floor spacing, not override it by rounding. Registry floor area / footprint can be a weak check, not ground truth. For GBA raster sample inward mask, NoData fraction, median/quantiles, count of distinct pixels and neighbor contamination; use robust representative height, not max pixel. Minimum pixel count/coverage calibrated; a 6x6m roof contains very few 3m samples. For GBA/DSM height differences do not subtract ground twice.

## 8. Terrain and imagery

- FABDEM is approximate bare-earth; DEM at 1 arc-second near 43.4N is about 22.5m east-west, 30.9m north-south. Upsampling does not add detail. DTM sampling from same processed terrain mesh ensures visual fit. Sample footprint boundaries/interior and surroundings; robust base elevation, horizontal walls, adaptive foundations on slope, step foundations or flag excessive relief. No per-house precise geometry inferred from a 30m raster.
- Prefer true ortho, RGB 0.2–0.5m for roof color and major form; 1m may support coarse large roofs, 3m only coarse tall/large roofs, Sentinel-2 10m neighborhood context. Capture epoch matters. No confirmed free citywide 30cm imagery found. Innoter historical coverage is proof of acquisition, not an open asset.
- Roof color: project roofprint or corrected footprint, erode mask, reject shadow/vegetation/edge pixels, estimate robust Lab dominant colors, map to palette with confidence. Separate baked lighting from albedo; don't infer exact facade colors from roofs. Never reuse tile imagery as PBR albedo without processing/rights.
- Roof shape: prioritize explicit tags or high-quality point cloud planes; imagery classifier optional later. Use existing geometry constraints and regional archetype for unknowns. Shadow-height approx H=L*tan(sun elevation) on level ground and correct shadow direction/terrain; unsuitable with occlusions, unknown metadata or low sun. No requirement to build ML before baseline.
- Source-dependent access rights are not removed by personal use; keep manifests and optional noncommercial inputs isolated. Do not scrape restricted map imagery or use unauthorized APIs.

## 9. Validation and deliverables

- Stratified truth set proposed: 200–400 buildings (not measured), across private houses, guest houses, apartment blocks, high-rise, industrial, slopes. Verified geometry/height reference from permitted sources; label method and uncertainty. Use separate calibration and held-out districts; avoid same upstream data as both prediction and truth.
- Metrics: per-source attribute present/valid/conflict fractions, distinct owner feature counts, candidate 1:1 and ambiguous group counts, unmatched candidates, spatial precision/recall against independent reference, IoU, height MAE/RMSE/bias by class, floor count exact and +/-1, roof shape macro F1, observed vs imputed proportions, terrain contact errors, confidence calibration.
- QA outputs: GeoPackage or GeoParquet normalized buildings/parts, match ledger, attribute observations, resolved descriptors, AOI report JSON, human-readable map of conflicts, manifest with checksums, versions, source URLs, licenses, dates, CRS, processing options. Use GeoJSON only for small debug exports. CityJSON optional interchange for semantic LoD. Runtime binary tile format separate.
- Storage: DuckDB + spatial/GeoParquet for offline analysis, GEOS/Shapely + PROJ + GDAL/Rasterio, pyosmium/libosmium for OSM. PostGIS optional if multi-user editing or persistent server workflows become necessary; not required now. Use COG for raster data. Python research pipeline first, C++ only when benchmark justifies it. Independent building compiler consumes stable descriptors and never calls GIS services at runtime.

## 10. Blockers and next stage

No full data downloads from Overpass, Overture S3 or Tochno in container: DNS resolution failed; browser API query also unavailable. Published NextGIS counts not equal to a fresh measurement. Actual Adler coverage, heights, roof data, exact overlap and local accuracy remain unresolved until running the supplied audit locally. No purchase, licensed image download or engine test performed. Source release and local data coverage must be verified before integration. Proposed AOI, match thresholds, storage choices, hierarchy are design decisions, not benchmarks. Next research: computational geometry (OSM part topology, robust polygon repair, straight/weighted skeleton, triangulation, terrain foundations, test corpus).
