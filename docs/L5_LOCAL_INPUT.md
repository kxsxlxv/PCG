# L5.1 — local source snapshot contract

This increment accepts an explicitly supplied local WGS84 GeoJSON FeatureCollection and a separate JSON manifest. It does not download data, establish usage rights, infer a building's identity or normalize attributes. The synthetic fixture is original PCG data, not an observed Adler building.

## Validate a local snapshot

From the repository root, using the existing pinned Lab environment:

```sh
PYTHONPATH=src python lab/audit_source_snapshot.py \
  --manifest tests/fixtures/l5_synthetic.snapshot.json \
  --payload tests/fixtures/l5_synthetic.geojson \
  --output artifacts/l5-source-snapshot/report.json
```

For a real authorized dataset, use a local manifest and explicit local payload path outside tracked source. Compute SHA-256 over the exact original file bytes (e.g. `sha256sum input.geojson`), record the actual release, date of retrieval with UTC offset, source URI without authentication or query parameters, license and access terms. Mark the actual capture epoch or `unknown`; do not substitute retrieval date for capture epoch. Declare `authorized_local` only when use is permitted; this declaration is not independent legal validation. Keep the raw data, manifests containing confidential metadata and any derived output subject to license restrictions local. CI only consumes the checked-in synthetic fixture.

The current adapter accepts OGC:CRS84 or EPSG:4326 with explicit longitude/latitude ordering and rejects legacy GeoJSON CRS members. `not_applicable_2d` denotes no absolute elevation datum in a 2D footprint source; it does not give a datum or reference to building height attributes. The audit counts declared geometry types, not valid polygons, distinct physical buildings or reliable attribute observations. Coordinate magnitude, geometry validity, source IDs, part relationships and height reference normalization are later gates.

## World frame

`WorldFrame` accepts a projected CRS with east/north metre axes and an explicitly named vertical datum. The flat local convention is (E-E0,N-N0,H-H0). The horizontal projection uses the already pinned pyproj/PROJ implementation with longitude-first axis order and no ballpark transform; network-enabled PROJ is rejected. The caller must choose the actual host CRS and supply a matching datum; no vertical transformation or geoid grid is performed. EPSG:32637 is the proposed pilot projection, not proof of host compatibility.

Next: normalized immutable source features and observations, preservation of invalid/rejected records, then candidate identity graph and local DTM sampling. Real Adler quality acceptance needs authorized snapshots and independent reference evidence.

## L5.2 — normalized source records

```sh
PYTHONPATH=src python lab/normalize_gis.py --manifest snapshot.json \
  --payload input.geojson --frame frame.json --kind osm \
  --bbox 39.91 43.42 39.94 43.44 --output normalized.json
```

Supported kinds: `osm`, `overture_building`, `overture_part`. The kind is explicitly supplied by the caller, not guessed from a filename. OSM requires typed way/relation identity (osmium type_id or @type/@id); Overture requires an explicit feature ID. Duplicate IDs are all rejected rather than selecting the first. Missing ID is not replaced by array position. Each accepted or rejected record retains canonical JSON of the raw feature and the verified source snapshot digest. Parts and buildings are separate records; parent relationships remain in raw properties for later fusion.

Only valid 2D Polygon/MultiPolygon geometries with closed finite WGS84 rings are accepted. GEOS validates original and projected topology. Representative-point AOI ownership preserves whole polygons without clipping; no repair or antimeridian handling in this increment. PROJ projects to the declared metric frame and subtracts the horizontal origin; no absolute Z is invented. Normalized WKB uses GEOS canonical ordering; the exact original input remains in the verified snapshot. Reject ledger accounts for every excluded feature, including outside-AOI and nonbuilding entries.

Numeric observations distinguish OSM `height` (top above ground) from Overture `height` (vertical extent), and `min_height` (base above ground). Roof height is a separate extent. OSM accepts plain decimal metres, optional m suffix, decimal comma and integral floor count; ambiguous units, fractional floors and malformed values are retained as invalid rather than coerced. Overture numeric fields must be numbers. Source enum strings remain candidates, not resolved architecture classes. Unknown evidence and uncertainty are explicit; provider-reported values are not verified measurements. OSM building material and Overture facade material are not treated as equivalent. No automatic part parent inference, identity fusion, height arbitration, complete provider schema validation, or PBF relation reconstruction.

The original synthetic fixture exercises holes, MultiPolygon, building/part overlap, invalid geometry, duplicate/missing IDs and source-specific height semantics. CI uploads both normalized/rejected JSON records and a fixed metric top-view of actual projected geometries. Do not publish real normalized output containing restricted source properties or coordinates. A real authorized Adler sample remains required for L5 acceptance.

## L5.3 — cross-source candidate graph

`pcg.geo_match.build_graph(left, right, policy)` consumes normalized SourceFeature records from two distinct source layers, one snapshot per side, in exactly the same WorldFrame. It excludes parts from building matching and preserves their keys for subsequent parent handling. The existing pinned Shapely STRtree/GEOS dwithin query limits pair evaluation; intersection, IoU, overlap over smaller area, centroid distance and boundary gap use metric coordinates. The previous research audit thresholds are retained as candidate heuristics: IoU >= 0.5 high, overlap over smaller >= 0.8 or IoU >= 0.3 with centroid containment review. Disjoint footprints within 3 m are proximity review only; boundary touch is separate, not a match. Weak overlaps are logged separately. Values are uncalibrated, not confidence probabilities.

The bipartite graph partitions into isolated high candidates, 1:1 review, ambiguous 1:N/N:1/N:M groups and unmatched nodes. Even an isolated high candidate is NOT accepted identity. Group IDs are deterministic hashes of source-key membership for audit, not persistent building IDs. No transitive union, no attribute transfer, no source geometry mutation. Exact bridge-ID matching, epoch/address compatibility, persistent identity registry and manual adjudication remain later work.

Run `PYTHONPATH=src python lab/match_gis.py` after the L5.2 synthetic Lab to create the candidate ledger and metric plot of actual source geometries. Unit fixtures include one-to-many, many-to-many, proximity, touching, weak overlap and incompatible frames; the plotted L5.2 fixture covers isolated high, containment review, and unmatched cases. Real AOI validation is still required.

## L5.4 — explicit adjudication and attribute resolution

The caller supplies an internal building ID, graph digest, group ID, chosen geometry source and reason. The resolver rebuilds the graph and rejects stale graph digests, duplicate approvals and geometry sources outside the approved group. Only isolated high candidates and unmatched singletons can be explicitly approved in this bounded increment; all review and ambiguous groups remain unresolved. Internal IDs are caller-owned, not derived from vendor IDs or geometry hashes. Persistent registry storage and split/merge history remain deferred. No automatic identity acceptance.

Attribute resolution is deliberately conservative: exact agreement of compatible source observations, a single candidate when no competing usable value exists, or abstention on conflict. No averaging or blanket provider priority. Every candidate records the source, snapshot, original field/reference, evidence class and any assumption. OSM top AGL is used directly; Overture vertical extent is added to the explicit base AGL or a flagged ordinary-building zero-base assumption if absent. Invalid base prevents extent conversion. Positive elevated base is not supported for building mesh generation. Eave AGL = top AGL minus roof rise; zero roof rise is derived for explicitly flat roofs only. Pitched roof rise and floor count must be available. Contradictory height/roof/floor values block readiness. The broad 2–6 m floor-height check is a synthetic policy, not calibrated regional evidence. Only specific known house/apartments/office and flat/gable/hip aliases map to existing profiles/roof kinds; generic residential or unsupported labels are not guessed. Invalid/unsupported observations remain in the decision ledger. Candidate readiness is not verified height accuracy or a successful mesh build.

Run `PYTHONPATH=src python lab/resolve_gis.py` after normalization. The explicit original synthetic fixture approval yields top 7.5 m, roof rise 1.5 m, eave 6 m, two floors, house profile and hip roof; other groups remain unapproved. No geometry changes or new visual output in this increment. Real data still requires independent reference and source provenance review.

## L5.5 — local DTM input and foundation datum proposal

Use a separate verified GeoTIFF snapshot with `format: geotiff`, `axis_order: east_north`, projected metric horizontal CRS, explicit vertical datum in `vertical_reference`, source/license/release/epoch/access declaration and SHA-256 of the actual raster bytes. The caller must establish that the raster is a DTM, not a DSM, and that the elevation values in metres and vertical reference match the final host terrain. No claim that TIFF metadata establishes a reliable vertical datum. The source and actual raster CRS must equal the WorldFrame CRS. No automatic reprojection, datum transformation, scale/offset application or terrain generation. Only single-band north-up affine GeoTIFF, at least 2x2, max 128 MiB file and 16 million cells; GCP/RPC/rotated rasters are rejected. This bounded initial adapter is intended for local cropped samples, not a full regional mosaic.

Rasterio/GDAL performs the actual raster I/O, CRS metadata, affine transform, and validity mask. The thin sampler uses the inverse affine and bilinear interpolation between pixel centers, rejecting extrapolation and any contributing NoData/nonfinite cell. The raster is read from the same checksum-verified bytes through MemoryFile. No remote URLs or implicit sidecar files. The world vertical origin is subtracted once.

`foundation_plan` accepts an explicitly approved candidate-ready building in the matching world frame. GEOS segmentizes its exterior and hole boundaries; interior points on a metric lattice and a representative point supplement the boundary. Missing sample coverage or sampled relief above the configured limit (default 3 m) yields review_required. The proposed horizontal base is the maximum sampled local elevation; eave/top are translated by the same base, preserving vertical walls. These are sampled extrema, not continuous terrain guarantees, not a constructed foundation or a structural design. The default spacing of 2 m and fill limit are synthetic policy defaults. MultiPolygon foundation plans and elevated buildings are deferred. Real raster values, terrain alignment and foundation geometry are not yet validated.

Run `PYTHONPATH=src python lab/terrain_gis.py` after the L5.4 lab to produce an original analytic synthetic GeoTIFF, manifest, actual sampled-footprint plot and JSON. No real data is downloaded.

### Dependency review
Rasterio 1.4.3, upstream tag commit `7f8bda2e32df5b5eefc6e1ea6b5f04a946567ede`, BSD-3-Clause ([license](https://github.com/rasterio/rasterio/blob/7f8bda2e32df5b5eefc6e1ea6b5f04a946567ede/LICENSE.txt)). Inspected `rasterio/io.py` MemoryFile, `rasterio/transform.py` affine/pixel-center API and setup.py dependencies: GDAL native extension, affine, attrs, certifi, click, cligj, numpy, click-plugins and pyparsing (importlib-metadata only on older Python). Uses installed library, no copied upstream code. Direct package pinned; transitive/GDAL versions recorded in CI, not fully locked yet. Rasterio solves raster decoding and georeferencing; small explicit interpolation/NoData and foundation policy remain PCG adapters.

## L5.6 — local package compiler and review gate

The integrated compiler accepts a local JSON package, e.g.:

```json
{
  "schema_version": 1,
  "frame": "frame.json",
  "bbox_wsen": [39.91, 43.42, 39.94, 43.44],
  "sources": {
    "osm": {"manifest": "osm.snapshot.json", "payload": "osm.geojson"},
    "overture_building": {"manifest": "overture.snapshot.json", "payload": "overture.geojson"},
    "overture_part": {"manifest": "parts.snapshot.json", "payload": "parts.geojson"}
  },
  "dtm": {"manifest": "dtm.snapshot.json", "payload": "terrain.tif"},
  "seed": 17
}
```

Part layer optional. Paths are local, relative to package directory or absolute. Manifest and raster constraints from L5.1–L5.5 apply. No provider download or hidden use of latest release. Do not store restricted data in this public repository or Actions artifacts.

```sh
PYTHONPATH=src python lab/compile_gis_pilot.py prepare --package /local/package.json --out /local/prepared
PYTHONPATH=src python lab/compile_gis_pilot.py compile --package /local/package.json --approvals /local/approvals.json --out /local/compiled --preview
```

Each output directory must not exist. Review prepare.json before making an approval file. Its package digest binds all source manifests, frame, AOI, seed and graph; graph digest binds candidate membership/metrics. Approval JSON fields: schema_version (1), package_digest, graph_digest, approvals (array of group_id, building_id, geometry_source {source_id, feature_id}, reason). Do not approve a geometric candidate as actual identity without independent evidence. The compiler rechecks both digests and refuses stale approvals. Unapproved groups, invalid attributes, part layers and rejection reasons stay in output. No automatic split/merge or source transfer.

Compiler integration reuses existing polygon massing, facade plans, opening meshes, shared material/asset recipes, occupancy and DTM. Supported flat and convex hip single polygons (holes supported for flat). Unsupported gable facade integration, roof fallback, ambiguous identity, unresolved attributes, elevated buildings and invalid terrain sampling require review. All intersecting building render envelopes in the approved batch are conservatively rejected; attached/party-wall complexes remain for L6. This is not a complete building collision test and does not compare against unapproved geometry. No new dependency. Semantic welded shell validity is checked independently of split rendering geometry. Actual footprint retained in source report; canonical building-local copy preserves area/holes. Each building receives two neutral day/night GLBs and a semantic sidecar; GLB local Y-up converts to Z-up (x,-z,y) then adds anchor_local_z_up; anchor_projected_enh includes WorldFrame origin. This avoids storing large projected coordinates in GLB float32 vertices. No integrated foundation mesh or terrain surface; DTM proposes a horizontal floor datum. No scene tile pack or native engine claim.

The mandatory synthetic lab constructs two supported buildings (hip house and flat courtyard office), one approved invalid record and one unapproved group using original synthetic inputs; checks actual GLB roundtrip and optional CPU views from reloaded meshes. Real Adler acceptance is a separate stop gate: supply authorized pinned OSM/Overture exports, DTM aligned to actual host, explicit approval ledger and an independently reviewed subset (source feature IDs, geometry/identity verdict, floor/height/roof observations with provenance, ground elevation reference and uncertainty). Record discrepancies and unresolved cases, inspect aerial/street-scale previews, and report coverage/accuracy separately from synthetic checks. No real data or independent reference evidence has been provided yet.
