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
