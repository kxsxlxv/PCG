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
