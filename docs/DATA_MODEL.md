# Canonical contracts v1 (initial)

JSON schemas and Python types to be implemented in first compiler increment, not silently inferred from old probes. All length units meters; coordinate arrays explicitly ordered. Reject nonfinite numbers, unknown major schema version, missing CRS/vertical reference where required. Record schema revision and compiler revision on every persisted artifact. Stable IDs are strings; no engine object ID or array index used as durable ID.

| Record | Required fields / invariants |
|---|---|
| SourceSnapshot | source_id, URI, version/epoch, license/access terms, checksum, CRS, vertical datum/reference, retrieval metadata |
| WorldFrame | horizontal CRS, flat/curved mode, axis order, origin float64, vertical datum, units, transform revision; flat pilot x=E-E0 y=N-N0 z=H-H0 |
| Observation | entity candidate, attribute, value, unit, source snapshot+feature ID, method, epoch, quality/uncertainty, reference (eaves/total/relative etc.), evidence class observed/estimated/inferred, status |
| BuildingDescriptor | stable ID, source links, geometry Polygon/MultiPolygon with holes, part IDs/relationships, observations, resolved use/height/floors/roof, decisions and fallbacks; retain original tags |
| ResolvedBuilding | accepted geometry and attributes, authoritative/inferred distinction, floor datum, explicit eave/roof top/min offsets, profile ID/revision |
| SemanticSurface | stable ID, building/part, role, polygon 3D with holes, plane/frame, edge IDs, material role, source edge, LOD policy |
| SemanticEdge | endpoint IDs/coordinates, incident surfaces, role ridge/valley/hip/eave/gable/party/foundation/seam |
| ArchitecturePlan | building ID, profile revision, floor levels, facade IDs, local frames, bays/slots, openings (u,z,w,h), accessory anchors, material/occupancy intents; one plan for all LOD |
| AssetRecipe/Manifest | family/revision, metric size, fixed profile dimensions, fit bounds, mount sockets, material roles, cache hash, LOD resources/errors/bounds, license/QA |
| MaterialRecipe/Manifest | ID/revision, physical repeat, map semantics, sRGB/linear, normal convention, amplitude, source/hash, channel packing, resolution/mips |
| MeshGroup | float32 local positions, normals, UV0, optional UV1/color, indices, role/material; split at hard normal/UV/discrete attribute boundaries; topology separate |
| Placement | stable placement ID, building+anchor ID, asset ID, local transform, variant fields; parent tile origin double |
| TileManifest | ID/revision, world frame, origin, owned building IDs, actual bounds, parent/children, coverage, geometric error, resource hashes/dependencies, byte/triangle/instance counts |
| BuildReport | source/config/code hashes, tool versions, seed policy, stage timings, per-stage status, rejected/fallback reasons, QA and preview paths |

Canonical geometry is right-handed Z-up (XY horizontal); glTF Y-up conversion (x,y,z)->(x,z,-y), inverse (x,-z,y). Asset glTF +Z outward maps to -Y, placement explicitly rotates to wall frame. Use right-handed frames, no hidden reflection. Orthometric vs ellipsoidal transform requires grid/version; don't add curvature to flat host. Canonical metric double before tile-local float. Semantic height reference: ground datum, min offset, eave, top separately; never conflate roof height and total height. Preserve exact footprint unless repair accepted with measured change. Geometry validity and rejected parts logged. Stable ID registry supports 1:N/N:1 conflict groups rather than blind merge.
Serialization: GeoParquet for large normalized GIS tables when needed, JSON/GeoJSON fixtures and manifests, neutral GLB debug + semantic sidecar. Later packed tile binary schema explicitly versioned; no speculative native file format. Deterministic canonical JSON (sorted keys, UTF-8, finite values, documented quantization) and SHA256 resource hashes. Timings excluded from semantic deterministic hash. 
