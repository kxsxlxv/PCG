# GeoWorld PCG — 02. Open Source Analysis

Date: 2026-11-24. Target: UNIGINE 2.22. Independent from Sandworm. Research prototype, not a production implementation.

## Decision summary

Use an engine-independent offline compiler, not a monolithic fork. Maintain two complementary baselines: OSM2World for OSM semantics/geometry and Condor Buildings Generator for flight-simulator-oriented generation. Investigate OSM Building Grammar for semantic facade geometry and asset export, osm2city for classification and material compatibility, and CGAL for robust roof construction. Keep optional systems behind adapters. Blender is an offline asset authoring/baking tool, not a runtime dependency.

## Evidence and donor map

| Source | Evidence inspected | Candidate reuse | Important caveat |
|---|---|---|---|
| [OSM2World](https://github.com/tordanik/OSM2World) | [BuildingPart.java](https://raw.githubusercontent.com/tordanik/OSM2World/master/core/src/main/java/org/osm2world/world/modules/building/BuildingPart.java), [Roof.java](https://raw.githubusercontent.com/tordanik/OSM2World/master/core/src/main/java/org/osm2world/world/modules/building/roof/Roof.java), [HippedRoof.java](https://raw.githubusercontent.com/tordanik/OSM2World/master/core/src/main/java/org/osm2world/world/modules/building/roof/HippedRoof.java), [AbstractHippedRoof.java](https://raw.githubusercontent.com/tordanik/OSM2World/master/core/src/main/java/org/osm2world/world/modules/building/roof/AbstractHippedRoof.java) | OSM part semantics, inherited tags, level structure, roof interfaces; GLB baseline | Hipped roof uses ridge-based approximation; not general straight skeleton |
| [OSM2World default style](https://github.com/tordanik/OSM2World-default-style) | [standard.properties](https://github.com/tordanik/OSM2World-default-style/blob/master/standard.properties) | Physically scaled materials, tint metadata | Inspect each resource and attribution |
| [osm2city](https://gitlab.com/osm2city/osm2city/) | [building_lib.py](https://gitlab.com/osm2city/osm2city/-/raw/master/osm2city/building_lib.py), [roofs.py](https://gitlab.com/osm2city/osm2city/-/raw/master/osm2city/roofs.py), [texture.py](https://gitlab.com/osm2city/osm2city/-/raw/master/osm2city/textures/texture.py) | analyse_height_and_levels, analyse_textures, RoofHint, semantic texture matching | FlightGear-specific pipeline; regional heuristics need calibration |
| [Condor Buildings Generator](https://github.com/jlgabriel/condor-buildings-generator) | [building_generator.py](https://raw.githubusercontent.com/jlgabriel/condor-buildings-generator/main/condor_buildings/generators/building_generator.py), [roof_polyskel.py](https://raw.githubusercontent.com/jlgabriel/condor-buildings-generator/main/condor_buildings/generators/roof_polyskel.py), README | Roof dispatch, fallback logic, MeshData, terrain solver and atlas mapping candidates | Complex roof path depends on mathutils and embedded bpypolyskel; fallback outside Blender unless dependency supplied |
| [OSM Building Grammar](https://github.com/p-schulz/osm_building_grammar) | [grammar.py](https://github.com/p-schulz/osm_building_grammar/blob/main/grammar.py), [builder.py](https://github.com/p-schulz/osm_building_grammar/blob/main/builder.py), [exporter.py](https://github.com/p-schulz/osm_building_grammar/blob/main/exporter.py) | generate_building_spec, BuildingSpec, MeshSpec, geometry deduplication, GLB blueprint | Single-ring API in inspected function; test holes, transforms, scale/UV and batching; Blender-facing wrapper requires bpy |
| [Symbios Shape](https://github.com/TheJanusStream/symbios-shape) | README and [API documentation](https://docs.rs/symbios-shape/latest/symbios_shape/) | CGA-like derivation, split/repeat, seeded stochastic rules | Experimental candidate; no benchmark or full code audit |
| [GameRealisticMap](https://github.com/jetelain/ArmaRealMap) | [Architecture](https://github.com/jetelain/ArmaRealMap/blob/master/docs/architecture/overview.md), [builders reference](https://github.com/jetelain/ArmaRealMap/blob/master/docs/architecture/builders-reference.md) | Typed lazy data builders, local coordinates, game adapter, asset matching | Model placement, not precise arbitrary building reconstruction |
| [osm2xp](https://github.com/32kda/osm2xp) | README | Hybrid facade/model selection; regional library ideas | Simulator-specific assets and output |
| [rsgeotools](https://github.com/romanshuvalov/rsgeotools) | README | Batch generation, semantic vertex attributes for material selection | Limited roof shapes documented |
| [BCGA](https://github.com/vvoovv/bcga), [Building Tools](https://github.com/ranjian0/building_tools) | README | Offline procedural architectural modules | Validate Blender compatibility, unattended execution |
| [CGAL](https://doc.cgal.org/latest/Straight_skeleton_2/index.html) | [Skeleton extrusion](https://doc.cgal.org/6.2/Straight_skeleton_2/group__PkgStraightSkeleton2Extrusion.html) | Weighted roof skeleton and extrusion, including holes | Input must be valid; handle numeric and topological failure |
| [Roofer](https://github.com/3DBAG/roofer) | [Input requirements](https://innovation.3dbag.nl/roofer/data_requirements.html), [API](https://github.com/3DBAG/roofer/blob/main/docs/api_py.md) | Optional point-cloud reconstruction | Requires suitable roofprint and point cloud; not a substitute for OSM-only generation |
| [Material Maker](https://github.com/RodZill4/material-maker) | README, release notes | Procedural PBR material production | Test batch exporter on installed version |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | README/API | Vertex cache, mesh compression, attribute-aware simplification | Preserve semantic silhouette and UV seams |

## Proposed interfaces (not existing APIs)

- GeoDataProvider -> BuildingDescriptor[] (IDs, provenance, polygons/holes/parts, dimensions, roof hints, confidence).
- SemanticResolver -> ResolvedBuilding (height and levels with unambiguous reference; archetype; material requirements; deterministic seed).
- RoofBackend -> RoofSurface[] (face polygons, role, pitch, eave/ridge elevations, diagnostics). Backends: analytic, CGAL, flat fallback; optional Roofer.
- FacadeBackend -> FacadePlan (floor/bay dimensions, facade-facing direction, semantic slots, material recipe, instances). Candidate OSM Building Grammar or Symbios.
- MeshCompiler -> BuildingMeshLOD[] (local coordinates, semantic material IDs, normals, tangents, UV, bounds, error, instance records).
- TileCompiler -> engine-neutral packages; UNIGINE adapter is the only engine-specific layer.
- AssetFactory -> reusable meshes and PBR material sets; run offline.

## Pre-implementation experiment

1. Pin dependency commits and record licenses and access conditions; do not depend on proprietary procedural runtime. Clone donors locally; build baseline generators.
2. Build identical fixture inputs: rectangles, rotated rectangles, L/U/T footprints, hole, near-collinear edges, tiny edges, multipolygon, parts with different heights, passage, terrain slope, invalid geometry, roof-direction tags, missing and contradictory heights.
3. Compare baseline geometry using footprint IoU, roof coverage, validity, height error, deterministic results and fallback counts. Include finite-coordinate checks, triangle winding, nondegenerate triangle checks, closed-shell tests where applicable, UV seam tests.
4. Run OSM Building Grammar on the same fixture descriptors; compare direct core output and Blender adapter, count asset uniqueness, test export transforms including rotation and nonuniform scale. Do not assume support for holes.
5. Compare analytical roofs, CGAL and bpypolyskel. Test weighted slopes, vertical edges, courtyard holes, offsets, failure cases.
6. Compile a small real Adler sample to GLB and inspect with repeatable camera shots. Then test a native UNIGINE adapter using installed SDK, not hypothetical APIs.
7. Measure per-stage wall time, peak RAM, mesh count, vertices/triangles, material groups, tile bytes, reuse ratio. Runtime benchmark is separate (CPU/GPU frame time, P95/P99, VRAM, streaming spikes, shadows).

## Research limitations

Selected raw source files were inspected remotely; repositories were not fully cloned or built because the working container could not resolve GitHub. No commit SHAs were pinned; no actual Adler data, renders, performance measurements or UNIGINE SDK experiments were run. The other candidates are documented at README/API level. Proposed architecture and priorities are engineering judgments, not empirical results. Source links point to mutable branches and must be pinned before implementation. This file is a compact handoff, not a complete transcript of the research.
