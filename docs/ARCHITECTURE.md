# PCG — agreed architecture, v0.1

Status: design baseline from research 00–09, subject to explicit acceptance gates. This document and the contracts in docs/ supersede preliminary proposals in research/. No production implementation is implied.

## Scope and hard constraints
Independent real-world building compiler for UNIGINE 2.22, first Adler pilot, then Sochi. Flight simulator, regional architectural realism, no mandatory 3D artist, no Sandworm, one RTX 5070 Ti desktop. No training, fine-tuning, manual model tuning or runtime AI. Reuse OSS; document licenses and access rights. Public repository contains original fixtures and approved redistributable assets only. No terrain generator, roads, vegetation, traffic or building interiors in initial scope. Existing terrain is an integration input, not a dependency on Sandworm. Do not require SDK to develop the core.

## Boundaries
```
Source snapshots -> GIS adapters -> fusion / identity / observations
       -> resolved descriptors -> massing / roof / foundation
       -> architecture plan (facade, openings, anchors)
       -> semantic mesh + reusable asset placements
       -> material resolver + authored LOD
       -> neutral tile pack + QA
                       |-> Visual Lab (CPU / Blender)
                       |-> native resource packer (future SDK)
                                  -> native runtime (future SDK)
```
Offline compiler: Python 3.11 initially, typed dataclasses + validated JSON contracts, NumPy/Shapely/PROJ. Keep expensive backends behind narrow interfaces. C++ geometry backend only when complexity/correctness/performance justify it; use CGAL for general straight skeleton after independent build tests, not hand-written universal skeleton. C++ SDK packer and runtime separate; no Python, PyTorch, Blender or GIS dependency in simulator. Engine-independent Lab validates actual intermediate meshes, not approximated drawings. Standard GLB for interchange and preview, canonical semantic data separate. Do not use glTF as sole source of truth or as an in-flight parser requirement.
 
## Modules and dependencies
Proposed repo: src/pcg/{contracts,geo,resolve,geometry,architecture,assets,materials,compile,tiles,validation}; lab/ for render/report CLI; tests/{unit,fixtures,integration,golden}; configs/{regions,profiles,materials}; research/ historical probes; native/{packer,runtime,tests} deferred. Core has no import from lab/native. GIS source records immutable. Resolver makes explicit accepted/fallback decisions. Stable internal identity independent of changing vendor IDs; deterministic per-domain substreams independent of ordering. Geometry produces topology and semantic surface/edge graph; architecture owns one coherent floor/bay plan across all LOD; asset factory produces recipe-driven modules with fit anchors; material resolver uses palette/material IDs, not per-building shaders. Tile compiler owns unique building sets, resource manifests and shared placements. No cyclic module dependencies.

## Conflict resolutions from research
- Geo: source provenance and uncertainty first, no global priority that always chooses inferred floors over measured GBA/other heights; compare reference type and confidence. Overture consolidated candidate base + OSM semantic/parts + alternative sources, do not append all datasets. Flat projected pilot EPSG:32637 only if matches host; explicit vertical datum required; no curved ENU sag on flat terrain.
- Geometry: preserve footprint/holes/parts; constrained triangulation; analytic simple roofs, complex skeleton later; safe flat fallback with reason. Roof shape hypothesis not proof of actual shape. Foundation follows final DTM and does not tilt walls. Keep shell topology separate from split render vertices and visual attachments. Correct previous probe deficiencies (proxy windows, arbitrary fixed slopes, changing LOD bounds, no full intersections).
- Architecture: six illustrative profile seeds now, expand after regional evidence. Typed JSON rules over embedded executable DSL; stable floor/bay/stack IDs; no fabricated historical series. No random variation between LOD. Geometry handles actual recessed loggias; modules handle protruding details.
- Materials: shared PBR and measured-scale UV; arrays/atlas as adapter choice. Custom Parameters direct draw-call read documented, native gate pending; merged buildings require vertex attributes or bounded groups. Painted metal dielectric. Far masks/emission preserve mean radiance. Start basic materials; optional interiors/POM/AI never block base compiler.
- Runtime: spatial batches for unique shells, shared meshes for repeats; native cluster and streaming after SDK tests, own HLOD and predictive tile scheduler. Leaf ownership fixed, content bounds include protrusions; parent/child coverage exclusive except bounded transition. Distinct collision/query proxies. No GPU-driven renderer assumption.
- AI: optional offline enrichment of missing attributes only; existing footprints and heights preferred; CPU color/line analysis first. Pretrained checkpoints only; skip uncertain/unavailable/over-budget predictions. No training. No neural dependency for MVP.
 
## Development gates
First build Visual Lab + deterministic compiler independently of SDK, then actual local-data pilot. Introduce native packer only when canonical geometry, material/axis contract and representative visual quality have passed review; do G0–G8 on local SDK before native scale work. G9–G12 before runtime deployment. SDK can be absent for the first several increments; cannot validate its shaders, upload, draw counts, collisions or FPS in Lab. See [roadmap](IMPLEMENTATION_ROADMAP.md), [testing](TESTING.md), [CI lab](CI_VISUAL_LAB.md), [data model](DATA_MODEL.md).
