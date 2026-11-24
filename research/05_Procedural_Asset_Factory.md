# Research 05 — Procedural Asset Factory

Date: 2026-11-24 | PCG | UNIGINE 2.22 | Independent of Sandworm

Status: design and source/documentation survey, with executed dependency-free procedural mesh experiment. No Blender/Cycles, AI inference, UNIGINE or external glTF validator execution. All geometry in the probe is original generated geometry; no third-party models or textures are bundled.

## 1. Decision

Make an offline factory of parameterized architectural components, not a warehouse of fixed buildings. ArchitecturePlan owns semantic placements and dimensions; AssetRecipe specifies geometry and materials; the compiler emits independently replaceable geometry, textures and metadata. Analytical geometry is preferred for dimension-critical, regular hard-surface elements. Blender Python/BMesh and Geometry Nodes are authoring/baking tools for complex components, not runtime dependencies. AI-generated assets are optional donor geometry for non-regular decoration, never authoritative architectural dimensions.

Proposed flow: recipe validation -> canonical size resolution -> high-poly and semantic low-poly -> metric UV/trim or unique UV -> baking -> semantic LOD -> mesh optimization -> technical and visual QA -> asset manifest + export. Use content-addressed cache keyed by canonical recipe, backend revision, material schema, processing options and dependency versions. Keep source recipes authoritative. Build in isolated worker processes; write artifacts atomically and record per-job diagnostics. Version toolchains and seeds. Do not deduplicate models across incompatible semantic anchors or materials.

## 2. Donor tools and evidence

| Tool | Evidence / decision |
|---|---|
| [Building Tools](https://github.com/ranjian0/building_tools), [usage](https://github.com/ranjian0/building_tools/wiki/Usage), [window](https://github.com/ranjian0/building_tools/wiki/Window), [balcony](https://github.com/ranjian0/building_tools/wiki/Balcony) | Window/door/balcony/railing operators. Some rely on edit-mode face selection. Adapt algorithms rather than UI automation. |
| [Archimesh](https://extensions.blender.org/add-ons/archimesh/), [source](https://github.com/blender/blender-addons/tree/main/archimesh) | Door and window model construction, archviz reference. Pin and test compatibility. |
| [Archipack](https://blender-archipack.org/features), [historical open repo](https://github.com/s-leger/archipack) | Parametric primitives and profile construction. Historical public repo targets old Blender versions; do not assume parity with current commercial edition. |
| [BCGA](https://github.com/vvoovv/bcga), [Prokitektura](https://github.com/nortikin/prokitektura-blender) | Candidate donors for procedural architectural operations. |
| [Blender BMesh](https://docs.blender.org/api/current/bmesh.html), [Geometry Nodes instances](https://docs.blender.org/manual/en/4.5/modeling/geometry_nodes/instances.html), [Realize Instances](https://docs.blender.org/manual/en/4.5/modeling/geometry_nodes/instances/realize_instances.html) | BMesh for connectivity and mesh edits; node groups for complex parametric families. Realize only at individual resource boundaries, not the whole city. |
| [Blender command line](https://docs.blender.org/manual/en/5.0/advanced/command_line/arguments.html), [Cycles baking](https://docs.blender.org/manual/en/4.5/render/cycles/baking.html) | Background worker, Python exit code, selected-to-active baking and margins. Prefer a pinned 4.5 LTS build as initial candidate; test actual SDK/OS compatibility. |
| [xatlas](https://github.com/jpcy/xatlas) | Automatic chart generation and packing for unique bake UVs. |
| [meshoptimizer](https://github.com/zeux/meshoptimizer), [gltfpack](https://meshoptimizer.org/gltf/) | Vertex/index optimization, attribute-aware simplification; preserve semantic metadata separately, and verify output before enabling compression. |
| [MikkTSpace](https://github.com/mmikk/MikkTSpace), [glTF specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html), [glTF Validator](https://github.com/KhronosGroup/glTF-Validator) | Tangent convention and portable interchange validation. |
| [KTX-Software](https://github.com/KhronosGroup/KTX-Software), [KTX guide](https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md) | Portable texture compression candidate; native UNIGINE format/support to be measured. |
| [Material Maker](https://github.com/RodZill4/material-maker) | Procedural PBR authoring; research 06. |

Source survey, not a complete repository audit or executable compatibility certification.

## 3. Asset catalog and recipes

Proposed target catalog (not implemented): windows 24, entrances/doors 12, balconies/loggias 16, railings 10, verandas/terraces/awnings 8, AC/ventilation 12, drainage 6, roof equipment 14, cornice profiles 12, garage doors/shopfront 10 = 124 recipe families/variants. Prioritize common BlackSea types; calibrate dimensions and frequencies against independent regional reference sample. Distinguish balcony (projecting) from loggia (recessed, requires massing/opening geometry). Do not claim typological accuracy from arbitrary parameter ranges.

Example recipe (proposed full schema, not the probe's actual interface):
    schema_version: 1
    generator: window.casement
    width_m: 1.5
    height_m: 1.4
    frame_depth_m: 0.09
    frame_width_m: 0.065
    recess_m: 0.08
    vertical_sections: 2
    material_roles: {frame: pvc_ivory, glass: residential_glass, seal: rubber}
    details: {sill: true, handle: true, bevel_m: 0.003}

Validate constraints (positive dimensions, thickness smaller than available aperture, min/max divisions, collision envelope, profile clearances). Size bins and fixed-thickness profile rebuilds preserve visual dimensions and reuse. Quantize canonical sizes only within declared fitting tolerance, not actual GIS openings. Long profiles: retain end pieces, repeat/resize middle spans. Reject nonuniform scaling for handles and circular parts. Semantic anchors and allowed transforms must be explicit.

AssetManifest fields: asset_id, recipe_hash, generator_revision, schema_version, family, parameters_m, material_roles, local coordinate convention, pivot, mount plane, named sockets, fit/opening bounds, per-LOD AABB, geometric_error_m, triangles, vertex_count, surface_count, UV conventions, dependencies, license/provenance, QA results. Define fit bounds independently of changing visual AABB. Window mounting origin bottom center on facade; +Z outward, +Y up in glTF right-handed space. Roof modules use base-center attachment; slope-aware flashing adapter. UNIGINE axis conversion and normal/tangent handedness must be explicit.

ArchitecturePlan placement uses stable semantic ID, asset ID, transform, dimensions, optional material variation, LOD policy. Preserve apertures and facade alignment across LOD. No per-building independent copies of identical meshes; instance reuse keyed by actual geometry and pipeline state. Color variation should not multiply materials unless engine constraints require it.

## 4. UV and baking

Use analytical metric UV for planar surfaces and extruded profiles; shared trim-sheet UV for repetitive frame/metal/concrete; unique non-overlapping UV atlas (xatlas/Blender) for baking. The probe only provides independent face-local metric UV: overlaps are deliberate for tiled materials and are NOT a bake atlas. Define seam and tangent policy per normal-map UV set; export tangents or generate consistently. Hard edges and UV seams can require duplicate vertices. Preserve physical texel density; adapt atlas resolution and padding to mip count and compression. Initial proposed packing targets 256–512 px/m for close modules, lower for distant facade proxies; tune by screen coverage, not fixed rule. Base color uses sRGB textures; normal/roughness/metallic/AO are non-color data. glTF metallic-roughness uses roughness G and metallic B; occlusion R can share image where compatible. UNIGINE channel mapping must be verified.

Baking worker: generate high and low meshes from same recipe; select high, activate low, create target image with UV atlas and material node; selected-to-active bake tangent normal and AO with appropriate cage/ray distance; bake base color without directional lighting, material channels separately; pad/dilate chart borders; save textures; reconnect low-poly materials; test visual match in controlled illumination. Do not bake sun shadows into albedo. Preserve silhouettes in geometry, use maps for screws, small seams, grill detail. Avoid cage hits from adjacent parts by exploding or matching named bake groups. No baking was executed in research 05.

Candidate Blender invocation:
    blender --background --factory-startup --python-exit-code 1 --python build_asset.py -- --recipe window.json --output generated/
build_asset.py is a future production worker, not included. Pin Blender and render configuration before comparing images.

## 5. External resources and AI

[Poly Haven](https://polyhaven.com/) assets CC0; [API](https://polyhaven.com/our-api) and [API terms](https://github.com/Poly-Haven/Public-API/blob/master/ToS.md) separately govern service access. [ambientCG](https://ambientcg.com/), [license](https://docs.ambientcg.com/license/), [API](https://docs.ambientcg.com/api/) CC0 assets, programmatic retrieval. [OSM2World default style](https://github.com/tordanik/OSM2World-default-style) and [osm2city-data](https://gitlab.com/osm2city/osm2city-data/) semantic material references; verify per-resource rights. Do not publish unverified assets in public PCG. Maintain provenance and checksums even for CC0.

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (4B parameters; repo states MIT for model/code, separate dependency terms) and [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (shape and PBR synthesis; [license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE) excludes EU, UK and South Korea, so not default candidate) are research candidates for ornaments, statues and complex decorative details. No model weights downloaded, no GPU inference performed. AI results require dimension calibration, cleanup/remesh, UV, baking, silhouette evaluation and semantic anchors. Do not use them as authority for hidden geometry or exact dimensions. Deterministic geometry wins for regular windows, rails, pipes and HVAC shells.

## 6. LOD and performance

LOD0: detailed geometry where visible (frame profile, rails, HVAC grille, cap). LOD1: retain architectural silhouettes and primary divisions, remove small handles, seals, bars and sub-millimeter detail; baked normals for removed detail. LOD2: silhouette proxy, facade shader or baked appearance; retain roof equipment if silhouette important. LOD3 at city scale: merge/disappear by projected error, not arbitrary per-asset distance. Use stage 03 screen-space error and hysteresis; select LOD from projected detail size and measured geometry error. Near windows should have matching facade openings; far windows can be material masks. Transparent glass at scale can cause sorting/overdraw; prefer opaque reflection proxy or tested interior mapping for distant LOD. Do not promise engine-specific instancing or bindless until SDK investigation.

Proposed QA gates: finite attributes, valid indices, area/winding/normal/tangent checks, self-intersection where solid is required, allowed contacts and no visible z-fighting, UV seam/overlap rules by channel, bounds and opening fit, anchors stable across LOD, monotonic triangle counts, measured silhouette/Hausdorff or projected error, texture padding and channel sanity, Khronos validator, automated reference renders (front/side/three-quarter, day/night, multiple distances), native import and GPU profiling. Mesh watertightness is required for individual closed solids/collision proxies, not necessarily assembled visual modules; intersection between intentionally touching parts is allowed. gltfpack may change scene structure/custom attributes; preserve placement metadata outside optimized mesh and test round-trip.

## 7. Executed experiment

Files in [05_asset_factory](05_asset_factory/):
- [asset_probe.mjs](05_asset_factory/asset_probe.mjs) — original deterministic geometry generator and glTF exporter; dependency-free JS core, Node.js CLI.
- [test_results.json](05_asset_factory/test_results.json) — 40 recipes × 3 LOD, all 120 checked.
- [gallery_lod0.gltf](05_asset_factory/gallery_lod0.gltf), [gallery_lod1.gltf](05_asset_factory/gallery_lod1.gltf), [gallery_lod2.gltf](05_asset_factory/gallery_lod2.gltf) — self-contained galleries (8 meshes each, embedded base64 buffers). Each gallery uses middle size variant v2. Individual assets generated by CLI.

Run:
    node research/05_asset_factory/asset_probe.mjs generated/assets

CLI writes 24 individual glTF assets, three galleries and test_results.json. Gallery meshes use meters, Y-up, +Z outward, translations on a 4 m grid for review. Asset-local geometry retains the mount origin. Probe uses boxes and polygonal cylinders grouped by material; real frame bars and rail elements, simple opaque colored glass, constant PBR factors, no actual texture maps. Geometry was executed in a JS runtime; CLI wrapper is provided but Node CLI itself was not separately launched.

| Default asset | LOD0 tris | LOD1 tris | LOD2 tris |
|---|---:|---:|---:|
| Casement window | 156 | 96 | 12 |
| Sliding door | 204 | 120 | 12 |
| Open balcony | 444 | 168 | 24 |
| Glazed balcony | 240 | 156 | 24 |
| AC unit | 308 | 92 | 12 |
| Chimney | 72 | 36 | 24 |
| Roof vent | 260 | 84 | 52 |
| Entrance door | 96 | 60 | 12 |
| TOTAL | 1780 | 812 | 172 |

LOD1 reduction vs LOD0 54.4%; LOD2 90.3%. These are triangle-count comparisons, not frame-time benchmarks. All 120 generated configurations passed finite attributes, index validity, nondegenerate triangles, normal length and face winding consistency, nondegenerate triangle UV, exact repeat generation and nonincreasing triangles across LOD. Readback of GitHub files verified JSON parse, embedded buffer size and bufferView extent, eight meshes/nodes each and matching triangle counts. Full external glTF validator and importer not run. No visual render inspected.

Known shortcomings: independent closed primitive parts overlap; no union manifold/self-intersection or z-fighting validation; flat normals and no bevels; opaque glass proxy; no bake atlas, tangents, textures or transparent materials; LOD2 is simple geometry rather than baked representation; some LOD AABBs differ materially (chimney height 1.345 m vs 1.140 m; AC support disappears and min Y changes 0 to 0.09 m). Preserve canonical fit anchors and major silhouette; add bounds/SSE checks before production. No visual quality, geometry performance, UNIGINE import or regional accuracy established. Current code is a research probe, not production core. JS chosen to execute and publish self-contained experiment within available environment, not as mandated language for production.

## 8. Codex implementation sequence and acceptance

1. Pin toolchain; implement recipe schema, validators, stable IDs, canonical dimensions, anchors and content hashes.
2. Implement robust profile sweep and mesh builders with fixed thickness, analytic UV and explicit material roles; baseline window, door, balcony, HVAC, chimney, vents.
3. Add compatible opening/facade adapter; validate cross-element collisions and avoid coplanar overlaps.
4. Add Blender headless worker for bevels, complex meshes, unique UV, baking and preview images. Separate queue, cache and error handling.
5. Generate LOD from same recipe, preserve canonical envelope and anchors; store error metrics; run meshoptimizer only after semantic reduction.
6. Establish visual QA contact sheets and engine tests. Test Khronos validator and UNIGINE import, material role mapping and instancing.
7. Expand regional library based on measured reference dimensions, not random variation alone.
8. Research 06: material system, PBR library, facade shader, trim/array strategy, night windows and weathering.

Initial choices (4.5 LTS, 124 candidate recipes, UV densities, module counts and LOD strategy) are proposed engineering decisions, subject to measurements. No Sandworm dependency.
