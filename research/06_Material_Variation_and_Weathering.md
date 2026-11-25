# Research 06 — Multi-scale facade materials, anti-tiling and weathering

Date: 2026-11-26 | UNIGINE 2.22 | Flight simulator | Research proposal, not production specification

Companion to [Materials and Shaders](06_Materials_and_Shaders.md), [Architecture Grammar](04_Architecture_Grammar.md) and [Rendering Optimization](08_Rendering_Optimization.md).

## 1. Decision and motivation

For ordinary residential, commercial and industrial buildings, prefer **shared tileable PBR surface materials + semantic architectural grammar + reusable architectural modules + multi-scale variation**. Do not make a complete photographic facade, a unique texture per building, or a separate material instance per building the default. Full facade strips or baked module combinations remain useful as a bounded shared far-LOD representation. Unique textures/models are reserved for exceptional landmarks where grammar cannot reproduce the shape.

A facade is not a material: a material describes what a surface is made of; grammar defines where each material and architectural element belongs. A small library of 20–30 base material recipes can produce many coherent facade compositions if paired with a layout grammar and stable variation. This is a target for testing, not a demonstrated sufficiency threshold.

Two failure modes must be addressed separately:

- A strongly variable tile (large stains, prominent cracks) creates recognizable repeated landmarks when repeated across a wall.
- A uniform tile loses its fine texture in the mip chain and becomes a flat color at flight distances. Increasing resolution alone does not solve this.

Design the base material to carry local physical detail, and introduce building-scale heterogeneity independently. Do not bake strong large-scale dirt or lighting into the repeating base tile.

## 2. Proposed shared library

Start with 30 base recipes (illustrative, refine using regional evidence):

| Family | Count | Examples |
|---|---:|---|
| Plaster/render | 5 | smooth, coarse, mineral, painted, aged |
| Brick | 4 | red, pale, dark, silicate |
| Concrete | 3 | cast, precast panel, exposed aggregate |
| Stone | 2 | masonry, dressed stone |
| Wood | 2 | painted siding, natural siding |
| Wall metal/cladding | 2 | corrugated, sandwich panel |
| Roof | 10 | clay/concrete tile, standing seam, profiled painted metal, membrane, bitumen, slate, gravel and regional alternatives |
| Glass/trim bases | 2 | glazing and painted frame/trim |
| Total | 30 | recipes, not unique runtime materials |

Plinths, trims, glass and doors can reuse compatible bases. Additional shared window/door/vent motifs and masks are separate small assets; do not count them as unique building textures. Use realistic material combinations, e.g. brick with stone/concrete plinth, plaster with painted trim, industrial metal with concrete plinth. A combination count is not evidence of perceptual diversity or architectural validity.

PBR sources: base color (sRGB), tangent-space normal (linear, explicit convention), roughness/metalness/AO (linear packed map), optional height (near only). Painted metal is generally dielectric at the coating, exposed metal is metallic. Avoid baked sun, cast shadows and reflections in base color. AO applies to indirect light, not arbitrary multiplication of all lighting. Dirt and moisture can alter roughness and color, not indiscriminately metalness.

Use metric UVs, independent of mesh size. A 1024² image repeated every 2 m provides 512 texels/m; a 1024² image spanning 4 m provides 256 texels/m. 1024² is a reasonable starting resolution for base tiles; 512² may be adequate for distant/low-frequency materials, 2048² only where tests show a benefit. Window/trim atlases can be 1024–2048² with measured texel density. Repetition is governed by physical scale and distinct features, not resolution alone.

## 3. Variation across spatial frequencies

| Scale (illustrative) | Content | Representation |
|---|---|---|
| Micro, mm–cm | pores, grain, mortar, scratches | base tile normal/albedo/roughness |
| Meso, 0.1–3 m | individual bricks, panel color, local stains | structured variation, shared masks, optional stochastic sampling |
| Macro, 2–30 m | facade mottling, uneven repainting, runoff, lower-wall dirt | seeded low-frequency noise or one shared mask sampled in stable coordinates |
| Whole building/block | paint palette, age, maintenance, roof tone | semantic profile and stable building/block seed |
| Sparse accents | localized repair, graffiti, strong leaks | semantic patch/mask or exceptional decal |

The ranges overlap intentionally; calibrate them for material families and flight distances. Building-level random UV offsets and tint reduce inter-building similarity but **do not eliminate periodic repetition within a wall**. Large-scale variance must survive mip filtering, while subpixel detail must average out cleanly. Never amplify fine detail merely to keep it visible from afar.

### Anti-tiling candidates

1. Baseline: repeated metric tile + stable tint/UV offset per building or surface. Maintain seam phase at corners when appropriate.
2. Recommended first upgrade: one shared low-frequency macro mask (or cheap band-limited noise), sampled at a different physical scale and seed. Modulate base color and roughness gently; preserve actual brick/mortar structure.
3. Dual-scale modulation/blend: combine detail with an independent low-frequency field. Do not rescale and blend brick courses or roof tiles at unrelated scales, which creates ghosting and destroys physical proportions.
4. Stochastic hex/triangle tiling or histogram-preserving randomized blending: appropriate for approximately isotropic plaster, concrete and similar surfaces. Match sampling transforms across PBR channels, rotate/reflection-transform tangent normals consistently, preserve derivatives and mip gradients. Profile multiple texture fetches rather than assuming free quality.
5. Wang tiles: non-periodic compatible-edge tiling; useful if texture structure demands continuity. Requires suitable source tiles and more asset preparation.
6. Texture bombing: irregularly place sparse shared mask stamps for stains/patches, not as a universal brick pattern generator. Bound sample count and filter properly.

Directional textures (brick courses, wood siding, profiled sheet metal, standing seams, tile rows) must preserve orientation, physical scale, and continuity. Prefer per-element tint, constrained offsets, or macro modulation over arbitrary rotation. Roof coordinates follow slope/drainage. No runtime neural inference.

## 4. Architectural grammar and module library

Compiler chooses building archetype, floors, facade roles and compatible materials before generating render geometry. Separate base materials from window/door/trim/roof-detail motifs, which may use small shared atlases or texture arrays. A regular facade can be represented by a floor/bay grid plus a compact set of exceptions. Irregular architecture can use explicit coarse geometry or module placements. Avoid a fragment shader that interprets an unbounded recursive grammar.

Suggested semantic roles: main wall, foundation/plinth, gable, ground floor, upper floor, top floor, window, frame, door, storefront, cornice, balcony, roof, roof edge. Use geometry for silhouette, significant depth and near-field relief; small trim and opening appearance can use atlas samples or analytic masks. Preserve window alignment and number of floors. Do not place doors on upper floors or windows on arbitrary roof/gable surfaces.

Use block/regional context to correlate age, materials, palette, roof types and style; adjacent buildings should be plausible together, not independent random combinations.

## 5. Weathering without per-building decal objects

Use a bounded weathering function in the primary facade material. Candidate inputs: building/surface seed, local height above foundation, local facade coordinates, orientation, material family, maintenance/age profile, semantic distances or masks for sills, eaves and drainage. These are offline compiler outputs or inexpensive analytic quantities, not runtime ray queries.

| Effect | Suggested mask / method |
|---|---|
| Ground-contact dirt | decaying height mask modulated by shared noise; allow clean/maintained buildings |
| Rain streaks | vertical anisotropic shared mask with stable offsets; anchor to sill/eave/drain locations when known |
| Uneven aging | broad macro noise in tile/building-local coordinates |
| Repainting / repairs | sparse patch masks with changed tint and roughness |
| Dampness / algae | region/exposure-dependent mask, often near drainage or shaded areas |
| Panel discoloration | stable per-panel hash, bounded by architectural grid |
| Roof weathering | slope-oriented runoff field, broad color variation, seam-aware local detail |

Illustrative shader formulation (not validated UUSL):

```text
base = SampleBasePBR(material_id, metric_uv)
macro = SampleSharedMask(local_position, seed, macro_scale)
dirt = saturate(ground_contact * dirt_strength
               + runoff_mask * runoff_strength
               + patch_mask * age_strength)
albedo_linear = lerp(base.albedo * tint * macro_modulation, dirt_color, dirt)
roughness = clamp(lerp(base.roughness, dirt_roughness, dirt), 0, 1)
```

Choose dirt colors and strengths by material/profile. Avoid applying wall dirt to glazing, and avoid interpreting procedural AO as sunlight. No arbitrary per-pixel expensive noise stacks at far LOD. Keep pattern stable under world-origin rebasing: use tile-local coordinates plus stable offsets/IDs; don't hash camera-relative positions. Use consistent corner behavior and explicit physical source heights.

### Decals

UNIGINE supports deferred material projection with decal material masks. Real decals are suitable for exceptional local features (graffiti, a specific leak, near-field repair, landmarks). They are not the default mechanism for ordinary building aging: additional scene nodes/projection passes, screen coverage, overdraw, culling and management cost need profiling. A shared mask composited inside the base shader can look like a decal without a separate decal node. Alternatively use bounded overlay geometry only when that has a demonstrated cost/quality advantage. No blanket decal per building. No numerical claim about decal performance until measured.

## 6. Runtime contract and LOD

Keep material IDs distinct from engine material instances. Group shared maps into arrays by resolution/format, stable indices across channels; group surfaces spatially and by compatible pipeline. Material changes within a tile must not automatically create separate draw calls for every building. Supply stable parameters using a supported compact mesh attribute or instance-data path. See Research 06 for the UNIGINE 2.22 custom-parameter caveat: access during the original material pass is not established by post-effect examples. Verify native array sampling, attribute packing, per-instance support, shader passes and import conventions with SDK before committing.

| LOD | Proposed representation |
|---|---|
| Near | Shared full PBR, module details or geometry, constrained weathering, optional sparse decals |
| Medium | Low-poly shell and roofs, shared material arrays, simple window grid/motifs, one macro/weathering sample |
| Far | Shared baked archetype strips or simplified filtered analytic layout; mean material color and macro variation, reduced map sampling |
| HLOD | Spatially batched geometry or shared coarse representations; preserve roof/ground contrast and regional appearance |

Use projected feature size and screen coverage, not distance alone. Filter procedural window grids analytically (e.g. fwidth), preserve average emission across LOD, avoid abrupt tint or layout changes. Texture atlas gutters and correct derivatives are required to prevent mip bleeding; test anisotropic filtering and glancing angles. Do not randomly re-seed on LOD transition.

### Memory estimate

30 materials × 1024² × three 8-bpp BC maps (BC7 albedo, BC5 normal, BC7 ORM) with full mip chains ≈ **120 MiB** compressed texture payload. 20 materials ≈ 80 MiB; 30 at 512² ≈ 30 MiB; 30 at 2048² ≈ 480 MiB. A shared 1024² BC4 mask is ≈ 0.67 MiB including mips. Additional module atlases, masks, optional height, allocation alignment, staging, CPU copies and render targets are excluded. This is a calculated payload, not measured VRAM. Array residency/streaming must be measured on target SDK. A 1K tile is reused, not allocated per building.

Example sampling budgets (hypotheses): baseline PBR 3 map lookups; macro breakup adds one shared mask sample; fully stochastic three-neighbor PBR can require nine map samples before masks. Actual hardware transactions, ALU, bandwidth and cache behavior differ; compare real GPU timings and screen coverage.

## 7. Evidence from previous PCG / Condor experiment

- Condor Stage 2 CI: [run 45699306383](https://github.com/kxsxlxv/PCG/actions/runs/45699306383), [research PR #21](https://github.com/kxsxlxv/PCG/pull/21). 49 PCG unit tests passed, six synthetic fixtures, 64 buildings / 1,132 triangles (17.6875 per building), seven material groups, no unique building textures. Courtyard roof projection IoU 1 and zero covered courtyard area after enabling Earcut.
- Original Condor UVs were used with synthetic shared test atlases, CPU diffuse preview only. Native DDS were not available; complex L-shaped roof fell back to flat; no Blender or UNIGINE GPU timings, no demonstrated visual superiority to Sandworm. Triangle budget is promising for distant shells, not an apples-to-apples comparison against detailed PCG/BeamNG meshes.
- Existing [material probe](06_materials/material_probe.py) produced synthetic PBR maps and occupancy tests. It is a useful controlled baseline, not a photorealistic library. The new proposal must remain compatible with the canonical material/geometry contracts and does not select Condor as the default backend.

## 8. Material Variation Lab — proposed controlled experiments

Keep geometry, semantic layout, camera, exposure, light, and base material fixed while changing one mechanism.

| Lane | Change |
|---|---|
| A | Existing shared atlas / base tiling baseline |
| B | Metric tileable PBR + semantic window/door/trim modules |
| C | B + stable tint + one macro mask |
| D | C + semantic weathering (ground/runoff/repair) |
| E | D + stochastic anti-tiling for suitable material families |
| F | D + sparse decals for the same local effects (cost comparison) |

Also compare 512/1024/2048 source resolutions, and separately evaluate plain tile vs dual-scale vs stochastic sampling on identical material patches. Use identical geometry for each A/B pair; separate material and geometry changes. Use brick/plaster/panel/metal/wood and roof fixtures, then matched real districts (Adler/Sochi priority, plus Baltic and other regional samples if authorized data is available). Use native Sandworm output as an external visual control only, not a dependency; compare existing PCG and BeamNG reference lanes.

Camera distances: 10, 30, 50, 150, 300, 1000 m, with fixed FOV and resolution; representative aviation oblique views, top-down, multiple sun directions, night, and moving-camera stability. Record actual generated images/contact sheets, UV orientation and mip/LOD transitions. No mockups as evidence.

Metrics:
- visual preference at matched views, material/architectural plausibility, recognizable repetition and macro variance;
- periodicity proxy: normalized autocorrelation at base-tile lags (exclude intentional brick/window rhythm and report masks separately); spatial band variance at downsampled resolutions; perceptual evaluation still required;
- determinism of seeds and parameter distributions, seam continuity, valid UV/normal handling, no persistent baked lighting, no per-building texture allocation;
- triangle counts, vertices, material groups, draw calls, active nodes, compressed texture bytes, actual residency; SDK CPU/GPU frame time P50/P95/P99, shadow costs and streaming stalls on matched scene/flight route;
- scene sizes 100 / 1,000 / 10,000 / 100,000 buildings as feasible. CPU Lab cannot establish GPU costs; native SDK tests separate.

First bounded next increment: compare A/B/C/D on same low-poly fixtures, using a small shared library and no decals. Add E/F only when baseline quality and cost are known. Require actual visual review before promotion; no default backend or production material changes in this research note. Quality target is better than Sandworm at an agreed comparable performance budget, not merely successful CI.

## 9. Technical references

These are technique references, not claims of equivalent quality in PCG.

- [Krecklau & Kobbelt, Realtime Compositing of Procedural Facade Textures on the GPU (2011)](https://www.graphics.rwth-aachen.de/publication/0326/) — shared architectural tiles, GPU composition and randomization.
- [X-Plane Facade Overview](https://developer.x-plane.com/article/facade-overview/) — floor/panel grammar and repeat/non-repeat logic.
- [FlightGear osm2city texture metadata](https://wiki.flightgear.org/Osm2city.py_Textures) — physical dimensions, permissible cuts and compatibility metadata.
- [CityGen3D building generator](https://www.citygen3d.com/buildings) — blueprint and shared facade atlas concepts.
- [Cohen et al., Wang Tiles for Image and Texture Generation (2003)](https://graphics.uni-konstanz.de/publikationen/Cohen2003WangTilesImage/index.html) — aperiodic tiling with compatible edges.
- [Mikkelsen, Practical Real-Time Hex-Tiling (2022)](https://jcgt.org/published/0011/03/05/) — stochastic sampling candidate.
- [Burley, On Histogram-Preserving Blending for Randomized Texture Tiling (2019)](https://jcgt.org/published/0008/04/02/) — preserving texture statistics.
- [Lutz et al., Preserving the autocovariance of texture tilings using importance sampling (2023)](https://doi.org/10.1111/cgf.14766) — structured correlations and reduced-fetch blending candidate.
- [GPU Gems, Texture Bombing](https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-20-texture-bombing) — sparse pattern placement and sampling costs.
- [GPU Gems 3, Generating Complex Procedural Terrains Using the GPU](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu) — combining texture projection and multi-scale procedural variation.
- [UNIGINE 2.22 decal abstract material](https://developer.unigine.com/en/docs/2.22/code/materials_shaders/abstract_materials/decal), [mesh abstract material](https://developer.unigine.com/en/docs/2.22/code/materials_shaders/abstract_materials/mesh), [custom parameters](https://developer.unigine.com/en/docs/2.22/content/materials/custom_parameters/) — implementation candidates; native verification required.

## 10. Status and boundaries

This file records research conclusions and test proposals. No new renderer, material library, shader, native integration, actual compression or benchmark is implemented by this document. Material counts, scale bands, and sample budgets are provisional engineering choices. No external imagery or third-party assets are included. Existing production contracts and decisions in docs/ remain normative.
