# Research — Zero-geometry facades and spatial batching

Status: research proposal, UNIGINE 2.22. Companion to [Research 06](06_Materials_and_Shaders.md), [Research 08](08_Rendering_Optimization.md), and [SDK acceptance](09_integration/SDK_ACCEPTANCE.md). Production contracts in docs/ remain normative; proposals below require measured acceptance before promotion.

## 1. User requirements

For ordinary mass buildings, including the closest LOD:
- Windows, doors, frames and sills are represented by texture/material effects only. Zero extra triangles, zero separate opening meshes/quads, zero individual surfaces or nodes per opening, zero per-window lights.
- No unique image textures or material instances per ordinary building. Shared tileable PBR (initial target 20–30 base materials), semantic layout masks, window/door motifs, optional parallax/interior illusion, stable procedural night emission and multi-scale weathering.
- Prefer combined geometry and spatial batching, not one object for every building. Geometry describes footprint, roof, foundation and major silhouette volumes, not individual opening detail. Exceptions for landmarks are outside this experiment.
- Maintain regional and architectural plausibility, not independent random selection of incompatible facade elements. Preserve semantic floor/bay plan and stable IDs across LOD and streaming.

Sandworm's city mesh with multiple material surfaces and tiled floor textures is a user observation, not an independently audited implementation. Use as external comparator only; no dependency.

## 2. Facade composition

Keep opaque wall shell (including glass illusion) in deferred material pass. Use separate shared wall PBR tiles, roof PBR tiles and semantic window/door/trim masks/motifs. A floor or bay template describes coverage and the opening type, not a complete wall photograph. Shared window normal/height/roughness maps can represent frame/reveal depth; optional simple parallax only for sufficiently large projected openings. Interior mapping is optional near-field experiment, not a required dependency. No real transparent window surfaces or geometric interiors.

Compare:
- whole wall quad, grid calculated from local facade UVs;
- architecture-zone strips (plinth / ground / repeated upper / top / gable), same window count-independent mesh;
- per-floor wall quads (two triangles per wall floor), allowing floor-specific layouts without geometry for openings.

For a rectangular five-floor building, 8 wall triangles for whole wall, 32 for four strips per side, 40 for per-floor quads, plus roof/base/major volumes. These are idealized counts, not benchmark results. Never create a quad per opening. Exact number of bays chosen offline to fit length; no half windows at corners. UV0 metric material coordinates; UV1 candidate unwrapped facade grid. Non-regular floor bands, courtyard walls, ends and blank industrial walls require explicit grammar. Maintain scale and seams. Doors only where ground/entrance intent permits. No arbitrary windows on roof/gable.

For fragment at grid UV: floor/bay integer cell identifies opening; fractional cell UV selects shared window/door pattern. Base wall material, frame, glass and door are composed using masks. Weathering affects wall separately from glazing. Normal maps and other PBR channels use matching material coordinates and correct tangent transforms. Sample semantic masks with proper filtering (categorical labels must not be linearly interpolated as IDs). Avoid loops over windows/floors and unbounded shader branches. Geometry/UV partition changes are not allowed merely to increase window count.

Night: stable hash of durable building ID (or compiler-derived stable seed), facade ID, floor and room/bay identity. One room can correlate several windows; color, brightness and curtains can vary. Occupancy depends on building type and time of day. No actual sources of light per opening. Emit only on glass mask; do not use a lightmap as arbitrary additive color across walls. At far distance preserve expected linear radiance weighted by glass coverage and occupancy, filter/subpixel average; avoid shimmer and abrupt LOD brightness change. Stable across reload, reorder and origin shift. Window identity must not depend on tile row index or camera-relative coordinates.

## 3. Spatial tile batching and parameter ABI

Compare a city-wide mesh with surfaces per material, tile meshes per material, and tile meshes with texture arrays. One giant mesh reduces object count but can hurt visibility culling, shadow submission, local LOD, streaming, editing and float precision. Start with 128/256/512 m cells as hypotheses; consider 64/1024/2048 at extremes. Assign each complete building to one tile deterministically; compute true bounds including overhangs; avoid duplicate ownership. Keep tile-local float coordinates plus double world origin. HLOD coverage must be exclusive, without double-rendering parent and children.

Proposed active tile groups: facade, roof, optional major detail, split further only for distinct pipeline/LOD/array bank or measured reason. Same shared material may serve all building surfaces in a group. Surface count is not draw-call count (passes, batching, shadows, culling and driver overhead differ). Texture arrays can potentially reduce groups compared with per-base-material surfaces; exact renderer behavior requires native SDK.

Two candidate parameter paths:
1. Packed vertex attributes (UV0/UV1/vertex color) for material ID, style, stable seed or local record index. All three vertices of a triangle must carry identical discrete IDs; split render vertices across semantic boundaries, never interpolate different IDs. Decode UNORM8 using round(x*255), test color space and precision; a single 8-bit channel seed is too small for broad variation. Two bytes can encode up to 65536 local indices; use validated range and reserve sentinel explicitly. Test actual import/export of UV1 and color. A 16-bit local index is not a globally durable ID. Keep stable full seed in table or use compiler-derived stable hash with measured collisions. No silent wrapping.
2. Per-tile data texture or GPU parameter buffer indexed by local building/facade record. Records contain stable seed, material/style IDs, physical grid, tint, age, orientation and lighting profile. Index is ephemeral, stable seed independent of tile rebuild. Candidate data texture point sampling, no sRGB/mips, explicit addressing and bounds; buffer access depends on supported material interface. Prefer fetching per vertex and forwarding constant values where valid and measured, rather than repeating every lookup per pixel. Native compiler must verify varying precision/limits and access in the main geometry pass.

Use arrays for shared PBR channels with consistent layer index, formats and resolution. Independent wall/roof shader families if justified. 64-byte record payload: 10k records 625 KiB, 100k ~6.1 MiB, 1M ~61 MiB; additional facade records, alignment and GPU residency excluded. 30 base 1K materials with three BC8bpp maps and full mipmaps ~120 MiB payload. No linear texture allocation per building.

Native docs: [Mesh abstract material](https://developer.unigine.com/en/docs/2.22/code/materials_shaders/abstract_materials/mesh), [Custom Parameters](https://developer.unigine.com/en/docs/2.22/content/materials/custom_parameters/), [Texture 2D Array](https://developer.unigine.com/en/docs/latest/content/materials/graph/node_library/textures/texture_2d_array), [Culling](https://developer.unigine.com/en/docs/2.22/content/optimization/geometry/culling/). Surface custom parameter is shared by the whole merged surface, not per building. Post-effect lookup examples do not establish availability during original geometry pass. Native GPU tables, per-vertex storage, texture arrays, emission output, actual draw calls and performance remain N0/N1 verification gates.

## 4. Overnight research plan (bounded independent experimental branch)

Use one grouped CI per coherent increment, with several comparable lanes. Prefer reuse of existing PCG geometry, material recipes, architecture plan, CPU raster and occupancy code; audit before reuse, isolate experimental probes. No changes to production src/ or docs/ contracts, no new mandatory dependencies, no native GPU or paid runners. Inspect exact head logs, downloadable PNG/GLB/JSON and actual images before drawing conclusions. If CI fails, diagnose and fix rather than advancing. Never merge research automatically. No generated binaries or unlicensed images committed.

| Batch | Related concepts to test in one CI | Evidence |
|---|---|---|
| 1 | Whole-wall vs zone/floor geometry; merged per-material vs shared group; vertex attribute packing and stable ID table; same geometry with different window density | Triangle/vertex/group counts, exact decode and overflow gates, geometry-derived day/night previews, stable hash tests |
| 2 | Same mesh, semantic window/door coverage over base material; tint, macro variation, ground dirt and runoff; compare repeating floor atlas baseline | Contact sheets at matched camera, no new opening triangles, material continuity and periodicity diagnostics |
| 3 | Window night variation: deterministic room correlation, time profiles, glazing-only emission, subpixel filter and radiance conservation | Day/dusk/night images, LOD stability, linear energy metrics and no geometry changes |
| 4 | Spatial grouping sizes 128/256/512 m, 1k/10k/100k synthetic buildings if feasible, ownership, bounds, frustum proxy, buffer sizes and per-material vs array grouping | CPU counts, estimated submissions and bytes, synthetic visibility, measured offline build time; no FPS/DIP/GPU claims |
| 5 | Material LOD comparison, 512/1024/2048 tiles if useful, shader sample-count model, optional simple parallax on CPU or supported existing renderer; review against known references | Matched actual renders, explicitly label CPU model limitations; no invented native timings |
| 6 | Consolidate findings, regression tests, compare lanes, choose next bounded experiment and write SDK acceptance checklist | Commit/run references, rejected options, measured vs proposed values, visual review notes |

If a batch is blocked, record reason and switch only to another independent safe research task; prioritize first failing gate. Stop at N0 SDK or unauthorized data/runner requirement. Schedule hourly checkpoints for the night; when a short CI finishes within the current run, inspect immediately and batch subsequent work where possible. No promise of persistent live execution; scheduled invocations may be delayed, and exact number of completed batches is not guaranteed. At each checkpoint record exact commit/run, results, artifact links, blockers and next single batch. Do not claim an image inspected unless actually opened. 

## 5. Acceptance rules

- Same input geometry and camera for material comparisons; opening count cannot change triangle or vertex count (within the same wall representation).
- Zero opening-specific geometry/objects/surfaces/lights. No per-building unique texture or material instances.
- Stable deterministic IDs, correct UV and byte decoding, explicit overflow/rejection, no interpolation of distinct IDs. Geometry preserves footprint/holes and actual bounds. No duplicates or missing buildings.
- Real render from generated/exported geometry; label CPU diffuse vs genuine PBR/SDK. Do not use image generation to illustrate results as measured.
- CPU reference does not establish native shader compilation, GPU timing, native draw-call count, batching/culling granularity, memory residency or texture format compatibility. Native N0/N1 must measure on target workstation with same scene and flight route.
- No final backend decision based on synthetic triangle counts alone. Photorealism and superiority over Sandworm remain unproven.

## 6. Reference-driven scope correction (2026-11-26)
User reference photographs indicate roof service volumes, chimneys, vents, parapets, eaves, balcony stacks, facade-specific structural zones, panel joints, heterogeneous glazing and permanent localized material aging are higher-priority than expensive virtual interiors. The photos are qualitative references only; do not publish or derive textures from them without authorization. **Snow cover, icicles and dynamic weather/season overlays are explicitly outside PCG scope; UNIGINE owns them.** Retain permanent weathering (corrosion, dirt, cracks, repairs and coating degradation). Geometry only for silhouette/major architectural volumes; no added window/door meshes or lights. Compare actual fixed-camera GLB lanes and measure cost before promoting any budget.
